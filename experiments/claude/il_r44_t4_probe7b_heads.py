"""Round 44, Task 4: P_7b probe-head candidates on the cached DINOv3-7B @ 144x272 features (laptop GPU, 0 Colab CU).

    python experiments/claude/il_r44_t4_probe7b_heads.py --smoke        # ~1 min end-to-end check on 600 rows
    python experiments/claude/il_r44_t4_probe7b_heads.py                # train the four candidates (resumable), then judge
    python experiments/claude/il_r44_t4_probe7b_heads.py --judge-only   # re-run the judge on saved candidate files

Everything except the head is the v41 P_7b recipe (scripts/train_task4_probe_7b.py): 16,384-d features (mid_patch, last_patch,
last_cls, last_std), per-subblock + per-backbone L2-norm, fold standardisation from the `full` training rows, trained on the four
views full / zoom70 / centre_grey / dark_lc, Feature-Space Mixup p=0.8 Beta(0.4, 0.4), label smoothing 0.1, AdamW 1e-3 / wd 0.05,
30 epochs, batch 512, pipeline folds (t4_cnxt_fold*.npz val_idx).

Candidates (fixed before the run):
  S10  the v41 head with 10 seeds: seeds 3..9 trained here and averaged with the cached 3-seed P_7b (seeds 0..2), i.e. the exact
       10-net softmax mean (OOF from ii_probe_v3_views.npz, test from t4_probe7b_fold*.npz)
  RES  residual MLP head (Linear 16384->1024, LN, GELU, + one pre-LN residual block 1024->1024->1024, LN, Linear 1024->5), 3 seeds
  SWI  SwiGLU head (Linear 16384->2x1024, silu(g) * v, LN, Linear 1024->5), 3 seeds
  SBD  the v41 head + sub-block dropout (each 4096-d sub-block of a training row zeroed with p=0.25, inverted scaling), 3 seeds
Preflight: the v41 head with seeds 0..2 on fold 0 must reproduce the cached P_7b OOF (mean |diff| < 5e-4, argmax agreement >= 0.999).

Judge (fixed before the run; same members, gate, slices and rule as ij_r43_t4_reweight_v41.py):
  v41 with P_7b replaced by candidate C at PW in {0.15, 0.20, 0.25} -> 4 x 3 = 12 settings. Nested N: in each held-out fold, the
  setting with the best mean (zoom70, centre_grey, dark) delta among those meeting the rule on the other four folds, else v41.
  N passes vs v41 if zoom70 > 0, centre_grey > 0, dark/low-contrast > 0, full >= -0.0005 and the zoom / HUD mean is > 0 in >= 3/5
  folds. If N passes, the pipeline setting is the best in-sample setting meeting the rule (as in ij_r43). Test probabilities are
  computed for each fold model but never read by the judge.
"""
import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, str(Path("scripts").resolve()))
import train_task4_probe_7b as p7  # noqa: E402
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

DEV = p7.DEV
OUT_DIR = Path("experiments/claude/outputs/r44")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "il_probe7b_heads.log"
JUDGE_LOG = OUT_DIR / "il_probe7b_heads_judge.log"
P7B_VIEWS = Path("experiments/claude/outputs/r42/ii_probe_v3_views.npz")
EVAL_VIEWS = ["full", "zoom70", "centre_grey"]
SBD_P = 0.25
CANDS = {  # name -> (head, seeds, sub-block dropout)
    "SBD": ("v41", (0, 1, 2), SBD_P),
    "RES": ("res", (0, 1, 2), 0.0),
    "SWI": ("swiglu", (0, 1, 2), 0.0),
    "S10": ("v41", (3, 4, 5, 6, 7, 8, 9), 0.0),
}
PW_GRID = (0.15, 0.20, 0.25)
assert len(set(p7.SUBBLOCK_DIMS)) == 1


class ResHead(nn.Module):
    def __init__(self, dim, h=1024):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.25), nn.Linear(dim, h), nn.LayerNorm(h), nn.GELU())
        self.blk = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(0.5), nn.Linear(h, h))
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Dropout(0.5), nn.Linear(h, 5))

    def forward(self, x):
        z = self.inp(x)
        return self.out(z + self.blk(z))


class SwiGLUHead(nn.Module):
    def __init__(self, dim, h=1024):
        super().__init__()
        self.drop_in, self.fc, self.norm = nn.Dropout(0.25), nn.Linear(dim, 2 * h), nn.LayerNorm(h)
        self.drop, self.out = nn.Dropout(0.5), nn.Linear(h, 5)

    def forward(self, x):
        g, v = self.fc(self.drop_in(x)).chunk(2, dim=-1)
        return self.out(self.drop(self.norm(F.silu(g) * v)))


def make_head(kind, dim):
    if kind == "v41":   # identical to scripts/train_task4_probe_7b.py (same init order -> seeds 0..2 reproduce P_7b)
        return nn.Sequential(nn.Dropout(0.25), nn.Linear(dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.5), nn.Linear(1024, 5))
    if kind == "res":
        return ResHead(dim)
    if kind == "swiglu":
        return SwiGLUHead(dim)
    raise ValueError(kind)


def fit_nets(A_tr_f16, Y_tr, mu, sd, head, seeds, sbd=0.0, epochs=p7.EPOCHS, p_mix=0.8, mix_alpha=0.4, label_smoothing=0.1):
    """p7.fit_mlp_p7b with a head choice and optional sub-block dropout (its RNG streams untouched when sbd == 0)."""
    n_rows, bs, sub = len(Y_tr), p7.BS, p7.SUBBLOCK_DIMS[0]
    nsub = A_tr_f16.shape[1] // sub
    mu_gpu = torch.from_numpy(mu).to(DEV, dtype=torch.float32)
    inv_sd = float(1.0 / sd)
    on_gpu = False
    try:
        A_t = torch.from_numpy(A_tr_f16).to(DEV)
        Y_t = torch.from_numpy(Y_tr).to(DEV)
        on_gpu = True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        A_t, Y_t = torch.from_numpy(A_tr_f16), torch.from_numpy(Y_tr)
    spe = math.ceil(n_rows / bs)
    total_steps = epochs * spe
    nets = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        g_cpu = torch.Generator(device="cpu").manual_seed(seed)
        g_dev = torch.Generator(device=DEV).manual_seed(seed + 100)
        g_sbd = torch.Generator(device=DEV).manual_seed(seed + 200)
        torch.manual_seed(seed)
        net = make_head(head, A_tr_f16.shape[1]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / spe if s < spe else 0.5 * (1.0 + math.cos(math.pi * (s - spe) / max(1, total_steps - spe))))
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n_rows, generator=g_cpu)
            if on_gpu:
                perm = perm.to(DEV)
            for i in range(spe):
                idx = perm[i * bs:(i + 1) * bs]
                xb = A_t[idx].to(DEV, dtype=torch.float32)
                xb.sub_(mu_gpu).mul_(inv_sd)
                yt = F.one_hot(Y_t[idx].to(DEV), 5).float()
                if rng.random() < p_mix and len(xb) > 1:
                    lam = torch.from_numpy(rng.beta(mix_alpha, mix_alpha, size=(len(xb), 1)).astype(np.float32)).to(DEV)
                    perm_m = torch.randperm(len(xb), generator=g_dev, device=DEV)
                    xb = lam * xb + (1.0 - lam) * xb[perm_m]
                    yt = lam * yt + (1.0 - lam) * yt[perm_m]
                if sbd > 0:
                    keep = (torch.rand(len(xb), nsub, generator=g_sbd, device=DEV) >= sbd).float() / (1.0 - sbd)
                    xb = xb * keep.repeat_interleave(sub, dim=1)
                yt = yt * (1.0 - label_smoothing) + label_smoothing / 5.0
                loss = (-yt * F.log_softmax(net(xb), dim=-1)).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        nets.append(net)
        del opt, sched
    del A_t, Y_t, mu_gpu
    return nets, on_gpu


def train_folds(mm, y, folds, test_len, head, seeds, sbd, epochs, tag, log, only_folds=None):
    n = len(y)
    pv = {v: np.zeros((n, 5), dtype=np.float64) for v in EVAL_VIEWS}
    tps = np.zeros((len(folds), test_len, 5), dtype=np.float64)
    for f, va in enumerate(folds):
        if only_folds is not None and f not in only_folds:
            continue
        t_f = time.time()
        tr = np.setdiff1d(np.arange(n), va, assume_unique=True)
        A_tr = p7.build_train_matrix(mm, tr)
        mu, sd = p7.compute_fold_stats(A_tr, len(tr))
        Y_tr = np.tile(y[tr], len(p7.TRAIN_VIEWS))
        if DEV == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        nets, on_gpu = fit_nets(A_tr, Y_tr, mu, sd, head, seeds, sbd, epochs=epochs)
        peak = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0
        del A_tr, Y_tr
        gc.collect()
        for v in EVAL_VIEWS:
            pv[v][va] = p7.predict_rows(nets, mm[f"train_{v}"], va, mu, sd)
        tps[f] = p7.predict_rows(nets, mm["test_full"], np.arange(test_len), mu, sd)
        del nets
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()
        log(f"  [{tag}] fold {f}: {len(seeds)} seeds x {epochs} ep in {time.time() - t_f:.0f}s | on_gpu={on_gpu} | peak {peak:.2f} GiB"
            f" | full val acc {np.mean(pv['full'][va].argmax(1) == y[va]):.4f}")
    return pv, tps


def save_cand(path, pv, tps, **meta):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{v: pv[v].astype(np.float32) for v in EVAL_VIEWS}, test_folds=tps.astype(np.float32),
                        **{k: np.asarray(v) for k, v in meta.items()})
    tmp.replace(path)


def run_training(y, folds, test_names, log, smoke=False):
    n_test = len(test_names)
    if smoke:
        n, n_t, epochs = 600, 256, 1
        rng = np.random.default_rng(0)
        folds = [np.sort(a) for a in np.array_split(rng.permutation(n), 5)]
        y = y[:n]
        mm = p7.build_l2_memmaps(n, n_t, prefix="_tmp_il_smoke", log=log)
        try:
            for name, (head, seeds, sbd) in CANDS.items():
                pv, tps = train_folds(mm, y, folds, n_t, head, seeds[:1], sbd, epochs, f"smoke {name}", log)
                assert all(np.isfinite(pv[v]).all() and np.allclose(pv[v].sum(1), 1, atol=1e-4) for v in EVAL_VIEWS)
                assert np.isfinite(tps).all() and np.allclose(tps.sum(2), 1, atol=1e-4)
                save_cand(OUT_DIR / f"_smoke_{name}.npz", pv, tps, head=head)
                (OUT_DIR / f"_smoke_{name}.npz").unlink()
        finally:
            p7.cleanup_memmaps(mm)
        log("SMOKE PASS")
        return

    p7.verify_feature_files(len(y), n_test, test_names)
    z3 = np.load(P7B_VIEWS)
    cached = {v: z3[f"p7b_{v}"].astype(np.float64) for v in EVAL_VIEWS}
    todo = [nm for nm in CANDS if not (OUT_DIR / f"il_{nm}.npz").exists()]
    log(f"Candidates to train: {todo or 'none (all saved)'}")
    if not todo:
        return
    mm = p7.build_l2_memmaps(len(y), n_test, prefix="_tmp_il_l2", log=log)
    try:
        pre = OUT_DIR / "il_preflight_ok.txt"
        if not pre.exists():
            pv0, _ = train_folds(mm, y, folds, n_test, "v41", (0, 1, 2), 0.0, p7.EPOCHS, "preflight v41 seeds 0-2", log, only_folds={0})
            va = folds[0]
            d = np.abs(pv0["full"][va] - cached["full"][va])
            agr = float(np.mean(pv0["full"][va].argmax(1) == cached["full"][va].argmax(1)))
            log(f"Preflight parity vs cached P_7b (fold 0, full): mean|diff| {d.mean():.2e}, max {d.max():.2e}, agree {agr:.4f}")
            assert d.mean() < 5e-4 and agr >= 0.999, "preflight parity failed: the training path does not reproduce P_7b"
            pre.write_text(f"mean {d.mean():.3e} max {d.max():.3e} agree {agr:.4f}\n")
        for name in todo:
            head, seeds, sbd = CANDS[name]
            t_c = time.time()
            log(f"=== {name}: head={head}, seeds={seeds}, sub-block dropout={sbd} ===")
            pv, tps = train_folds(mm, y, folds, n_test, head, seeds, sbd, p7.EPOCHS, name, log)
            if name == "S10":   # exact 10-net mean with the cached seeds 0..2
                k_new, k_old = len(seeds), 3
                cached_t = np.stack([np.load(PRED_DIR / f"t4_probe7b_fold{f}.npz")["test_probs"].astype(np.float64) for f in range(5)])
                save_cand(OUT_DIR / "il_S10_new7.npz", pv, tps, head=head, seeds=np.array(seeds))
                pv = {v: (k_old * cached[v] + k_new * pv[v]) / (k_old + k_new) for v in EVAL_VIEWS}
                tps = (k_old * cached_t + k_new * tps) / (k_old + k_new)
            save_cand(OUT_DIR / f"il_{name}.npz", pv, tps, head=head, seeds=np.array(seeds), sbd=sbd)
            log(f"=== {name} saved in {(time.time() - t_c) / 60:.1f} min ===")
    finally:
        p7.cleanup_memmaps(mm)
        log("Cleaned up temporary L2 memmaps.")


def judge(y, folds, log):
    L2, D3, D384, RAD, HP, HP384 = "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384", "cradio3L_384", "dinov3Hp_144", "dinov3Hp_384"
    PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
    V20 = PLAIN + HUDM + ["dinov2_v2"]
    V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
    n = len(y)
    fo = np.zeros(n, int)
    for f, va in enumerate(folds):
        fo[va] = f
    P = {t: {v: np.zeros((n, 5)) for v in EVAL_VIEWS} for t in V20 + [L2, D3, D384, RAD, HP, HP384]}
    for t in P:
        for f, va in enumerate(folds):
            z30 = np.load(V30 / f"{t}_fold{f}.npz")
            z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
            for v in EVAL_VIEWS:
                P[t][v][va] = (z30 if v in z30.files else z19)[v]
    z3 = np.load(P7B_VIEWS)
    PR = {v: z3[f"p7b_{v}"].astype(np.float64) for v in EVAL_VIEWS}
    ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
    GATE = {v: np.zeros(n, bool) for v in EVAL_VIEWS}
    for f in range(5):
        tr, ho = fo != f, fo == f
        tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
        for v in EVAL_VIEWS:
            GATE[v][ho] = (ST[f"brightness_{v}"][ho] <= tb) | (ST[f"contrast_{v}"][ho] <= tc)
    st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
    dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
    pi = np.bincount(y, minlength=5) / n
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)
    allm = np.ones(n, bool)
    SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
    VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
    msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m

    w = {**{t: 0.25 for t in PLAIN}, **{t: 0.75 for t in HUDM}, "dinov2_v2": 8.0, D3: 10.0, D384: 20.0, HP: 40.0, HP384: 40.0}
    gated = {L2: (8.0, 2.0), RAD: (10.0, 2.0)}
    FT = {}
    for v in EVAL_VIEWS:
        num = sum(wt * P[t][v] for t, wt in w.items())
        den = np.full(n, sum(w.values()))
        for t, (wn, wd) in gated.items():
            ww = np.where(GATE[v], wd, wn)
            num, den = num + ww[:, None] * P[t][v], den + ww
        FT[v] = num / den[:, None]
    ref = {v: rule(0.80 * FT[v] + 0.20 * PR[v]) for v in EVAL_VIEWS}
    ref_sc = {k: bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    log("v41 reference: " + ", ".join(f"{k} {ref_sc[k]:.4f}" for k in SC))
    for k, exp_v in {"full": 0.9959, "zoom70": 0.9897, "centre_grey": 0.9826, "dark/low-contrast": 0.9901}.items():
        assert abs(ref_sc[k] - exp_v) <= 1.5e-4, f"v41 mismatch on {k}: got {ref_sc[k]:.5f}, expected {exp_v:.4f}"

    PC = {}
    for nm in CANDS:
        p = OUT_DIR / f"il_{nm}.npz"
        if p.exists():
            z = np.load(p)
            PC[nm] = {v: z[v].astype(np.float64) for v in EVAL_VIEWS}
    missing = [nm for nm in CANDS if nm not in PC]
    if missing:
        log(f"WARNING: candidates missing {missing} -> the decision below is NOT the pre-declared one")

    log("\nAlone (v17 rule, balanced; errors in brackets):")
    for nm, pm in [("P_7b (v41)", PR)] + list(PC.items()):
        log(f"  {nm:11s}: " + ", ".join(f"{k} {bal(rule(pm[VIEW_OF[k]]), msk(k)):.4f}({int((rule(pm[VIEW_OF[k]])[msk(k)] != y[msk(k)]).sum())})" for k in SC))

    SET = {(c, pw): {v: rule((1 - pw) * FT[v] + pw * PC[c][v]) for v in EVAL_VIEWS} for c in PC for pw in PW_GRID}
    name = lambda s: f"{s[0]} @ {s[1]:.2f}"
    deltas = lambda lab, m=allm: {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}
    meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    crit = lambda d: float(np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]]))

    def report(nm, lab):
        d = deltas(lab)
        pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
        pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
        ok = meets(d) and sum(p > 0 for p in pf) >= 3
        log(f"  {nm}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
            + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
            + f" | full err {int((ref['full'] != y).sum())}->{int((lab['full'] != y).sum())},"
            + f" dark err {int(((ref['full'] != y) & dark).sum())}->{int(((lab['full'] != y) & dark).sum())},"
            + f" labels changed (full) {int((lab['full'] != ref['full']).sum())}\n    -> {'PASS' if ok else 'fail'}")
        return ok

    outN, picks = {v: ref[v].copy() for v in EVAL_VIEWS}, []
    for f in range(5):
        okk = {s: crit(d) for s, d in ((s, deltas(SET[s], fo != f)) for s in SET) if meets(d)}
        s = max(okk, key=okk.get) if okk else None
        picks.append(s)
        if s is not None:
            for v in EVAL_VIEWS:
                outN[v][fo == f] = SET[s][v][fo == f]
    log("\nNested choice per held-out fold:")
    for f, s in enumerate(picks):
        log(f"  fold {f}: {name(s) if s else 'v41 (no setting meets the rule)'}")
    log("\nCandidate (decision):")
    okN = report(f"N nested over the {len(SET)}-setting grid", outN)

    log("\nInformation: every setting in-sample")
    for s in SET:
        report(name(s), SET[s])
    ins = sorted(((crit(deltas(SET[s])), s) for s in SET if meets(deltas(SET[s]))), reverse=True)
    best = ins[0][1] if ins else None
    log(f"\n-> N {'PASSES' if okN else 'fails'}; pipeline setting if N passes: {name(best) if best else None}")
    return okN, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--judge-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    log, log_f = p7.make_logger(LOG_PATH if not args.judge_only else JUDGE_LOG, write_log=not args.smoke)
    try:
        train, test = task4.load_frames()
        y = train.label.values.astype(np.int64)
        test_names = np.array(test.file_name.tolist(), dtype="U")
        folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
        if not args.judge_only:
            log(f"il_r44 start (dev={DEV}); candidates {list(CANDS)}; PW grid {PW_GRID}")
            run_training(y, folds, test_names, log, smoke=args.smoke)
        if not args.smoke:
            judge(y, folds, log)
    finally:
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()

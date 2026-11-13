"""Task 5 leak hunt, part 2: id-encoding columns, copied values between rows, exact period ratios, and how much the
block-only values (device_count_4 = 4, guild_changes_1 = 3, guild_changes_4 = 4) add on top of our families + pocket."""
import warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
t = pd.read_csv("05_anomaly_detection/test.csv")
idn = t.player_id.str[1:].astype(int).values
blk = idn >= 23819
cols = t.columns[1:]
# 1) does any column encode the id (sorted order)?
r = []
for c in cols:
    v = t[c].values; ok = ~np.isnan(v)
    r.append((c, spearmanr(idn[ok], v[ok])[0], spearmanr(idn[ok & ~blk], v[ok & ~blk])[0]))
r = pd.DataFrame(r, columns=["col", "rho_all", "rho_outside_block"])
print("largest |Spearman(id, column)| outside the block (noise ~0.012):")
print(r.reindex(r.rho_outside_block.abs().sort_values(ascending=False).index).head(5).to_string(index=False))
# 2) values copied between rows (anomaly made by cloning another account?)
print("\nshare of non-integer values that occur in more than one row (block vs rest), top 5:")
d = []
for c in cols:
    v = t[c]; nonint = v.notna() & (v != v.round())
    dup = v.duplicated(keep=False) & nonint
    if nonint.sum():
        d.append((c, dup[blk].sum() / max(nonint[blk].sum(), 1), dup[~blk].sum() / max(nonint[~blk].sum(), 1)))
d = pd.DataFrame(d, columns=["col", "dup_blk", "dup_rest"]).sort_values("dup_blk", ascending=False)
print(d.head(5).to_string(index=False))
whole = t[cols].round(10).duplicated(keep=False)
print(f"fully duplicated rows: {whole.sum()}")
# 3) exact ratios between period 4 and period 3 (multiplicative injection leaves constant ratios)
print("\nmost common period-4 / period-3 ratio per family (share of block rows hitting it vs rest):")
fams = sorted({c.rsplit("_", 1)[0] for c in cols if c[-2:] in ("_1", "_2", "_3", "_4")})
q = []
for f in fams:
    a, b = t[f"{f}_4"], t[f"{f}_3"]
    ratio = (a / b).round(6).replace([np.inf, -np.inf], np.nan)
    top = ratio[blk].value_counts().head(1)
    if len(top):
        val = top.index[0]
        q.append((f, val, (ratio[blk] == val).mean(), (ratio[~blk] == val).mean()))
q = pd.DataFrame(q, columns=["family", "ratio", "share_blk", "share_rest"]).sort_values("share_blk", ascending=False)
print(q.head(6).to_string(index=False))
# 4) block-only values vs our flags
odd = (t.device_count_4 == 4) | (t.guild_changes_1 == 3) | (t.guild_changes_4 == 4)
sub = pd.read_csv("submission_clean_try.csv")
clean = sub.task5.values == 1
v11 = pd.read_csv("submission_v11_try.csv").task5.values == 1
print(f"\nblock-only values: {odd.sum()} accounts, inside block {(odd & blk).sum()}, outside {(odd & ~blk).sum()}")
print(f"  already flagged by the clean pipeline: {(odd & clean).sum()}; new: {(odd & ~clean).sum()}")
print(f"  of the clean pipeline's 24 missed block accounts, caught by these values: {(odd & blk & ~clean).sum()}")
f3 = lambda flag: 10 * (flag & blk).sum() / (9 * blk.sum() + flag.sum())
print(f"F3 vs id block: clean {f3(clean):.4f}, clean + block-only values {f3(clean | odd.values):.4f}, v11 {f3(v11):.4f}")
for c in ["password_changed", "mass_item_sale", "vpn_usage"]:
    m = t[c].values == 1
    print(f"{c}=1: {m.sum()} accounts, {(m & blk).sum()} in block ({(m & blk).sum() / m.sum():.0%}); not flagged by clean & in block: {(m & blk & ~clean).sum()}")

"""Small shared helpers: seeding, memory optimisation, timing, prediction caching."""
import json
import os
import random
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .config import PARAM_DIR, PRED_DIR


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def quiet():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def reduce_mem_usage(df, verbose=False):
    """Downcast numeric columns to the smallest dtype that holds their range."""
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"memory {start:.1f} MB -> {end:.1f} MB ({100 * (start - end) / start:.0f}% saved)")
    return df


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] {time.time() - t0:.1f}s")


def save_preds(name, **arrays):
    np.savez_compressed(PRED_DIR / f"{name}.npz", **arrays)


def load_preds(name):
    path = PRED_DIR / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_params(name, params):
    (PARAM_DIR / f"{name}.json").write_text(json.dumps(params, indent=2))


def load_params(name):
    path = PARAM_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None

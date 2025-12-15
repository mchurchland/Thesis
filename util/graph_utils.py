import os
import glob
import pandas as pd
import numpy as np

def _safe_path(path: str) -> str:
    """
    Generate a non-clobbering path by appending version suffixes.
    Pattern mirrors common tempfile/pathlib patterns; see:
    https://github.com/python/cpython/blob/main/Lib/tempfile.py
    """
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{root}.v{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def _read_glob(pattern: str) -> pd.DataFrame | None:
    """
    Read and concatenate CSVs matching a glob pattern using pandas
    (https://github.com/pandas-dev/pandas/blob/main/pandas/io/parsers/readers.py).
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if "src" not in df.columns:
                df["src"] = os.path.basename(os.path.dirname(p))  # chunk_x
            frames.append(df)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing columns with defaults and order them (pandas-style align/assign)."""
    needed = ["mode","shuffle_id","rho_target","leak","input_scale","MC","IPC","KR","GR","src"]
    for c in needed:
        if c not in df.columns:
            if c in ("MC","IPC","KR","GR"):
                df[c] = np.nan
            elif c == "shuffle_id":
                df[c] = -1
            elif c == "mode":
                df[c] = "unknown"
            elif c == "src":
                df[c] = "unknown"
            else:
                raise ValueError(f"Missing required column: {c}")
    return df[needed].copy()

def _dispersion_cv(a: np.ndarray) -> float:
    """Coefficient of variation (Sokal & Rohlf, 1995, Biometry 3rd ed.)."""
    a = np.asarray(a, float).ravel()
    m = float(np.mean(a)) 
    s = float(np.std(a))
    return s/(abs(m)+1e-12)


def _dispersion_v(a: np.ndarray) -> float:
    """Standard deviation (variation)."""
    a = np.asarray(a, float).ravel()
    s = float(np.std(a))
    return s

def _unique_hparam_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Average duplicate hyperparameter rows; thin wrapper around pandas groupby/mean."""
    keys = ["rho_target","leak","input_scale"]
    metrics = [c for c in ("MC","IPC","KR","GR") if c in df.columns]
    if not metrics:
        return df.copy()
    return (df.groupby(keys, as_index=False)[metrics]
              .mean()
              .sort_values(keys)
              .reset_index(drop=True))

def _build_combined(df_shuf: pd.DataFrame | None, df_var: pd.DataFrame | None) -> pd.DataFrame:
    """Combine shuffle/variant tables with standardized columns (pandas concat/assign)."""
    parts = []
    if df_shuf is not None and not df_shuf.empty:
        a = df_shuf.copy()
        a.loc[a["mode"] == "real", "mode"] = "CE-real"
        a.loc[a["mode"] == "shuffle", "mode"] = "CE-shuffle"
        parts.append(a)
    if df_var is not None and not df_var.empty:
        parts.append(df_var.copy())
    if not parts:
        raise FileNotFoundError("No input CSVs found for either shuf or variants.")
    comb = pd.concat(parts, ignore_index=True, sort=False)
    # standardize columns and dtypes
    comb = _ensure_columns(comb)
    # canonicalize mode strings (tiny cleanup)
    comb["mode"] = comb["mode"].astype(str)
    return comb

def _compute_dispersion_table(combined: pd.DataFrame,mode: str = "cv") -> pd.DataFrame:
    """
    modes = "cv" or "v" for coefficient of variation or variation
    For each (mode, src, group_id), compute dispersion across hyper-params:
      - group_id = shuffle_id if shuffle_id != -1 else src
    Uses pandas groupby/melt pattern; see https://github.com/pandas-dev/pandas/blob/main/pandas/core/frame.py
    """
    df = combined.copy()
    df["group_id"] = df["shuffle_id"].astype(str)
    df.loc[df["shuffle_id"] == -1, "group_id"] = df["src"].astype(str)

    # dedup repeated measurements within the same hyperparam triple
    keys = ["mode","src","group_id","rho_target","leak","input_scale"]
    metrics = [m for m in ("MC","IPC","KR","GR") if m in df.columns]
    df_agg = (df.groupby(keys, as_index=False)[metrics]
                .mean()
                .sort_values(keys)
                .reset_index(drop=True))

    # melt to long form and aggregate dispersion by group/metric
    df_long = df_agg.melt(id_vars=keys, value_vars=metrics, var_name="metric", value_name="value")
    agg_fn = _dispersion_cv if mode == "cv" else _dispersion_v
    disp = (df_long.groupby(["mode","src","group_id","metric"])
                     .agg(dispersion=("value", lambda x: agg_fn(x.to_numpy())),
                          n_hparams=("value", "size"))
                     .reset_index())
    return disp


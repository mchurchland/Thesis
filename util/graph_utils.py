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
    df = df.copy()
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
    extras = [c for c in df.columns if c not in needed]
    return df[needed + extras].copy()

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

def _assign_group_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Attach group_id to each row (shuffle_id when available, synthetic otherwise)."""
    df = df.copy()
    df["group_id"] = df["shuffle_id"].astype(object)

    # Some variants (e.g., cel+randN, er+randN, ws_p0.1+randN) store shuffle_id=-1
    # even though they were repeated many times. Those repeats are laid out in
    # blocks of one full hyperparameter grid per repeat inside each src chunk.
    mask_no_sid = df["shuffle_id"] == -1
    if mask_no_sid.any():
        hparam_cols = ["rho_target", "leak", "input_scale"]
        hparam_counts = {
            mode_name: sub[hparam_cols].drop_duplicates().shape[0]
            for mode_name, sub in df.loc[mask_no_sid].groupby("mode")
        }
        for (mode_name, src), sub in df.loc[mask_no_sid].groupby(["mode", "src"]):
            block_size = hparam_counts.get(mode_name, 0)
            # If we cannot evenly partition, fall back to src-level grouping.
            if block_size <= 0 or len(sub) % block_size != 0:
                continue
            rep_idx = np.arange(len(sub)) // block_size
            df.loc[sub.index, "group_id"] = [f"{src}_r{r}" for r in rep_idx]

    df["group_id"] = df["group_id"].astype(str)
    return df

def _aggregate_over_hparams(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Deduplicate repeated hparams within (mode, src, group_id) by averaging metrics."""
    metrics = [m for m in metrics if m in df.columns]
    keys = ["mode","src","group_id","rho_target","leak","input_scale"]
    if not metrics:
        return pd.DataFrame(columns=keys)
    return (df.groupby(keys, as_index=False)[metrics]
              .mean()
              .sort_values(keys)
              .reset_index(drop=True))

def _compute_dispersion_table(combined: pd.DataFrame,mode: str = "cv") -> pd.DataFrame:
    """
    modes = "cv" or "v" for coefficient of variation or variation
    For each (mode, src, group_id), compute dispersion across hyper-params:
      - group_id = shuffle_id if shuffle_id != -1 else src
    Uses pandas groupby/melt pattern; see https://github.com/pandas-dev/pandas/blob/main/pandas/core/frame.py
    """
    disp_metrics = [m for m in ("MC","IPC","KR","GR") if m in combined.columns]
    if not disp_metrics:
        return pd.DataFrame(columns=["mode","src","group_id","metric","dispersion","n_hparams"])

    df = _assign_group_ids(combined)
    df_agg = _aggregate_over_hparams(df, disp_metrics)
    if df_agg.empty:
        return pd.DataFrame(columns=["mode","src","group_id","metric","dispersion","n_hparams"])

    keys = ["mode","src","group_id","rho_target","leak","input_scale"]
    df_long = df_agg.melt(id_vars=keys, value_vars=disp_metrics, var_name="metric", value_name="value")
    agg_fn = _dispersion_cv if mode == "cv" else _dispersion_v
    disp = (df_long.groupby(["mode","src","group_id","metric"])
                     .agg(dispersion=("value", lambda x: agg_fn(x.to_numpy())),
                          n_hparams=("value", "size"))
                     .reset_index())
    return disp

def _compute_mean_table(combined: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """
    Compute per-(mode, src, group_id) means for the requested metrics without
    dispersion. If metrics is None, defaults to the standard metrics plus the
    available invariance columns when present.
    """
    df = _assign_group_ids(combined)
    if metrics is None:
        metrics = [m for m in ("MC","IPC","KR","GR","cosine_similarity","wt_mean","mean") if m in df.columns]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return pd.DataFrame(columns=["mode","src","group_id","metric","mean","n_hparams"])

    df_agg = _aggregate_over_hparams(df, metrics)
    if df_agg.empty:
        return pd.DataFrame(columns=["mode","src","group_id","metric","mean","n_hparams"])

    keys = ["mode","src","group_id","rho_target","leak","input_scale"]
    df_long = df_agg.melt(id_vars=keys, value_vars=metrics, var_name="metric", value_name="value")
    mean_tbl = (df_long.groupby(["mode","src","group_id","metric"])
                        .agg(mean=("value", "mean"),
                             n_hparams=("value", "size"))
                        .reset_index())
    return mean_tbl

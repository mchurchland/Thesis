import os
import glob
import pd
import matplotlib as plt
import numpy as np

def _safe_path(path: str) -> str:
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

def _dispersion(a: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    m = float(np.mean(a))
    s = float(np.std(a))
    return s/(abs(m)+1e-12)

def _unique_hparam_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["rho_target","leak","input_scale"]
    metrics = [c for c in ("MC","IPC","KR","GR") if c in df.columns]
    if not metrics:
        return df.copy()
    return (df.groupby(keys, as_index=False)[metrics]
              .mean()
              .sort_values(keys)
              .reset_index(drop=True))

def _build_combined(df_shuf: pd.DataFrame | None, df_var: pd.DataFrame | None) -> pd.DataFrame:
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

def _compute_dispersion_table(combined: pd.DataFrame) -> pd.DataFrame:
    """
    For each (mode, src, group_id), compute dispersion across hyper-params:
      - group_id = shuffle_id if shuffle_id != -1 else src
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

    rows = []
    for (mode, src, gid), grp in df_agg.groupby(["mode","src","group_id"]):
        for m in metrics:
            rows.append({
                "mode": mode,
                "src": src,
                "group_id": gid,
                "metric": m,
                "dispersion": _dispersion(grp[m].to_numpy()),
                "n_hparams": len(grp)
            })
    disp = pd.DataFrame(rows)
    return disp



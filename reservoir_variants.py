from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from network_stats.run_one import run_one
from util.util import build_reservoir, degree_matched_shuffle_directed


@dataclass(frozen=True)
class SimulationParams:
    """Container for the time-series/metric settings used by run_one."""

    washout: int = 500
    perturb_std: float = 0.01
    t_train: int = 1500
    t_test: int = 500
    mc_max_delay: int = 300
    ipc_max_delay: int = 50
    ipc_max_order: int = 3
    ridge_alpha: float = 1e-4
    k_controllability: int = 100
    sat_thresh: float = 2.0
    near_zero_std: float = 1e-3


DEFAULT_SIM_PARAMS = SimulationParams()


@dataclass(frozen=True)
class VariantContext:
    """Inputs that stay constant while sweeping (rho, leak, input_scale)."""

    ce_W_bio: np.ndarray | None
    ce_ei: np.ndarray | None
    ws_k: int
    col_params: Sequence[tuple[float, float, float]]
    device: torch.device
    seed: int
    sid: int
    er_p: float = 0.1
    ws_p: float = 0.1
    src_tag: str = "chunk_0"
    sim_params: SimulationParams = DEFAULT_SIM_PARAMS


@dataclass(frozen=True)
class VariantSpec:
    """Defines how to instantiate a reservoir variant."""

    key: str
    label: str
    feature_weights: str
    conn_fn: Callable[[VariantContext], str]
    description: str
    nnz_match_ce: bool = False
    custom_ce_fn: Callable[[VariantContext, np.random.Generator], np.ndarray] | None = None
    seed_offset: int = 0
    sid_stride: int = 0
    requires_ce: bool = True


def _count_edges(A: np.ndarray) -> int:
    M = np.abs(A) > 0
    np.fill_diagonal(M, False)
    return int(M.sum())


def _shuffle_ce_weights(Wbio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    W = Wbio.copy().astype(np.float32)
    nz = np.nonzero(W)
    vals = W[nz].copy()
    rng.shuffle(vals)
    W[nz] = vals
    return W


def _conn_shuffle_ce(Wbio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degree-matched shuffle of CE adjacency with CE weight multiset."""
    A_ce = (np.abs(Wbio) > 0)
    np.fill_diagonal(A_ce, False)
    nnz_target = int(A_ce.sum())
    ce_weights_all = Wbio[A_ce].astype(np.float32)

    As = degree_matched_shuffle_directed(A_ce.astype(np.float32), tries=20_000, rng=rng).astype(bool)
    # Defensive: ensure edge count matches the original
    if int(As.sum()) != nnz_target:
        flat_idx = np.flatnonzero(As.ravel())
        if len(flat_idx) > nnz_target:
            As = As.ravel()
            As[flat_idx[nnz_target:]] = False
            As = As.reshape(A_ce.shape)

    Wsh = np.zeros_like(Wbio, dtype=np.float32)
    perm = rng.permutation(len(ce_weights_all))
    Wsh[As] = ce_weights_all[perm][:nnz_target]
    np.fill_diagonal(Wsh, 0.0)
    return Wsh


def evaluate_reservoir(
    Wt: torch.Tensor,
    Win: torch.Tensor,
    leak: float,
    device: torch.device,
    sim_params: SimulationParams = DEFAULT_SIM_PARAMS,
):
    """Wrapper around run_one with shared defaults."""
    return run_one(
        Wt,
        Win,
        leak,
        device,
        sim_params.washout,
        sim_params.perturb_std,
        sim_params.t_train,
        sim_params.t_test,
        sim_params.mc_max_delay,
        sim_params.ipc_max_delay,
        sim_params.ipc_max_order,
        sim_params.ridge_alpha,
        sim_params.k_controllability,
        sim_params.sat_thresh,
        sim_params.near_zero_std,
    )


def _run_variant_row(
    ctx: VariantContext,
    *,
    feature_conn: str,
    feature_weights: str,
    mode_label: str,
    ce_override: np.ndarray | None,
    nnz_target: int | None,
    seed_base: int,
) -> list[tuple]:
    rows_local = []

    # choose CE matrix when needed
    ce_for_conn =  ctx.ce_W_bio
    if feature_conn == "cel" and ce_for_conn is None:
        raise ValueError("feature_conn='cel' requires a CE adjacency matrix.")
    if ctx.ce_W_bio is None and feature_conn != "cel":
        raise ValueError("Non-CEL variants need ce_W_bio to set N/nnz.")

    Nloc = ce_for_conn.shape[0] if ce_for_conn is not None else ctx.ce_W_bio.shape[0]

    for ci, (target_sr, leak, in_scale) in enumerate(ctx.col_params):
            Wt, Win, _, _ = build_reservoir(
                feature_conn=feature_conn,
                feature_weights=feature_weights,
                target_sr=target_sr,
                N=Nloc,
                ce_W_bio=ce_for_conn if feature_conn == "cel" else ctx.ce_W_bio,
                ce_ei=ctx.ce_ei,
                ws_k=ctx.ws_k,
                input_scale=in_scale,
                seed=seed_base + ci * 101,
                drive_idx=None,
                nnz_target=nnz_target,
                DEVICE=ctx.device,
            )
            scores = evaluate_reservoir(Wt, Win, leak, ctx.device, ctx.sim_params)
            rows_local.append(
                (
                    mode_label,
                    ctx.sid,
                    target_sr,
                    leak,
                    in_scale,
                    float(scores["MC"]),
                    float(scores["IPC"]),
                    float(scores["KR"]),
                    float(scores["GR"]),
                    ctx.src_tag,
                )
            )
    return rows_local


def run_variant(spec: VariantSpec, ctx: VariantContext) -> list[tuple]:
    """Instantiate a variant (including CE preprocessing) and sweep col_params."""
    if spec.requires_ce and ctx.ce_W_bio is None:
        raise ValueError(f"Variant '{spec.key}' requires ce_W_bio but none was provided.")

    feature_conn = spec.conn_fn(ctx)
    seed_base = ctx.seed + spec.seed_offset + ctx.sid * spec.sid_stride
    rng = np.random.default_rng(seed_base)
    ce_override = spec.custom_ce_fn(ctx, rng) if spec.custom_ce_fn is not None else None

    nnz_target = None
    if spec.nnz_match_ce and ctx.ce_W_bio is not None and feature_conn != "cel":
        nnz_target = _count_edges(ctx.ce_W_bio)

    return _run_variant_row(
        ctx,
        feature_conn=feature_conn,
        feature_weights=spec.feature_weights,
        mode_label=spec.label,
        ce_override=ce_override,
        nnz_target=nnz_target,
        seed_base=seed_base,
    )


def save_rows(out_csv: str, rows: list[tuple], *, append: bool = False):
    mode = "a" if append and Path(out_csv).exists() else "w"
    with open(out_csv, mode, newline="") as f:
        import csv

        w = csv.writer(f)
        if mode == "w":
            w.writerow(["mode", "shuffle_id", "rho_target", "leak", "input_scale", "MC", "IPC", "KR", "GR", "src"])
        w.writerows(rows)


def _conn_cel(_: VariantContext) -> str:
    return "cel"


def _conn_er(ctx: VariantContext) -> str:
    return f"er_p={ctx.er_p}"


def _conn_ws(ctx: VariantContext) -> str:
    return f"ws_p={ctx.ws_p}"


def _conn_local_sign(_: VariantContext) -> str:
    return "local_sign_match_guas"


VARIANT_REGISTRY: dict[str, VariantSpec] = {
    "real": VariantSpec(
        key="real",
        label="real",
        feature_weights="bio",
        conn_fn=_conn_cel,
        description="C. elegans adjacency with biological weights.",
        seed_offset=123,
    ),
    "shuffle_weights": VariantSpec(
        key="shuffle_weights",
        label="shuffle",
        feature_weights="bio",
        conn_fn=_conn_cel,
        description="CE adjacency with CE weight multiset shuffled across nonzeros.",
        custom_ce_fn=lambda ctx, _rng: _shuffle_ce_weights(ctx.ce_W_bio, np.random.default_rng(ctx.seed)),
        seed_offset=9_999,
        sid_stride=1,
    ),
    "cel_randN": VariantSpec(
        key="cel_randN",
        label="cel+randN",
        feature_weights="rand_gauss",
        conn_fn=_conn_cel,
        description="CE adjacency with Gaussian weights.",
        seed_offset=10_000,
    ),
    "er_randN": VariantSpec(
        key="er_randN",
        label="er+randN",
        feature_weights="rand_gauss",
        conn_fn=_conn_er,
        description="Directed ER topology with Gaussian weights; nnz matched to CE.",
        nnz_match_ce=True,
        seed_offset=20_000,
    ),
    "ws_p01_randN": VariantSpec(
        key="ws_p01_randN",
        label="ws_p0.1+randN",
        feature_weights="rand_gauss",
        conn_fn=_conn_ws,
        description="WS topology (p from ctx) with Gaussian weights; nnz matched to CE.",
        nnz_match_ce=True,
        seed_offset=20_000,
    ),
    "conn_shuf": VariantSpec(
        key="conn_shuf",
        label="celW+connShuf",
        feature_weights="bio",
        conn_fn=_conn_cel,
        description="Degree-matched shuffle of CE connections, CE weight multiset reassigned.",
        custom_ce_fn=lambda ctx, _rng: _conn_shuffle_ce(
            ctx.ce_W_bio, np.random.default_rng(ctx.seed + 40_000 + ctx.sid)
        ),
        seed_offset=50_000,
        sid_stride=911,
    ),
    # Alias for compatibility with run_arc and older job keys
    "local_sign_match_guas": VariantSpec(
        key="local_sign_match_guas",
        label="local_sign",
        feature_weights="local_sign_match_guas",
        conn_fn=_conn_local_sign,
        description="Alias: CE adjacency with edge signs preserved but magnitudes replaced by N(0,1) (local sign match).",
        seed_offset=30_000,
    ),
}


def list_variants() -> list[tuple[str, str]]:
    """Return (key, description) pairs sorted by key."""
    return sorted(((k, v.description) for k, v in VARIANT_REGISTRY.items()), key=lambda kv: kv[0])

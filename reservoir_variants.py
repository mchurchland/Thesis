from __future__ import annotations

from dataclasses import dataclass,field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from network_stats.run_one import run_one
from util.util import build_reservoir, degree_matched_shuffle_directed, _cel_to_bin, \
    _count_edges,_shuffle_ce_weights,_conn_and_w_shuffle_ce,_conn_shuffle_ce,_sample_from_cel,scale_to_sr
from sklearn.metrics.pairwise import cosine_similarity

@dataclass(frozen=True)
class SimulationParams:
    """Container for the time-series/metric settings used by run_one."""

    washout: int = int(500 * 0.25)
    perturb_std: float = 0.01
    t_train: int = int(1500 * 0.25)
    t_test: int = int(500 * 0.25)
    mc_max_delay: int = 30
    ipc_max_delay: int = 30
    ipc_orders: list[int] = field(default_factory=lambda: [1, 3, 5])
    ridge_alpha: float = 1e-4

DEFAULT_SIM_PARAMS = SimulationParams()


@dataclass(frozen=True)
class VariantContext:
    """Inputs that stay constant while sweeping (rho, leak, input_scale)."""

    ce_W_bio: np.ndarray
    ce_ei: np.ndarray | None
    ws_k: int
    col_params: Sequence[tuple[float, float, float]]
    device: torch.device
    seed: int
    sid: int
    er_p: float = 0.1
    ws_p: float = 0.1
    per_neg: float | None = None
    alpha: float | None = None
    src_tag: str = "chunk_0"
    sim_params: SimulationParams = DEFAULT_SIM_PARAMS

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
        sim_params.ipc_orders,
        sim_params.ridge_alpha,
    )


def _run_variant_row(
    ctx: VariantContext,
    *,
    feature_conn: str,
    mode_label: str,
    ce_override: np.ndarray | None,
    nnz_target: int | None,
    seed_base: int,
) -> list[tuple]:
    rows_local = []

    # choose CE matrix when needed
    ce_for_conn = ce_override if ce_override is not None else ctx.ce_W_bio
    if feature_conn == "cel" and ce_for_conn is None:
        raise ValueError("feature_conn='cel' requires a CE adjacency matrix.")
    if ctx.ce_W_bio is None and feature_conn != "cel":
        raise ValueError("Non-CEL variants need ce_W_bio to set N/nnz.")

    Nloc = ce_for_conn.shape[0] if ce_for_conn is not None else ctx.ce_W_bio.shape[0]
    for ci, (target_sr, leak, in_scale) in enumerate(ctx.col_params):
        assert ctx.ce_ei==None
        Wt, Win = build_reservoir( 
            feature_conn=feature_conn,
            target_sr=target_sr,
            N=Nloc,
            ce_W_bio=ce_for_conn if feature_conn == "cel" else ctx.ce_W_bio,
            input_scale=in_scale,
            seed=seed_base,
            drive_idx=None,
            nnz_target=nnz_target,
            DEVICE=ctx.device,
            per_neg=ctx.per_neg,
            alpha = ctx.alpha
        )
        scores = evaluate_reservoir(Wt, Win, leak, ctx.device, ctx.sim_params)
        Wt_ce = torch.from_numpy(_cel_to_bin(ctx.ce_W_bio)).to(ctx.device) ## for cos sim
        sigma_ce = scale_to_sr(Wt_ce,target_sr) ##for cos sim
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
                float(Wt.mean().item()),
                float(
                    cosine_similarity(
                        sigma_ce.reshape(1, -1).detach().cpu().numpy(),
                        Wt.reshape(1, -1).detach().cpu().numpy(),
                    )[0, 0]
                ),
                seed_base,
                ctx.src_tag,
            )
        )
    return rows_local


def save_rows(out_csv: str, rows: list[tuple], *, append: bool = False):
    mode = "a" if append and Path(out_csv).exists() else "w"
    with open(out_csv, mode, newline="") as f:
        import csv

        w = csv.writer(f)
        if mode == "w":
            w.writerow(["mode", "shuffle_id", "rho_target", "leak", "input_scale", "MC", "IPC", "KR", "GR","wt_mean","cosine_similarity", "seed", "src"])
        w.writerows(rows)


# ------------------------ variant plumbing ------------------------

# Human-facing labels for CSV output (mode column)
VARIANT_LABELS = {
    "real": "real",
    "shuffle_weights": "shuffle",
    "cel_randN": "cel+randN",
    "er_randN": "er+randN",
    "ws_p01_randN": "ws_p0.1+randN",
    "conn_shuf": "celW+connShuf",
    "local_sign": "local_sign",
    "conn_shuf_only" : "conn_shuf_only",
    "cel_sample" : "cel_sample",
    "local_sign+flat" : "local_sign+flat",
    "local_sign+sample" : "local_sign+sample",
    "local_sign+binary" : "local_sign+binary",
    "global_sign_pres" : "global_sign_pres",
    "sign_test" : "sign_test",
    "weight_test" : "weight_test",
    "binary+shuffle" : "binary+shuffle",
}

# Short descriptions (used by list_variants/help text)
VARIANT_DESCRIPTIONS = {
    "real": "C. elegans adjacency with biological weights.",
    "shuffle_weights": "CE adjacency with CE weight multiset shuffled across nonzeros.",
    "cel_randN": "CE adjacency with Gaussian weights.",
    "er_randN": "Directed ER topology with Gaussian weights; nnz matched to CE.",
    "ws_p01_randN": "WS topology (p from ctx) with Gaussian weights; nnz matched to CE.",
    "conn_shuf": "Degree-matched shuffle of CE connections, CE weight shuffleed.",
    "local_sign": "CE adjacency; preserve sign pattern, replace magnitudes with N(0,1) (local sign match).",
    "conn_shuf_only" : "just shuffle all of the connections dont do anyhting else, directed graph swap",
    "cel_sample" : "resample the weights from the celegan weights keep the celegan connections",
    "local_sign+flat" : "local_sign preserved with a weights from flat dist",
    "local_sign+sample" : "local_sign preserved with a celegan weight sample",
    "local_sign+binary" : "local_sign preserved with a binary weight sample, +1 or -1",
    "sign_test" : "celegan connectome, can pass in a percent to flip and it will flip the sign of that many conenctions",
    "weight_test" : "celegan connectome, can pass in an alpha to add a guasian dist times that alpha to the weights",
    "global_sign_pres" : " preserve global sign balance in the binary model but shuffle the signs so that they can be on different edges :-) smiley face for Jordi :-)",
    "binary+shuffle" : "convert the weights to binary (sign) and then do a degree-matched shuffle of the connections (and thus signs), so global sign balance is preserved but signs can be on different edges",
}

# Backwards-compatible keys allowed for callers; resolve to canonical names above.
VARIANT_ALIASES = {
}

VARIANT_KEYS: tuple[str, ...] = tuple(VARIANT_LABELS.keys())


def _seed(ctx: VariantContext, *, offset: int, sid_stride: int = 0) -> int:
    """Shared seed rule used across variants."""
    return ctx.seed + offset + ctx.sid * sid_stride


def _nnz_match_ce(ctx: VariantContext) -> int:
    if ctx.ce_W_bio is None:
        raise ValueError("nnz_match_ce requires ce_W_bio.")
    return _count_edges(ctx.ce_W_bio)


def _resolve_key(key: str) -> str:
    return VARIANT_ALIASES.get(key, key)


def _require_ce(ctx: VariantContext):
    if ctx.ce_W_bio is None:
        raise ValueError("This variant requires ce_W_bio (CE adjacency).")


def run_variant(key: str, ctx: VariantContext) -> list[tuple]:
    """Instantiate a reservoir variant (direct dispatch, no VariantSpec indirection)."""
    key = _resolve_key(key)
    if key not in VARIANT_KEYS:
        if (not key.startswith("sign_test") and (not key.startswith("weight_test"))):
            raise ValueError(f"Unknown variant key: {key}")

    _require_ce(ctx)

    if key == "real":
        seed_base = _seed(ctx, offset=123)
        return _run_variant_row(
            ctx,
            feature_conn="cel",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )

    if key == "shuffle_weights": ## investigate
        # Shuffle CE weight magnitudes across the existing edge set.
        ce_override = _shuffle_ce_weights(ctx.ce_W_bio, np.random.default_rng(ctx.seed))
        seed_base = _seed(ctx, offset=9_999, sid_stride=1)
        return _run_variant_row(
            ctx,
            feature_conn="cel",
            mode_label=VARIANT_LABELS[key],
            ce_override=ce_override,
            nnz_target=None,
            seed_base=seed_base,
        )

    if key == "cel_randN":
        seed_base = _seed(ctx, offset=10_000)
        return _run_variant_row(
            ctx,
            feature_conn="cel_randN",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )

    if key == "er_randN":
        seed_base = _seed(ctx, offset=21_000)
        return _run_variant_row(
            ctx,
            feature_conn=f"er_p={ctx.er_p}",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=_nnz_match_ce(ctx),
            seed_base=seed_base,
        )

    if key == "ws_p01_randN":
        seed_base = _seed(ctx, offset=20_000)
        return _run_variant_row(
            ctx,
            feature_conn=f"ws_p={ctx.ws_p}",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=_nnz_match_ce(ctx),
            seed_base=seed_base,
        )

    if key == "conn_shuf": ## investigate
        ce_override = _conn_and_w_shuffle_ce(
            ctx.ce_W_bio, np.random.default_rng(ctx.seed + 40_000 + ctx.sid)
        )
        seed_base = _seed(ctx, offset=50_000, sid_stride=911)
        return _run_variant_row(
            ctx,
            feature_conn="cel",
            mode_label=VARIANT_LABELS[key],
            ce_override=ce_override,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "local_sign":
        seed_base = _seed(ctx, offset=30_000)
        return _run_variant_row(
            ctx,
            feature_conn="local_sign",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "conn_shuf_only":
        ce_override = _conn_shuffle_ce(
            ctx.ce_W_bio, np.random.default_rng(ctx.seed + 31_000 + ctx.sid)
        )
        seed_base = _seed(ctx, offset=31_000)
        return _run_variant_row(
            ctx,
            feature_conn="cel",
            mode_label=VARIANT_LABELS[key],
            ce_override=ce_override,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "cel_sample":
        ce_override = _sample_from_cel(
            ctx.ce_W_bio, np.random.default_rng(ctx.seed + 32_000 + ctx.sid)
        )
        seed_base = _seed(ctx, offset=32_000)
        return _run_variant_row(
            ctx,
            feature_conn="cel",
            mode_label=VARIANT_LABELS[key],
            ce_override=ce_override,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "local_sign+flat":
        seed_base = _seed(ctx, offset=33_000)
        return _run_variant_row(
            ctx,
            feature_conn="local_sign+flat",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "local_sign+sample":
        seed_base = _seed(ctx, offset=34_000)
        return _run_variant_row(
            ctx,
            feature_conn="local_sign+sample",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "local_sign+binary":
        seed_base = _seed(ctx, offset=35_000)
        return _run_variant_row(
            ctx,
            feature_conn="local_sign+binary",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "global_sign_pres":
        seed_base = _seed(ctx, offset=35_500)
        return _run_variant_row(
            ctx,
            feature_conn="global_sign_pres",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )


    if key.startswith("sign_test"):
        seed_base = _seed(ctx, offset=36_000)
        return _run_variant_row(
            ctx,
            feature_conn="sign_test",
            mode_label=key,
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key.startswith("weight_test"):
        seed_base = _seed(ctx, offset=37_000)
        return _run_variant_row(
            ctx,
            feature_conn="weight_test",
            mode_label=key,
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "binary+shuffle":
        seed_base = _seed(ctx, offset=38_000)
        return _run_variant_row(
            ctx,
            feature_conn="binary+shuffle",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    

    # Defensive (should never reach here)
    raise ValueError(f"Variant key not implemented: {key}")

def list_variants() -> list[tuple[str, str]]:
    """Return (key, description) pairs sorted by key."""
    return sorted(((k, VARIANT_DESCRIPTIONS[k]) for k in VARIANT_KEYS), key=lambda kv: kv[0])


##redraw weights from cel, shuffle only 

from __future__ import annotations

from dataclasses import dataclass,field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import entropy, halfnorm

from network_stats.run_one import run_one
from util.util import build_reservoir, degree_matched_shuffle_directed, _cel_to_bin, \
    _count_edges,_shuffle_ce_weights,_conn_and_w_shuffle_ce,_conn_shuffle_ce,_sample_from_cel,scale_to_sr
from sklearn.metrics.pairwise import cosine_similarity

def _kl_empirical_to_fitted_gaussian(
    values: np.ndarray,
    bins: int = 64,
    eps: float = 1e-12,
) -> float:
    """
    Magnitude-only KL on edge weights.
    Computes KL(|W| || fitted shifted-half-normal) on nonzero finite weights.
    """
    x = values.reshape(-1) ## make it a 1d  array
    x = x[np.isfinite(x)] ## this is just for safety, the weights should be finite but just in case we remove any inf or nan values
    x = x[x != 0] ##get non zero
    if x.size < 2:
        return float("nan")

    x_abs = np.abs(x) ## get the absolute
    x_max = float(np.max(x_abs))
    if x_max <= 0.0:
        return 0.0

    loc = float(np.min(x_abs))
    y = x_abs - loc ## shift the distribution so that the minimum is at zero
    sigma = float(np.sqrt(np.mean(y * y))) ## this is the std of the shifted distribution, which is the scale parameter for the half-normal fit
    if sigma <= eps: ## if the std is zero (all values are the same), this is like invalid it should be inifinite but we do 0.0 it never happens
        return 0.0

    counts, edges = np.histogram(x_abs, bins=bins, range=(loc, x_max)) ## discretize the values into bins
    p = counts.astype(np.float64)
    p = np.clip(p, eps, None)
    p = p / p.sum() ## get the probabilities

    # Shifted half-normal CDF over the same bin edges.
    q_cdf = halfnorm.cdf(edges, loc=loc, scale=sigma)
    q = np.diff(q_cdf) ## this just gets the bin probabilties from the cumulative probabilities, so q_i = CDF(edge_i+1) - CDF(edge_i)
    q = np.clip(q, eps, None) ## clip at 0
    q = q / q.sum() ## renormalize
    return float(entropy(p, q)) ## this is just \sum 1/n log(p_i/q_i) where p is the empirical distribution and q is the fitted distribution


def _weight_magnitude_cv(values: np.ndarray, eps: float = 1e-12) -> float:
    x = values.reshape(-1)
    x = x[np.isfinite(x)]
    x = np.abs(x[x != 0])
    if x.size == 0:
        return float("nan")
    mean = float(np.mean(x))
    if abs(mean) <= eps:
        return float("nan")
    return float(np.std(x) / mean)


@dataclass(frozen=True)
class SimulationParams:
    """Container for the time-series/metric settings used by run_one."""

    washout: int = int(500)
    perturb_std: float = 0.01
    t_train: int = int(1500)
    t_test: int = int(500)
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
    input_idx: np.ndarray | None = None
    output_idx: np.ndarray | None = None

def evaluate_reservoir(
    Wt: torch.Tensor,
    Win: torch.Tensor,
    leak: float,
    device: torch.device,
    sim_params: SimulationParams = DEFAULT_SIM_PARAMS,
    output_idx: np.ndarray | torch.Tensor | None = None,
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
        output_idx,
    )


def _apply_input_subset(Win: torch.Tensor, input_idx: np.ndarray | None) -> torch.Tensor:
    """Keep the usual random input weights, but drive only selected nodes."""
    if input_idx is None:
        return Win
    idx = torch.as_tensor(input_idx, device=Win.device, dtype=torch.long)
    if idx.numel() == 0:
        raise ValueError("input_idx must contain at least one node when provided.")
    mask = torch.zeros_like(Win)
    mask.index_fill_(0, idx, 1.0)
    return Win * mask


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
        Win = _apply_input_subset(Win, ctx.input_idx)
        w_np = Wt.detach().cpu().numpy().reshape(-1)
        kl_to_gaussian = _kl_empirical_to_fitted_gaussian(w_np)
        wt_mag_cv = _weight_magnitude_cv(w_np)
        scores = evaluate_reservoir(Wt, Win, leak, ctx.device, ctx.sim_params, ctx.output_idx)
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
                kl_to_gaussian,
                wt_mag_cv,
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
            w.writerow(["mode", "shuffle_id", "rho_target", "leak", "input_scale", "MC", "IPC", "KR", "GR","wt_mean","cosine_similarity", "kl_to_gaussian", "wt_mag_cv", "seed", "src"])
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
    "binary_base": "binary_base",
    "binary_base_topology_shuffle": "binary_base_topology_shuffle",
    "binary+conshuffle+wshuffle": "binary+conshuffle+wshuffle",
    "sign_test_cel" : "sign_test_cel",
    "sign_test" : "sign_test",
    "sign_test_er" : "sign_test_er",
    "weight_test" : "weight_test",
    "weight_test_unsigned": "weight_test_unsigned",
    "weight_test_signed": "weight_test_signed",
    "weight_test_cel_to_shuffled_cel": "weight_test_cel_to_shuffled_cel",
    "weight_test_binary_to_cel": "weight_test_binary_to_cel",
    "weight_test_binary_to_shuffled_cel": "weight_test_binary_to_shuffled_cel",
    "binary+shuffle" : "binary+shuffle",
    "sign_test_og_cel": "sign_test_og_cel"
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
    "sign_test_cel" : "celegan connectome binarized, can pass in a percent to flip and it will flip the sign of that many conenctions",
    "sign_test_og_cel" : "celegan connectome, can pass in a percent to flip and it will flip the sign of that many conenctions does not convert the weights to binary",
    "sign_test_er" : "ER connectome, can pass in a percent to flip and it will flip the sign of that many connections",
    "weight_test" : "Backward-compatible alias for weight_test_unsigned.",
    "weight_test_unsigned": "C. elegans weights plus signed Gaussian noise; signs may change.",
    "weight_test_signed": "C. elegans weights plus Gaussian noise with original edge signs restored.",
    "weight_test_cel_to_shuffled_cel": "Interpolate from empirical C. elegans magnitude placement to shuffled empirical magnitude placement while preserving signs.",
    "weight_test_binary_to_cel": "Interpolate from sign-preserving binary weights to empirical C. elegans magnitudes.",
    "weight_test_binary_to_shuffled_cel": "Interpolate from sign-preserving binary weights to shuffled empirical C. elegans magnitudes.",
    "global_sign_pres" : " preserve global sign balance in the binary model but shuffle the signs so that they can be on different edges :-) smiley face for Jordi :-)",
    "binary_base": "Unsigned CE binary base (0/1): topology and magnitudes fixed except binarization.",
    "binary_base_topology_shuffle": "Unsigned CE binary base with degree-preserving topology shuffle.",
    "binary+conshuffle+wshuffle": "Signed binary CE with degree-preserving topology shuffle plus weight/sign shuffle on nonzero edges.",
    "binary+shuffle" : "convert the weights to binary (sign) and then do a degree-matched shuffle of the connections (and thus signs), so global sign balance is preserved but signs can be on different edges",
    "sign_test" : "both cel and er"
}

# Backwards-compatible keys allowed for callers; resolve to canonical names above.
VARIANT_ALIASES = {
}

VARIANT_KEYS: tuple[str, ...] = tuple(VARIANT_LABELS.keys())

WEIGHT_TEST_FEATURES = {
    "weight_test_binary_to_shuffled_cel": "weight_test_binary_to_shuffled_cel",
    "weight_test_cel_to_shuffled_cel": "weight_test_cel_to_shuffled_cel",
    "weight_test_binary_to_cel": "weight_test_binary_to_cel",
    "weight_test_unsigned": "weight_test_unsigned",
    "weight_test_signed": "weight_test_signed",
    "weight_test": "weight_test_unsigned",
}


def _seed(ctx: VariantContext, *, offset: int, sid_stride: int = 0) -> int:
    """Shared seed rule used across variants."""
    return ctx.seed + offset + ctx.sid * sid_stride


def _nnz_match_ce(ctx: VariantContext) -> int:
    if ctx.ce_W_bio is None:
        raise ValueError("nnz_match_ce requires ce_W_bio.")
    return _count_edges(ctx.ce_W_bio)


def _resolve_key(key: str) -> str:
    return VARIANT_ALIASES.get(key, key)


def _weight_test_feature_conn(key: str) -> str:
    for prefix, feature_conn in WEIGHT_TEST_FEATURES.items():
        if key.startswith(prefix):
            return feature_conn
    raise ValueError(f"Unknown weight_test variant key: {key}")


def _require_ce(ctx: VariantContext):
    if ctx.ce_W_bio is None:
        raise ValueError("This variant requires ce_W_bio (CE adjacency).")


def run_variant(key: str, ctx: VariantContext) -> list[tuple]:
    """Instantiate a reservoir variant (direct dispatch, no VariantSpec indirection)."""
    key = _resolve_key(key)
    if key not in VARIANT_KEYS:
        if ((not key.startswith("sign_test_cel")) and (not key.startswith("weight_test")) and (not key.startswith("sign_test_er")) and (not key.startswith("sign_test_og_cel"))):
            raise ValueError(f"Unknown variant key: {key}")

    _require_ce(ctx)

    if key == "real":
        seed_base = _seed(ctx, offset=1)
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
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="cel",
            mode_label=VARIANT_LABELS[key],
            ce_override=ce_override,
            nnz_target=None,
            seed_base=seed_base,
        )

    if key == "cel_randN":
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="cel_randN",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )

    if key == "er_randN":
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn=f"er_p={ctx.er_p}",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=_nnz_match_ce(ctx),
            seed_base=seed_base,
        )

    if key == "ws_p01_randN":
        seed_base = _seed(ctx, offset=1)
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
            ctx.ce_W_bio, np.random.default_rng(ctx.seed)
        )
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="cel",
            mode_label=VARIANT_LABELS[key],
            ce_override=ce_override,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "local_sign":
        seed_base = _seed(ctx, offset=1)
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
            ctx.ce_W_bio, np.random.default_rng(ctx.seed)
        )
        seed_base = _seed(ctx, offset=1)
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
            ctx.ce_W_bio, np.random.default_rng(ctx.seed)
        )
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="cel",
            mode_label=VARIANT_LABELS[key],
            ce_override=ce_override,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "local_sign+flat":
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="local_sign+flat",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "local_sign+sample":
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="local_sign+sample",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "local_sign+binary":
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="local_sign+binary",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "global_sign_pres":
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="global_sign_pres",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "binary_base":
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="binary_base",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "binary_base_topology_shuffle":
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="binary_base_topology_shuffle",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "binary+conshuffle+wshuffle":
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="binary+conshuffle+wshuffle",
            mode_label=VARIANT_LABELS[key],
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key.startswith("sign_test_cel"):
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="sign_test_cel",
            mode_label=key,
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key.startswith("sign_test_er"):
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="sign_test_er",
            mode_label=key,
            ce_override=None,
            nnz_target=_nnz_match_ce(ctx),
            seed_base=seed_base,
        )
    
    if key.startswith("sign_test_og_cel"):
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn="sign_test_og_cel",
            mode_label=key,
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key.startswith("weight_test"):
        seed_base = _seed(ctx, offset=1)
        return _run_variant_row(
            ctx,
            feature_conn=_weight_test_feature_conn(key),
            mode_label=key,
            ce_override=None,
            nnz_target=None,
            seed_base=seed_base,
        )
    if key == "binary+shuffle":
        seed_base = _seed(ctx, offset=1)
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

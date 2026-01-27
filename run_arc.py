from __future__ import annotations
import numpy as np
import os
from typing import Iterable

import torch

from reservoir_variants import (
    DEFAULT_SIM_PARAMS,
    VARIANT_KEYS,
    VariantContext,
    run_variant,
    save_rows,
)


def _prepare_out_csv(out_dir: str, csv_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, csv_name)


def _run_and_save(
    variant_key: str,
    *,
    WS_K: int,
    ce_W_bio: np.ndarray | None,
    ce_ei: np.ndarray | None,
    col_params: Iterable[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    seed: int,
    sid: int,
    er_p: float,
    ws_p: float,
    csv_name: str,
    src_tag: str,
    sim_params=DEFAULT_SIM_PARAMS,
    append: bool = False,
) -> list[tuple]:
    ctx = VariantContext(
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        ws_k=WS_K,
        col_params=tuple(col_params),
        device=device,
        seed=seed,
        sid=sid,
        er_p=er_p,
        ws_p=ws_p,
        src_tag=src_tag,
        sim_params=sim_params,
    )
    rows = run_variant(variant_key, ctx)
    out_csv = _prepare_out_csv(out_dir, csv_name)
    save_rows(out_csv, rows, append=append)
    return rows


def run_one_real(
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    nid: int = 1,
    csv_name: str = "bio_vs_shuffle_invariance.csv",
    src_tag: str = "chunk_0",
    append: bool = False,
):
    return _run_and_save(
        "real",
        WS_K=WS_K,
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        col_params=col_params,
        out_dir=out_dir,
        device=device,
        seed=seed,
        sid=nid,
        er_p=0.1,
        ws_p=0.1,
        csv_name=csv_name,
        src_tag=src_tag,
        append=append,
    )


def run_one_shuf_weights(
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    sid: int = -1,
    metric: str = "MC",
    csv_name: str = "bio_vs_shuffle_invariance.csv",
    src_tag: str = "chunk_0",
    append: bool = False,
):
    # metric kept for compatibility (results are stored regardless)
    return _run_and_save(
        "shuffle_weights",
        WS_K=WS_K,
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        col_params=col_params,
        out_dir=out_dir,
        device=device,
        seed=seed,
        sid=sid,
        er_p=0.1,
        ws_p=0.1,
        csv_name=csv_name,
        src_tag=src_tag,
        append=append,
    )


def run_one_cel_randN(
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    sid: int = -1,
    csv_name: str = "cel_variants.csv",
    src_tag: str = "chunk_0",
    append: bool = False,
):
    return _run_and_save(
        "cel_randN",
        WS_K=WS_K,
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        col_params=col_params,
        out_dir=out_dir,
        device=device,
        seed=seed,
        sid=sid,
        er_p=0.1,
        ws_p=0.1,
        csv_name=csv_name,
        src_tag=src_tag,
        append=append,
    )

def run_one_sign_pres(
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    sid: int = -1,
    csv_name: str = "cel_variants.csv",
    src_tag: str = "chunk_0",
    append: bool = False,
):
    return _run_and_save(
        "local_sign",
        WS_K=WS_K,
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        col_params=col_params,
        out_dir=out_dir,
        device=device,
        seed=seed,
        sid=sid,
        er_p=0,
        ws_p=0.1,  
        csv_name=csv_name,
        src_tag=src_tag,
        append=append,
    )

def run_one_esn_er_randN(
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    er_p: float = 0.1,
    seed: int = 0,
    sid: int = -1,
    csv_name: str = "cel_variants.csv",
    src_tag: str = "chunk_0",
    append: bool = False,
):
    return _run_and_save(
        "er_randN",
        WS_K=WS_K,
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        col_params=col_params,
        out_dir=out_dir,
        device=device,
        seed=seed,
        sid=sid,
        er_p=er_p,
        ws_p=0.1,
        csv_name=csv_name,
        src_tag=src_tag,
        append=append,
    )


def run_one_ws_p0_1_randN(
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    ws_p: float = 0.1,
    seed: int = 0,
    sid: int = -1,
    csv_name: str = "cel_variants.csv",
    src_tag: str = "chunk_0",
    append: bool = False,
):
    return _run_and_save(
        "ws_p01_randN",
        WS_K=WS_K,
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        col_params=col_params,
        out_dir=out_dir,
        device=device,
        seed=seed,
        sid=sid,
        er_p=0.1,
        ws_p=ws_p,
        csv_name=csv_name,
        src_tag=src_tag,
        append=append,
    )


def run_one_celW_connShuf(
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    sid: int = 1,
    seed: int = 0,
    csv_name: str = "invariance_variants.csv",
    src_tag: str = "chunk_0",
    append: bool = False,
):
    return _run_and_save(
        "conn_shuf",
        WS_K=WS_K,
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        col_params=col_params,
        out_dir=out_dir,
        device=device,
        seed=seed,
        sid=sid,
        er_p=0.1,
        ws_p=0.1,
        csv_name=csv_name,
        src_tag=src_tag,
        append=append,
    )

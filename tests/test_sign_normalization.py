import numpy as np
import pytest
import torch

from util.util import build_reservoir


def _signed_ce_reference(n: int, negative_fraction: float) -> np.ndarray:
    reference = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(reference, 0.0)
    edges = np.argwhere(reference != 0)
    n_negative = int(negative_fraction * len(edges))
    negative_edges = edges[:n_negative]
    reference[negative_edges[:, 0], negative_edges[:, 1]] = -1.0
    return reference


def test_sign_test_er_original_radius_uses_ce_sign_fraction_on_its_own_realization():
    n = 12
    target_sr = 1.0
    reference_negative_fraction = 0.25
    common = {
        "target_sr": target_sr,
        "input_scale": 0.5,
        "seed": 31415,
        "feature_conn": "sign_test_er",
        "N": n,
        "ce_W_bio": _signed_ce_reference(n, reference_negative_fraction),
        "nnz_target": 36,
        "DEVICE": torch.device("cpu"),
        "normalization_mode": "original_radius",
        "return_info": True,
    }

    _, _, baseline_info = build_reservoir(per_neg=reference_negative_fraction, **common)
    _, _, unsigned_info = build_reservoir(per_neg=0.0, **common)
    _, _, balanced_info = build_reservoir(per_neg=0.5, **common)

    # At the CE negative fraction, the raw ER matrix is exactly the paired
    # signed reference. The same reference is retained across all sign points.
    assert baseline_info["ref_rho"] == pytest.approx(baseline_info["raw_rho"])
    assert baseline_info["post_rho"] == pytest.approx(target_sr, rel=1e-5)
    assert unsigned_info["ref_rho"] == pytest.approx(baseline_info["ref_rho"])
    assert balanced_info["ref_rho"] == pytest.approx(baseline_info["ref_rho"])
    assert unsigned_info["raw_rho"] != pytest.approx(unsigned_info["ref_rho"])

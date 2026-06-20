import numpy as np
import pytest
import torch

from util.util import build_reservoir


def _dense_ce_reference(n: int) -> np.ndarray:
    reference = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(reference, 0.0)
    return reference


def test_sign_test_er_original_radius_uses_its_own_unsigned_realization():
    n = 12
    target_sr = 1.0
    common = {
        "target_sr": target_sr,
        "input_scale": 0.5,
        "seed": 31415,
        "feature_conn": "sign_test_er",
        "N": n,
        "ce_W_bio": _dense_ce_reference(n),
        "nnz_target": 36,
        "DEVICE": torch.device("cpu"),
        "normalization_mode": "original_radius",
        "return_info": True,
    }

    _, _, unsigned_info = build_reservoir(per_neg=0.0, **common)
    _, _, balanced_info = build_reservoir(per_neg=0.5, **common)

    # At zero negative edges, the raw ER matrix is exactly the pre-sign ER
    # reference. The same seed must retain that reference across sign points.
    assert unsigned_info["ref_rho"] == pytest.approx(unsigned_info["raw_rho"])
    assert unsigned_info["post_rho"] == pytest.approx(target_sr, rel=1e-5)
    assert balanced_info["ref_rho"] == pytest.approx(unsigned_info["ref_rho"])


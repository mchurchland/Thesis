import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import pytest
import torch

from inv_arc_test import _draw_node_subsets
from reservoir_variants import _apply_input_subset, _weight_test_feature_conn
from util.util import _binary_to_cel_interpolation, scale_weights
from network_stats.run_one import run_reservoir_with_pre
import network_stats.run_one as run_one_module


def test_draw_node_subsets_are_deterministic_and_disjoint():
    input_idx, output_idx = _draw_node_subsets(n_nodes=10, k_in=3, k_out=4, seed=123)
    input_idx_again, output_idx_again = _draw_node_subsets(n_nodes=10, k_in=3, k_out=4, seed=123)
    input_idx_other, output_idx_other = _draw_node_subsets(n_nodes=10, k_in=3, k_out=4, seed=124)

    assert np.array_equal(input_idx, input_idx_again)
    assert np.array_equal(output_idx, output_idx_again)
    assert not (
        np.array_equal(input_idx, input_idx_other)
        and np.array_equal(output_idx, output_idx_other)
    )

    assert len(input_idx) == 3
    assert len(output_idx) == 4
    assert len(np.unique(input_idx)) == 3
    assert len(np.unique(output_idx)) == 4
    assert set(input_idx).isdisjoint(set(output_idx))


def test_draw_node_subsets_rejects_too_many_disjoint_nodes():
    with pytest.raises(ValueError, match="k_in \\+ k_out"):
        _draw_node_subsets(n_nodes=5, k_in=3, k_out=3, seed=0)


def test_input_subset_masks_direct_drive_to_selected_nodes_only():
    n_nodes = 6
    input_idx = np.array([1, 4], dtype=np.int64)
    win = torch.arange(1, n_nodes + 1, dtype=torch.float32).reshape(n_nodes, 1)

    masked_win = _apply_input_subset(win, input_idx)
    selected = torch.as_tensor(input_idx, dtype=torch.long)
    unselected = torch.tensor([0, 2, 3, 5], dtype=torch.long)

    assert torch.equal(masked_win.index_select(0, selected), win.index_select(0, selected))
    assert torch.count_nonzero(masked_win.index_select(0, unselected)) == 0

    # With W=0 and leak=1, unselected nodes can only move if they receive input.
    W = torch.zeros(n_nodes, n_nodes)
    u = torch.full((5, 1), 0.25)
    X, _ = run_reservoir_with_pre(W, masked_win, u, leak=1.0)

    assert torch.count_nonzero(X.index_select(1, unselected)) == 0
    assert torch.all(torch.abs(X.index_select(1, selected)) > 0)


def test_weight_test_signed_preserves_original_edge_signs():
    W = np.array(
        [
            [0.0, 2.0, -3.0],
            [4.0, 0.0, -5.0],
            [0.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = scale_weights(
        W,
        alpha=10.0,
        rng=np.random.default_rng(123),
        preserve_sign=True,
    )

    mask = W != 0
    assert np.array_equal(np.sign(out[mask]), np.sign(W[mask]))


def test_binary_to_cel_interpolation_endpoints_and_shuffled_control():
    W = np.array(
        [
            [0.0, 2.0, -3.0],
            [4.0, 0.0, -5.0],
            [-6.0, 7.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = W != 0

    binary = _binary_to_cel_interpolation(
        W,
        alpha=0.0,
        rng=np.random.default_rng(1),
    )
    empirical = _binary_to_cel_interpolation(
        W,
        alpha=1.0,
        rng=np.random.default_rng(1),
    )
    shuffled = _binary_to_cel_interpolation(
        W,
        alpha=1.0,
        rng=np.random.default_rng(1),
        shuffle_magnitudes=True,
    )

    assert np.array_equal(binary[mask], np.sign(W[mask]))
    assert np.array_equal(empirical, W)
    assert np.array_equal(np.sign(shuffled[mask]), np.sign(W[mask]))
    assert np.array_equal(np.sort(np.abs(shuffled[mask])), np.sort(np.abs(W[mask])))
    assert not np.array_equal(np.abs(shuffled[mask]), np.abs(W[mask]))


def test_binary_to_cel_interpolation_rejects_out_of_range_alpha():
    with pytest.raises(ValueError, match="between 0 and 1"):
        _binary_to_cel_interpolation(
            np.array([[0.0, 2.0]], dtype=np.float32),
            alpha=1.5,
            rng=np.random.default_rng(1),
        )


def test_weight_test_dynamic_keys_resolve_to_expected_feature_connections():
    assert _weight_test_feature_conn("weight_test0.0") == "weight_test_unsigned"
    assert _weight_test_feature_conn("weight_test_unsigned10.0") == "weight_test_unsigned"
    assert _weight_test_feature_conn("weight_test_signed10.0") == "weight_test_signed"
    assert (
        _weight_test_feature_conn("weight_test_binary_to_cel0.5")
        == "weight_test_binary_to_cel"
    )
    assert (
        _weight_test_feature_conn("weight_test_binary_to_shuffled_cel0.5")
        == "weight_test_binary_to_shuffled_cel"
    )


def test_run_one_passes_only_output_subset_to_metrics(monkeypatch):
    n_nodes = 5
    output_idx = torch.tensor([1, 4], dtype=torch.long)
    win = torch.arange(1, n_nodes + 1, dtype=torch.float32).reshape(n_nodes, 1)
    selected_win = win.index_select(0, output_idx).flatten()
    calls = {"mc": 0, "ipc": 0, "kr": 0, "gr": 0}

    def assert_selected_states(X: torch.Tensor, u: torch.Tensor) -> None:
        expected = torch.tanh(u * selected_win.reshape(1, -1))
        assert X.shape[1] == len(output_idx)
        assert torch.allclose(X, expected, atol=1e-6)

    def fake_mc(Xtr, Xte, utr, ute, max_delay, ridge_alpha, device):
        calls["mc"] += 1
        assert_selected_states(Xtr, utr)
        assert_selected_states(Xte, ute)
        return 1.0, None

    def fake_ipc(Xtr, Xte, utr, ute, max_delay, ridge_alpha, device, orders):
        calls["ipc"] += 1
        assert Xtr.shape[1] == len(output_idx)
        assert Xte.shape[1] == len(output_idx)
        return 2.0

    def fake_kr(Xtr):
        calls["kr"] += 1
        assert Xtr.shape[1] == len(output_idx)
        return 3.0

    def fake_gr(Xtr, Xntr):
        calls["gr"] += 1
        assert Xtr.shape[1] == len(output_idx)
        assert Xntr.shape[1] == len(output_idx)
        return 4.0

    monkeypatch.setattr(run_one_module, "compute_MC", fake_mc)
    monkeypatch.setattr(run_one_module, "compute_IPC", fake_ipc)
    monkeypatch.setattr(run_one_module, "compute_KR", fake_kr)
    monkeypatch.setattr(run_one_module, "compute_GR", fake_gr)

    scores = run_one_module.run_one(
        W=torch.zeros(n_nodes, n_nodes),
        Win=win,
        leak=1.0,
        device=torch.device("cpu"),
        WASHOUT=2,
        PERTURB_STD=0.01,
        T_TRAIN=4,
        T_TEST=3,
        MC_MAX_DELAY=1,
        IPC_MAX_DELAY=1,
        IPC_ORDERS=[1],
        RIDGE_ALPHA=1e-4,
        output_idx=output_idx,
    )

    assert scores == {"MC": 1.0, "IPC": 2.0, "KR": 3.0, "GR": 4.0}
    assert calls == {"mc": 1, "ipc": 1, "kr": 1, "gr": 1}

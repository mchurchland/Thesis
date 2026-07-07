import numpy as np

from util.util import (
    assign_random_unknown_signs,
    load_connectome,
    load_connectome_node_names,
    load_unknown_sign_weights,
)


def test_load_connectome_uses_nodes_suffix_matching_adjacency(tmp_path):
    adj_path = tmp_path / "ce_adj_new.npy"
    ei_path = tmp_path / "ce_ei_new.npy"
    np.save(adj_path, np.zeros((3, 3), dtype=np.float32))
    np.save(ei_path, np.zeros(3, dtype=np.float32))
    (tmp_path / "ce_nodes.txt").write_text("old_a\nold_b\n", encoding="utf-8")
    (tmp_path / "ce_nodes_new.txt").write_text("a\nb\nc\n", encoding="utf-8")

    _, _, name2idx = load_connectome(str(adj_path), str(ei_path))

    assert name2idx == {"a": 0, "b": 1, "c": 2}


def test_load_connectome_uses_nodes_prefix_matching_adjacency(tmp_path):
    adj_path = tmp_path / "ce_new_adj.npy"
    np.save(adj_path, np.zeros((2, 2), dtype=np.float32))
    (tmp_path / "ce_new_nodes.txt").write_text("a\nb\n", encoding="utf-8")

    _, _, name2idx = load_connectome(str(adj_path), None)

    assert name2idx == {"a": 0, "b": 1}


def test_load_connectome_node_names_returns_index_aligned_fallbacks(tmp_path):
    adj_path = tmp_path / "ce_adj_new.npy"
    np.save(adj_path, np.zeros((3, 3), dtype=np.float32))
    (tmp_path / "ce_nodes_new.txt").write_text("a\n\nc\n", encoding="utf-8")

    assert load_connectome_node_names(str(adj_path), 3) == ["0", "1", "2"]


def test_load_unknown_sign_weights_uses_new_connectome_suffix(tmp_path):
    adj_path = tmp_path / "ce_adj_new.npy"
    unknown_path = tmp_path / "ce_unknown_sign_weights_new.npy"
    np.save(adj_path, np.zeros((2, 2), dtype=np.float32))
    np.save(unknown_path, np.array([[0, -3], [4, 0]], dtype=np.float32))

    W_unknown = load_unknown_sign_weights(str(adj_path), n_nodes=2)

    assert np.array_equal(
        W_unknown,
        np.array([[0, 3], [4, 0]], dtype=np.float32),
    )


def test_assign_random_unknown_signs_uses_exact_inhibitory_fraction():
    W_known = np.zeros((5, 5), dtype=np.float32)
    W_unknown = np.zeros((5, 5), dtype=np.float32)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 0)]
    for i, j in edges:
        W_unknown[i, j] = 1

    W = assign_random_unknown_signs(
        W_known,
        W_unknown,
        np.random.default_rng(0),
        inhibitory_fraction=0.2,
    )

    vals = W[W != 0]
    assert vals.size == 5
    assert np.count_nonzero(vals < 0) == 1
    assert np.count_nonzero(vals > 0) == 4

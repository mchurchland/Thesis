import numpy as np

from util.util import load_connectome


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

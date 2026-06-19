import numpy as np
import pandas as pd
import pytest

from util.read_xls import (
    build_matrix,
    new_cel_sign_is_unknown,
    new_cel_sign_to_edge_sign,
    process_new_cel,
)


def test_process_new_cel_uses_sign_column_and_tracks_unpredicted_edges():
    df = pd.DataFrame(
        {
            "Source": ["A", "A", "B", "C"],
            "Target": ["B", "C", "C", "A"],
            "Edge Weight": [2, 3, 4, 5],
            "Sign": ["+", "-", "complex", "no pred"],
        }
    )

    edge_map, unknown_map = process_new_cel(df, return_unknown=True)
    W, names = build_matrix(edge_map)
    W_unknown, _ = build_matrix(unknown_map, include_nodes=names)
    idx = {name: i for i, name in enumerate(names)}

    assert W[idx["A"], idx["B"]] == 2
    assert W[idx["A"], idx["C"]] == -3
    assert W[idx["B"], idx["C"]] == 0
    assert W[idx["C"], idx["A"]] == 0
    assert W_unknown[idx["A"], idx["B"]] == 0
    assert W_unknown[idx["A"], idx["C"]] == 0
    assert W_unknown[idx["B"], idx["C"]] == 4
    assert W_unknown[idx["C"], idx["A"]] == 5
    assert np.count_nonzero(W) == 2


def test_new_cel_sign_unknown_tokens():
    assert new_cel_sign_is_unknown("complex")
    assert new_cel_sign_is_unknown("no pred")
    assert new_cel_sign_is_unknown("unknown")
    assert not new_cel_sign_is_unknown("+")


def test_new_cel_sign_to_edge_sign_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown new_cel Sign value"):
        new_cel_sign_to_edge_sign("ambiguous")

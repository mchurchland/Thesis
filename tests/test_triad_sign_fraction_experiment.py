import numpy as np

from network_stats.triad_sign_fraction_experiment import triad_weight_summaries


def test_triad_weight_summary_scopes_and_weight_moments():
    W = np.zeros((4, 4), dtype=np.float32)
    W[0, 1] = 2.0
    W[1, 2] = -4.0
    W[2, 0] = 6.0
    W[2, 3] = 10.0

    rows, census, details = triad_weight_summaries(W)
    by_scope = {row["triad_scope"]: row for row in rows}

    assert by_scope["any_edge"]["triad_count"] == 4
    assert by_scope["weak_connected"]["triad_count"] == 3
    assert by_scope["closed"]["triad_count"] == 1

    closed = by_scope["closed"]
    assert closed["triad_avg_abs_w_mean"] == np.mean([2.0, 4.0, 6.0])
    assert closed["triad_avg_abs_w_std"] == 0.0
    assert closed["triad_avg_abs_w_min"] == np.mean([2.0, 4.0, 6.0])
    assert closed["triad_avg_abs_w_max"] == np.mean([2.0, 4.0, 6.0])
    assert closed["triad_cv_abs_w_mean"] == np.std([2.0, 4.0, 6.0]) / np.mean([2.0, 4.0, 6.0])

    assert census["030C"] == 1
    assert details["triples"].shape == (4, 3)

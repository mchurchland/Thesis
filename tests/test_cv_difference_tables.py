import numpy as np
import pandas as pd

from graph_hist import print_cv_difference_tables


def test_cv_difference_table_uses_paired_trial_differences(tmp_path):
    rows = []
    baseline = {"MC": [1.0, 2.0, 3.0], "IPC": [2.0, 2.0, 2.0]}
    control = {"MC": [1.5, 2.5, 3.5], "IPC": [1.0, 2.0, 3.0]}
    for metric in ("MC", "IPC"):
        for group_id, value in enumerate(baseline[metric]):
            rows.append(("real", "chunk_0", str(group_id), metric, value, 4))
        for group_id, value in enumerate(control[metric]):
            rows.append(("control", "chunk_0", str(group_id), metric, value, 4))
    dispersion = pd.DataFrame(
        rows,
        columns=["mode", "src", "group_id", "metric", "dispersion", "n_hparams"],
    )

    table = print_cv_difference_tables(dispersion, str(tmp_path), baseline_mode="real")

    mc = table[(table["metric"] == "MC") & (table["control_mode"] == "control")].iloc[0]
    ipc = table[(table["metric"] == "IPC") & (table["control_mode"] == "control")].iloc[0]
    assert mc["comparison_type"] == "paired"
    assert mc["n"] == 3
    assert mc["baseline_mean_cv"] == 2.0
    assert mc["control_mean_cv"] == 2.5
    assert mc["delta_cv"] == 0.5
    assert mc["pct_delta_cv"] == 25.0
    assert mc["ci95_low"] == 0.5
    assert mc["ci95_high"] == 0.5
    assert ipc["delta_cv"] == 0.0
    assert np.isclose(ipc["pct_delta_cv"], 0.0)
    assert not (tmp_path / "cv_mean_differences_vs_ce.csv").exists()
    assert (tmp_path / "cv_mean_differences_table.tex").is_file()
    assert not (tmp_path / "cv_mean_differences_memory_table.tex").exists()
    assert not (tmp_path / "cv_mean_differences_kernel_table.tex").exists()


def test_cv_difference_table_uses_pm1_not_pm1_shuffle_as_auto_baseline(tmp_path):
    rows = []
    for metric in ("MC", "IPC", "KR", "GR"):
        for group_id, value in enumerate([1.0, 1.2, 1.4]):
            rows.append(("local_sign+binary", "chunk_0", str(group_id), metric, value, 4))
        for group_id, value in enumerate([1.5, 1.7, 1.9]):
            rows.append(("global_sign_pres", "chunk_0", str(group_id), metric, value, 4))
    dispersion = pd.DataFrame(
        rows,
        columns=["mode", "src", "group_id", "metric", "dispersion", "n_hparams"],
    )

    table = print_cv_difference_tables(dispersion, str(tmp_path), baseline_mode="auto")

    assert set(table["baseline_mode"]) == {"local_sign+binary"}
    assert set(table["control_mode"]) == {"global_sign_pres"}
    assert (tmp_path / "cv_mean_differences_vs_local_sign_binary_table.tex").is_file()
    assert not (tmp_path / "cv_mean_differences_memory_vs_local_sign_binary_table.tex").exists()
    assert not (tmp_path / "cv_mean_differences_kernel_vs_local_sign_binary_table.tex").exists()

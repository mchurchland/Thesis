import unittest
from unittest import mock
import csv
import math
from pathlib import Path
import tempfile

import torch

from network_stats.run_one import (
    make_gr_input_streams,
    make_kr_input_streams,
    run_one,
    run_reservoir_stream_batch,
)
from network_stats.stats import compute_GR, compute_KR
from reservoir_variants import save_rows


class GeneralizationRankTests(unittest.TestCase):
    @staticmethod
    def _csv_row(*, mc: float, ipc: float, kr: float, gr: float) -> tuple:
        return (
            "real",
            "spectral_radius",
            1,
            0.95,
            0.5,
            0.1,
            0.0,
            mc,
            ipc,
            kr,
            gr,
            0.01,
            0.9,
            0.2,
            123,
            "chunk_0",
            1.2,
            1.2,
            0.95,
            0.8,
        )

    def test_kr_streams_are_seeded_independent_uniform_sequences(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        streams = make_kr_input_streams(
            n_streams=20,
            stream_length=10,
            n_inputs=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=generator,
        )

        self.assertEqual(tuple(streams.shape), (20, 10, 1))
        self.assertTrue(torch.all(streams >= -1.0))
        self.assertTrue(torch.all(streams <= 1.0))
        self.assertFalse(torch.equal(streams[0], streams[1]))

        repeated_generator = torch.Generator(device="cpu")
        repeated_generator.manual_seed(42)
        repeated = make_kr_input_streams(
            n_streams=20,
            stream_length=10,
            n_inputs=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=repeated_generator,
        )
        self.assertTrue(torch.equal(streams, repeated))

    def test_gr_streams_share_only_the_configured_tail(self) -> None:
        torch.manual_seed(12)
        streams = make_gr_input_streams(
            n_streams=20,
            stream_length=10,
            common_tail_length=3,
            n_inputs=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(tuple(streams.shape), (20, 10, 1))
        self.assertTrue(torch.equal(streams[:, -3:, :], streams[:1, -3:, :].expand(20, -1, -1)))
        self.assertFalse(torch.equal(streams[0, :-3, :], streams[1, :-3, :]))

    def test_gr_stream_generation_uses_its_dedicated_seed(self) -> None:
        def generate(seed: int) -> torch.Tensor:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            return make_gr_input_streams(
                n_streams=20,
                stream_length=10,
                common_tail_length=3,
                n_inputs=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
                generator=generator,
            )

        self.assertTrue(torch.equal(generate(42), generate(42)))
        self.assertFalse(torch.equal(generate(42), generate(43)))

    def test_stream_batch_returns_one_final_state_per_stream(self) -> None:
        streams = torch.tensor([[[0.1], [0.2]], [[-0.1], [0.2]]])
        states = run_reservoir_stream_batch(
            W=torch.eye(2) * 0.2,
            Win=torch.ones(2, 1),
            input_streams=streams,
            leak=1.0,
            initial_state=torch.zeros(2),
        )

        self.assertEqual(tuple(states.shape), (2, 2))

    def test_run_one_uses_final_states_from_reset_kr_and_gr_streams(self) -> None:
        with mock.patch(
                "network_stats.run_one.run_reservoir_stream_batch",
                wraps=run_reservoir_stream_batch,
            ) as batch_runner:
            run_one(
                W=torch.eye(4) * 0.2,
                Win=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
                leak=0.5,
                device=torch.device("cpu"),
                WASHOUT=2,
                T_TRAIN=8,
                T_TEST=4,
                MC_MAX_DELAY=1,
                IPC_MAX_DELAY=1,
                IPC_ORDERS=[1],
                RIDGE_ALPHA=1e-4,
                kr_num_streams=3,
                kr_stream_length=4,
                kr_seed=24,
                gr_num_streams=2,
                gr_stream_length=4,
                gr_common_tail_length=1,
                gr_seed=42,
            )

        self.assertEqual(batch_runner.call_count, 2)
        kr_call, gr_call = batch_runner.call_args_list
        kr_streams = kr_call.args[2]
        gr_streams = gr_call.args[2]
        kr_initial_state = kr_call.kwargs["initial_state"]
        gr_initial_state = gr_call.kwargs["initial_state"]
        self.assertEqual(tuple(kr_streams.shape), (3, 4, 1))
        self.assertEqual(tuple(gr_streams.shape), (2, 4, 1))
        self.assertTrue(torch.equal(kr_initial_state, torch.zeros(4)))
        self.assertTrue(torch.equal(gr_initial_state, torch.zeros(4)))

    def test_kr_returns_centered_shannon_effective_rank(self) -> None:
        states = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

        self.assertAlmostEqual(compute_KR(states), 2.0, places=6)

    def test_rank_only_skips_mc_ipc_trajectory_and_regressions(self) -> None:
        with (
            mock.patch("network_stats.run_one.run_reservoir_with_pre") as trajectory,
            mock.patch("network_stats.run_one.compute_MC") as mc,
            mock.patch("network_stats.run_one.compute_IPC") as ipc,
        ):
            scores = run_one(
                W=torch.eye(4) * 0.2,
                Win=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
                leak=0.5,
                device=torch.device("cpu"),
                WASHOUT=2,
                T_TRAIN=8,
                T_TEST=4,
                MC_MAX_DELAY=1,
                IPC_MAX_DELAY=1,
                IPC_ORDERS=[1],
                RIDGE_ALPHA=1e-4,
                kr_num_streams=4,
                kr_stream_length=4,
                gr_num_streams=4,
                rank_only=True,
            )

        trajectory.assert_not_called()
        mc.assert_not_called()
        ipc.assert_not_called()
        self.assertTrue(math.isnan(scores["MC"]))
        self.assertTrue(math.isnan(scores["IPC"]))
        self.assertIsInstance(scores["KR"], float)
        self.assertIsInstance(scores["GR"], float)

    def test_kr_only_skips_mc_ipc_and_gr(self) -> None:
        with (
            mock.patch("network_stats.run_one.run_reservoir_with_pre") as trajectory,
            mock.patch(
                "network_stats.run_one.run_reservoir_stream_batch",
                wraps=run_reservoir_stream_batch,
            ) as rank_runner,
            mock.patch("network_stats.run_one.compute_MC") as mc,
            mock.patch("network_stats.run_one.compute_IPC") as ipc,
            mock.patch("network_stats.run_one.compute_GR") as gr,
        ):
            scores = run_one(
                W=torch.eye(4) * 0.2,
                Win=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
                leak=0.5,
                device=torch.device("cpu"),
                WASHOUT=2,
                T_TRAIN=8,
                T_TEST=4,
                MC_MAX_DELAY=1,
                IPC_MAX_DELAY=1,
                IPC_ORDERS=[1],
                RIDGE_ALPHA=1e-4,
                kr_num_streams=4,
                kr_stream_length=4,
                kr_only=True,
            )

        trajectory.assert_not_called()
        self.assertEqual(rank_runner.call_count, 1)
        mc.assert_not_called()
        ipc.assert_not_called()
        gr.assert_not_called()
        self.assertTrue(math.isnan(scores["MC"]))
        self.assertTrue(math.isnan(scores["IPC"]))
        self.assertTrue(math.isnan(scores["GR"]))
        self.assertIsInstance(scores["KR"], float)

    def test_rank_only_csv_update_preserves_existing_mc_and_ipc(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "results.csv"
            save_rows(
                str(csv_path),
                [self._csv_row(mc=3.25, ipc=4.5, kr=7, gr=6)],
            )
            save_rows(
                str(csv_path),
                [self._csv_row(mc=float("nan"), ipc=float("nan"), kr=21, gr=13)],
                rank_only=True,
            )

            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["MC"]), 3.25)
        self.assertEqual(float(rows[0]["IPC"]), 4.5)
        self.assertEqual(float(rows[0]["KR"]), 21.0)
        self.assertEqual(float(rows[0]["GR"]), 13.0)

    def test_kr_only_csv_update_preserves_mc_ipc_and_gr(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "results.csv"
            save_rows(
                str(csv_path),
                [self._csv_row(mc=3.25, ipc=4.5, kr=7, gr=6)],
            )
            save_rows(
                str(csv_path),
                [self._csv_row(mc=float("nan"), ipc=float("nan"), kr=21, gr=float("nan"))],
                kr_only=True,
            )

            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["MC"]), 3.25)
        self.assertEqual(float(rows[0]["IPC"]), 4.5)
        self.assertEqual(float(rows[0]["KR"]), 21.0)
        self.assertEqual(float(rows[0]["GR"]), 6.0)

    def test_gr_returns_centered_shannon_effective_rank(self) -> None:
        states = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

        self.assertAlmostEqual(compute_GR(states), 2.0, places=6)

    def test_returns_zero_for_zero_state_matrix(self) -> None:
        self.assertEqual(compute_GR(torch.zeros(20, 4)), 0)

    def test_validates_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D state matrix"):
            compute_GR(torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_GR(torch.tensor([[0.0, float("nan")]]))


if __name__ == "__main__":
    unittest.main()

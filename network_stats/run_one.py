import torch
from torch import Tensor

# ---- repo helpers (reuse your utils/stats) ----
from network_stats.stats import compute_IPC, compute_KR, compute_GR, compute_MC


@torch.no_grad()
def run_reservoir_with_pre(
    W: Tensor,
    Win: Tensor,
    u: Tensor,
    leak: float,
    bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Basic ESN-style update loop with pre-activation tracking.
    See: https://github.com/cknd/pyESN/blob/master/pyESN.py for a similar state update.
    """
    N = W.shape[0]
    T = u.shape[0]
    z = torch.zeros(N, device=W.device)
    X = torch.zeros(T, N, device=W.device)
    Pre = torch.zeros(T, N, device=W.device)
    if bias is None:
        bias_vec = torch.zeros(N, device=W.device, dtype=W.dtype)
    else:
        bias_vec = bias.to(device=W.device, dtype=W.dtype).view(N)
    for t in range(T):
        pre = W.T @ z + (Win @ u[t:t+1, :].T).squeeze() + bias_vec
        h = torch.tanh(pre)
        z = (1 - leak) * z + leak * h
        X[t] = z
        Pre[t] = pre
    return X, Pre


@torch.no_grad()
def run_reservoir_stream_batch(
    W: Tensor,
    Win: Tensor,
    input_streams: Tensor,
    leak: float,
    initial_state: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Return one final reservoir state for every input stream in a batch."""
    if input_streams.dim() != 3:
        raise ValueError("input_streams must have shape [streams, time, inputs].")

    n_streams = input_streams.shape[0]
    n_nodes = W.shape[0]
    z = initial_state.to(device=W.device, dtype=W.dtype).view(1, n_nodes)
    z = z.expand(n_streams, n_nodes).clone()
    if bias is None:
        bias_vec = torch.zeros(n_nodes, device=W.device, dtype=W.dtype)
    else:
        bias_vec = bias.to(device=W.device, dtype=W.dtype).view(n_nodes)

    for t in range(input_streams.shape[1]):
        pre = z @ W + input_streams[:, t, :] @ Win.T + bias_vec
        z = (1 - leak) * z + leak * torch.tanh(pre)
    return z


def make_gr_input_streams(
    n_streams: int,
    stream_length: int,
    common_tail_length: int,
    n_inputs: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Construct the similar-input ensemble used by Vidamour/RCbench GR.

    Streams are independently sampled from U[-1, 1], except that the final
    ``common_tail_length`` input vectors are identical across every stream.
    The differing prefix represents task-irrelevant variation; the common
    tail defines the recent signal over which the reservoir should generalize.
    """
    if n_streams <= 0:
        raise ValueError("n_streams must be positive.")
    if stream_length <= 0:
        raise ValueError("stream_length must be positive.")
    if not 0 <= common_tail_length <= stream_length:
        raise ValueError("common_tail_length must lie in [0, stream_length].")

    streams = torch.rand(
        n_streams,
        stream_length,
        n_inputs,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * 2.0 - 1.0
    if common_tail_length:
        common_tail = torch.rand(
            common_tail_length,
            n_inputs,
            device=device,
            dtype=dtype,
            generator=generator,
        ) * 2.0 - 1.0
        streams[:, -common_tail_length:, :] = common_tail
    return streams


def make_kr_input_streams(
    n_streams: int,
    stream_length: int,
    n_inputs: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Construct the distinct-input ensemble used by Vidamour KR.

    Every value in every stream is sampled independently from U[-1, 1].
    Unlike GR, KR has no shared tail: all streams are mutually uncorrelated.
    """
    if n_streams <= 0:
        raise ValueError("n_streams must be positive.")
    if stream_length <= 0:
        raise ValueError("stream_length must be positive.")

    return torch.rand(
        n_streams,
        stream_length,
        n_inputs,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * 2.0 - 1.0


def run_one(W: Tensor, Win: Tensor, leak: float, device: torch.device,WASHOUT: int,
            T_TRAIN: int, T_TEST: int,
            MC_MAX_DELAY: int, IPC_MAX_DELAY: int, IPC_ORDERS: list[int],
            RIDGE_ALPHA: float, output_idx: Tensor | None = None,
            bias: Tensor | None = None,
            kr_rank_threshold: float = 1e-3,
            kr_num_streams: int = 200,
            kr_stream_length: int = 10,
            kr_seed: int = 0,
            gr_rank_threshold: float = 1e-3,
            gr_num_streams: int = 200,
            gr_stream_length: int = 10,
            gr_common_tail_length: int = 3,
            gr_seed: int = 0,
            rank_only: bool = False) -> dict:
    """
    End-to-end reservoir evaluation computing MC/IPC/KR/GR and controllability metrics.
    Wraps the ESN update plus metrics pipeline; see reservoirpy/pyESN for similar evaluation flows:
    [1] Jaeger, H. (2001). Short term memory in echo state networks.    """
    if not rank_only:
        T_total = WASHOUT + T_TRAIN + T_TEST
        u = (torch.rand(T_total, 1, device=device) * 2.0 - 1.0) ## rescale to [-1, 1]
        X, _ = run_reservoir_with_pre(W, Win, u, leak, bias=bias)

    # Vidamour et al. use 200 length-10 streams for both KR and GR. KR streams
    # are fully independent, while GR streams share their final three inputs.
    # Use at least one stream per observed node so neither rank is sample-capped.
    n_observed_nodes = (
        int(torch.as_tensor(output_idx).numel())
        if output_idx is not None
        else W.shape[0]
    )
    effective_kr_num_streams = max(kr_num_streams, n_observed_nodes) ## just a bit of protection
    effective_gr_num_streams = max(gr_num_streams, n_observed_nodes) ## just a bit of protection

    kr_generator = torch.Generator(device=device)
    kr_generator.manual_seed(kr_seed)
    kr_streams = make_kr_input_streams(
        n_streams=effective_kr_num_streams,
        stream_length=kr_stream_length,
        n_inputs=Win.shape[1],
        device=device,
        dtype=W.dtype,
        generator=kr_generator,
    )
    gr_generator = torch.Generator(device=device)
    gr_generator.manual_seed(gr_seed)
    gr_streams = make_gr_input_streams(
        n_streams=effective_gr_num_streams,
        stream_length=gr_stream_length,
        common_tail_length=gr_common_tail_length,
        n_inputs=Win.shape[1],
        device=device,
        dtype=W.dtype,
        generator=gr_generator,
    )

    # Vidamour reinitialises before every KR and GR sequence. Zero is this
    # ESN's natural common baseline and avoids contamination from the MC/IPC
    # trajectory above.
    rank_initial_state = torch.zeros(W.shape[0], device=device, dtype=W.dtype)
    X_kr = run_reservoir_stream_batch(
        W,
        Win,
        kr_streams,
        leak,
        initial_state=rank_initial_state,
        bias=bias,
    )
    X_gr = run_reservoir_stream_batch(
        W,
        Win,
        gr_streams,
        leak,
        initial_state=rank_initial_state,
        bias=bias,
    )
    if output_idx is not None:
        idx = torch.as_tensor(output_idx, device=device, dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError("output_idx must contain at least one node when provided.")
        if not rank_only:
            X = X.index_select(1, idx)
        X_kr = X_kr.index_select(1, idx)
        X_gr = X_gr.index_select(1, idx)

    if rank_only:
        MC_total = float("nan")
        IPC_total = float("nan")
    else:
        Xtr = X[WASHOUT:WASHOUT+T_TRAIN] ## t_train
        Xte = X[WASHOUT+T_TRAIN:] ## t_test
        utr = u[WASHOUT:WASHOUT+T_TRAIN] ## u_train
        ute = u[WASHOUT+T_TRAIN:] ## u_test
        MC_total, _ = compute_MC(Xtr, Xte, utr, ute, MC_MAX_DELAY, RIDGE_ALPHA,device)
        IPC_total = compute_IPC(
            Xtr,
            Xte,
            utr,
            ute,
            IPC_MAX_DELAY,
            RIDGE_ALPHA,
            device,
            IPC_ORDERS,
        )
    KR_val      = compute_KR(X_kr, threshold=kr_rank_threshold)
    GR_val      = compute_GR(X_gr, threshold=gr_rank_threshold)
    return dict(
        MC=MC_total, IPC=IPC_total, KR=KR_val, GR=GR_val,
    )

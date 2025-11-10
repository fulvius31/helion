from typing import Callable
import torch
import helion
import helion.language as hl


@helion.kernel(
    config=helion.Config(
        block_sizes=[16, 16, 256],
        indexing=["pointer", "pointer", "pointer"],
        l2_groupings=[32],
        load_eviction_policies=["last", "last"],
        loop_orders=[[0, 1]],
        num_stages=4,
        num_warps=4,
        pid_type="xyz",
        range_flattens=[None, False],
        range_multi_buffers=[None, True],
        range_num_stages=[0, 0],
        range_unroll_factors=[0, 0],
        range_warp_specializes=[None, True],
    ),
    static_shapes=True,
)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Shapes
    m, k = x.shape
    k2, n = y.shape
    assert k == k2, f"size mismatch {k} != {k2}"

    # Result dtype follows PyTorch promotion rules (bf16×bf16→bf16; bf16×fp32→fp32)
    out_dtype = torch.promote_types(x.dtype, y.dtype)
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)

    # GPU part
    for tile_m, tile_n in hl.tile([m, n]):
        # Accumulate in fp32 for numerical stability
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            # Use input dtypes; PyTorch addmm will promote as needed into fp32 acc
            a_tile = x[tile_m, tile_k]
            b_tile = y[tile_k, tile_n]
            acc = torch.addmm(acc, a_tile, b_tile)
        out[tile_m, tile_n] = acc.to(out_dtype)
    return out


def matmul_tritonbench(
    tb_op, a: torch.Tensor, b: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """TritonBench wrapper: MUST return a zero-arg callable for timing."""
    return lambda: matmul(a, b)

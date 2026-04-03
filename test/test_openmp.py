"""Tests for the OpenMP CPU backend."""

from __future__ import annotations

import math
import os
import unittest

import torch

import helion
import helion.language as hl

# --- Pointwise kernels ---


@helion.kernel(backend="openmp")
def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="openmp")
def _exp_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.exp(x[tile])
    return out


@helion.kernel(backend="openmp")
def _mul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * y[tile]
    return out


# --- Activation kernels ---


@helion.kernel(backend="openmp")
def _sigmoid_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.sigmoid(x[tile])
    return out


@helion.kernel(backend="openmp")
def _tanh_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.tanh(x[tile])
    return out


@helion.kernel(backend="openmp")
def _geglu_kernel(x: torch.Tensor) -> torch.Tensor:
    n = x.size(-1) // 2
    x1 = x[..., :n]
    x2 = x[..., n:]
    out = torch.empty_like(x1)
    for tile in hl.tile(x1.size()):
        out[tile] = x1[tile] * torch.sigmoid(x2[tile])
    return out


@helion.kernel(backend="openmp")
def _swiglu_kernel(x: torch.Tensor) -> torch.Tensor:
    n = x.size(-1) // 2
    x1 = x[..., :n]
    x2 = x[..., n:]
    out = torch.empty_like(x1)
    for tile in hl.tile(x1.size()):
        out[tile] = x1[tile] * x2[tile] * torch.sigmoid(x2[tile])
    return out


# --- Reduction kernels ---


@helion.kernel(backend="openmp")
def _sum_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size()):
        out[tile[0]] += x[tile].sum(-1)
    return out


@helion.kernel(backend="openmp")
def _softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        max_val = torch.amax(row, dim=-1, keepdim=True)
        exp_val = torch.exp(row - max_val)
        sum_val = torch.sum(exp_val, dim=-1, keepdim=True)
        out[tile_m, :] = exp_val / sum_val
    return out


@helion.kernel(backend="openmp")
def _rms_norm_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        ms = torch.mean(row * row, dim=-1, keepdim=True)
        out[tile_m, :] = row * torch.rsqrt(ms + 1e-5)
    return out


@helion.kernel(backend="openmp")
def _batch_softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        block = x[tile, :, :]
        max_val = torch.amax(block, dim=-1, keepdim=True)
        exp_val = torch.exp(block - max_val)
        sum_val = torch.sum(exp_val, dim=-1, keepdim=True)
        out[tile, :, :] = exp_val / sum_val
    return out


# --- Matmul kernels ---


@helion.kernel(backend="openmp")
def _matmul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.zeros(m, n, dtype=a.dtype, device=a.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = a[tile_m, :] @ b[:, tile_n]
    return out


@helion.kernel(backend="openmp")
def _bmm_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batch, m, k = a.size()
    _, _, n = b.size()
    out = torch.zeros(batch, m, n, dtype=a.dtype, device=a.device)
    for tile_b in hl.tile(batch):
        for tile_m in hl.tile(m):
            for tile_n in hl.tile(n):
                out[tile_b, tile_m, tile_n] = (
                    a[tile_b, tile_m, :] @ b[tile_b, :, tile_n]
                )
    return out


# --- Attention kernel ---


@helion.kernel(backend="openmp", static_shapes=True)
def _attention_kernel(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


# --- Additional kernels (Phase 4+) ---


@helion.kernel(backend="openmp")
def _kl_div_kernel(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        t = target[tile]
        out[tile] = t * (torch.log(t) - x[tile])
    return out


@helion.kernel(backend="openmp")
def _clamp_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.clamp(x[tile], min=-1.0, max=1.0)
    return out


@helion.kernel(backend="openmp")
def _leaky_relu_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        v = x[tile]
        out[tile] = torch.where(v > 0, v, v * 0.01)
    return out


class TestOpenMP(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["HELION_USE_DEFAULT_CONFIG"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("HELION_USE_DEFAULT_CONFIG", None)

    # --- Phase 1: Pointwise ---

    def test_add_1d(self) -> None:
        x = torch.randn(1024, device="cpu", dtype=torch.float32)
        y = torch.randn(1024, device="cpu", dtype=torch.float32)
        result = _add_kernel(x, y)
        torch.testing.assert_close(result, x + y, atol=1e-6, rtol=1e-5)

    def test_add_2d(self) -> None:
        x = torch.randn(64, 128, device="cpu", dtype=torch.float32)
        y = torch.randn(64, 128, device="cpu", dtype=torch.float32)
        result = _add_kernel(x, y)
        torch.testing.assert_close(result, x + y, atol=1e-6, rtol=1e-5)

    def test_exp_1d(self) -> None:
        x = torch.randn(1024, device="cpu", dtype=torch.float32)
        result = _exp_kernel(x)
        torch.testing.assert_close(result, torch.exp(x), atol=1e-6, rtol=1e-5)

    def test_mul_2d(self) -> None:
        x = torch.randn(64, 128, device="cpu", dtype=torch.float32)
        y = torch.randn(64, 128, device="cpu", dtype=torch.float32)
        result = _mul_kernel(x, y)
        torch.testing.assert_close(result, x * y, atol=1e-6, rtol=1e-5)

    # --- Phase 2: Activations ---

    def test_sigmoid(self) -> None:
        x = torch.randn(512, device="cpu", dtype=torch.float32)
        result = _sigmoid_kernel(x)
        torch.testing.assert_close(result, torch.sigmoid(x), atol=1e-6, rtol=1e-5)

    def test_tanh(self) -> None:
        x = torch.randn(512, device="cpu", dtype=torch.float32)
        result = _tanh_kernel(x)
        torch.testing.assert_close(result, torch.tanh(x), atol=1e-6, rtol=1e-5)

    def test_geglu(self) -> None:
        x = torch.randn(32, 128, device="cpu", dtype=torch.float32)
        result = _geglu_kernel(x)
        n = x.size(-1) // 2
        x1, x2 = x[..., :n], x[..., n:]
        torch.testing.assert_close(
            result, x1 * torch.sigmoid(x2), atol=1e-5, rtol=1e-5
        )

    def test_swiglu(self) -> None:
        x = torch.randn(32, 128, device="cpu", dtype=torch.float32)
        result = _swiglu_kernel(x)
        n = x.size(-1) // 2
        x1, x2 = x[..., :n], x[..., n:]
        torch.testing.assert_close(
            result, x1 * x2 * torch.sigmoid(x2), atol=1e-5, rtol=1e-5
        )

    # --- Phase 2: Reductions ---

    def test_sum_2d(self) -> None:
        x = torch.randn(64, 128, device="cpu", dtype=torch.float32)
        result = _sum_kernel(x)
        torch.testing.assert_close(result, x.sum(-1), atol=1e-4, rtol=1e-4)

    # --- Phase 3: Multi-dim reductions ---

    def test_softmax(self) -> None:
        x = torch.randn(32, 64, device="cpu", dtype=torch.float32)
        result = _softmax_kernel(x)
        torch.testing.assert_close(
            result, torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5
        )

    def test_rms_norm(self) -> None:
        x = torch.randn(32, 64, device="cpu", dtype=torch.float32)
        result = _rms_norm_kernel(x)
        ms = (x * x).mean(dim=-1, keepdim=True)
        expected = x * torch.rsqrt(ms + 1e-5)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_batch_softmax(self) -> None:
        x = torch.randn(4, 8, 16, device="cpu", dtype=torch.float32)
        result = _batch_softmax_kernel(x)
        torch.testing.assert_close(
            result, torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5
        )

    # --- Phase 3: Matmul ---

    def test_matmul(self) -> None:
        a = torch.randn(16, 32, device="cpu", dtype=torch.float32)
        b = torch.randn(32, 8, device="cpu", dtype=torch.float32)
        result = _matmul_kernel(a, b)
        torch.testing.assert_close(result, a @ b, atol=1e-4, rtol=1e-4)

    def test_bmm(self) -> None:
        a = torch.randn(2, 8, 16, device="cpu", dtype=torch.float32)
        b = torch.randn(2, 16, 4, device="cpu", dtype=torch.float32)
        result = _bmm_kernel(a, b)
        torch.testing.assert_close(result, torch.bmm(a, b), atol=1e-4, rtol=1e-4)

    # --- Attention ---

    def test_attention(self) -> None:
        q_t = torch.randn(1, 2, 32, 16, device="cpu", dtype=torch.float32)
        k_t = torch.randn(1, 2, 32, 16, device="cpu", dtype=torch.float32)
        v_t = torch.randn(1, 2, 32, 16, device="cpu", dtype=torch.float32)
        result = _attention_kernel(q_t, k_t, v_t)
        ref = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        torch.testing.assert_close(result, ref, atol=1e-4, rtol=1e-3)

    # --- Phase 4+: Additional kernels ---

    def test_kl_div(self) -> None:
        x = torch.randn(32, 16, device="cpu", dtype=torch.float32)
        target = torch.softmax(torch.randn(32, 16, device="cpu"), dim=-1)
        result = _kl_div_kernel(x, target)
        expected = target * (torch.log(target) - x)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_clamp(self) -> None:
        x = torch.randn(512, device="cpu", dtype=torch.float32) * 3
        result = _clamp_kernel(x)
        torch.testing.assert_close(
            result, torch.clamp(x, min=-1.0, max=1.0), atol=1e-6, rtol=1e-5
        )

    def test_leaky_relu(self) -> None:
        x = torch.randn(1024, device="cpu", dtype=torch.float32)
        result = _leaky_relu_kernel(x)
        expected = torch.where(x > 0, x, x * 0.01)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    # --- Edge cases ---

    def test_add_non_power_of_2(self) -> None:
        """Non-power-of-2 tensor sizes exercise boundary tile handling."""
        x = torch.randn(100, 200, device="cpu", dtype=torch.float32)
        y = torch.randn(100, 200, device="cpu", dtype=torch.float32)
        result = _add_kernel(x, y)
        torch.testing.assert_close(result, x + y, atol=1e-6, rtol=1e-5)

    def test_exp_float64(self) -> None:
        """Verify double precision works."""
        x = torch.randn(512, device="cpu", dtype=torch.float64)
        result = _exp_kernel(x)
        torch.testing.assert_close(result, torch.exp(x), atol=1e-12, rtol=1e-10)

    def test_add_size_1(self) -> None:
        """Size-1 dimension edge case."""
        x = torch.randn(1, 128, device="cpu", dtype=torch.float32)
        y = torch.randn(1, 128, device="cpu", dtype=torch.float32)
        result = _add_kernel(x, y)
        torch.testing.assert_close(result, x + y, atol=1e-6, rtol=1e-5)

    def test_softmax_large(self) -> None:
        """Larger softmax to exercise multi-tile reduction."""
        x = torch.randn(256, 512, device="cpu", dtype=torch.float32)
        result = _softmax_kernel(x)
        torch.testing.assert_close(
            result, torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5
        )

    def test_matmul_non_square(self) -> None:
        """Non-square matmul with non-power-of-2 dims."""
        a = torch.randn(100, 200, device="cpu", dtype=torch.float32)
        b = torch.randn(200, 50, device="cpu", dtype=torch.float32)
        result = _matmul_kernel(a, b)
        torch.testing.assert_close(result, a @ b, atol=1e-3, rtol=1e-3)

    # --- Codegen verification ---

    def test_codegen_no_triton_references(self) -> None:
        x = torch.randn(256, device="cpu", dtype=torch.float32)
        y = torch.randn(256, device="cpu", dtype=torch.float32)
        _add_kernel(x, y)
        bound = _add_kernel.bind((x, y))
        code = bound.to_triton_code()
        self.assertNotIn("tl.", code)
        self.assertNotIn("triton", code)
        self.assertIn("_default_cpu_launcher", code)

    def test_codegen_no_torch_tensor_warning(self) -> None:
        """Verify cast_expr doesn't produce torch.tensor() wrapping warnings."""
        import warnings

        x = torch.randn(32, 64, device="cpu", dtype=torch.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _softmax_kernel(x)
            tensor_warnings = [
                warning
                for warning in w
                if "torch.tensor" in str(warning.message)
                and "copy construct" in str(warning.message)
            ]
            self.assertEqual(
                len(tensor_warnings),
                0,
                f"Got {len(tensor_warnings)} torch.tensor copy-construct warnings",
            )


class TestOpenMPAutotuning(unittest.TestCase):
    """Test real autotuning (no HELION_USE_DEFAULT_CONFIG)."""

    def test_softmax_autotuning(self) -> None:
        """Verify the autotuner runs and selects a valid config."""
        # Temporarily remove default config override
        os.environ.pop("HELION_USE_DEFAULT_CONFIG", None)

        @helion.kernel(backend="openmp")
        def softmax_autotune(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                row = x[tile_m, :]
                max_val = torch.amax(row, dim=-1, keepdim=True)
                exp_val = torch.exp(row - max_val)
                sum_val = torch.sum(exp_val, dim=-1, keepdim=True)
                out[tile_m, :] = exp_val / sum_val
            return out

        x = torch.randn(128, 64, device="cpu", dtype=torch.float32)
        result = softmax_autotune(x)
        expected = torch.softmax(x, dim=-1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

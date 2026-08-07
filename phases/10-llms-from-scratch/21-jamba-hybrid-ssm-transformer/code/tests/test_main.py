"""Tests for the Jamba hybrid memory calculator.

Pins the lesson's structural claims: the KV cache is paid for by attention
layers only and grows with context, the SSM state is fixed-size, a hybrid
with every layer marked attention degenerates to the pure-Transformer
budget, and one with no attention layer degenerates to a constant.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    BYTES_BF16,
    BYTES_FP8,
    HybridConfig,
    fmt_bytes,
    kv_cache_bytes,
    ssm_state_bytes,
)

CTX_256K = 262_144


def hybrid(attn_layers: int, total_layers: int = 32, n_kv_heads: int = 32,
           ssm_state_size: int = 16) -> HybridConfig:
    return HybridConfig(
        name=f"{attn_layers}/{total_layers}",
        total_layers=total_layers,
        attn_layers=attn_layers,
        hidden=4096,
        n_q_heads=32,
        n_kv_heads=n_kv_heads,
        head_dim=128,
        ssm_state_size=ssm_state_size,
    )


class TestKvCache(unittest.TestCase):
    def test_only_attention_layers_pay_for_kv_cache(self) -> None:
        self.assertEqual(kv_cache_bytes(hybrid(attn_layers=0), CTX_256K, BYTES_BF16), 0)

    def test_kv_cache_is_linear_in_attention_layer_count(self) -> None:
        one = kv_cache_bytes(hybrid(attn_layers=1), CTX_256K, BYTES_BF16)
        for attn in (2, 4, 8, 32):
            with self.subTest(attn_layers=attn):
                self.assertEqual(
                    kv_cache_bytes(hybrid(attn_layers=attn), CTX_256K, BYTES_BF16),
                    attn * one,
                )

    def test_kv_cache_is_linear_in_context_length(self) -> None:
        cfg = hybrid(attn_layers=4)
        base = kv_cache_bytes(cfg, 8_192, BYTES_BF16)
        self.assertEqual(kv_cache_bytes(cfg, 16_384, BYTES_BF16), 2 * base)
        self.assertEqual(kv_cache_bytes(cfg, 32 * 8_192, BYTES_BF16), 32 * base)

    def test_jamba_one_to_seven_is_eight_times_smaller_than_full_mha(self) -> None:
        full_mha = kv_cache_bytes(hybrid(attn_layers=32), CTX_256K, BYTES_BF16)
        jamba = kv_cache_bytes(hybrid(attn_layers=4), CTX_256K, BYTES_BF16)
        self.assertEqual(full_mha, 8 * jamba)

    def test_hybrid_still_beats_the_gqa_transformer_baseline(self) -> None:
        gqa_transformer = kv_cache_bytes(
            hybrid(attn_layers=32, n_kv_heads=8), CTX_256K, BYTES_BF16
        )
        jamba = kv_cache_bytes(hybrid(attn_layers=4), CTX_256K, BYTES_BF16)
        self.assertEqual(gqa_transformer, 2 * jamba)

    def test_fp8_halves_the_cache(self) -> None:
        cfg = hybrid(attn_layers=4)
        self.assertEqual(
            2 * kv_cache_bytes(cfg, CTX_256K, BYTES_FP8),
            kv_cache_bytes(cfg, CTX_256K, BYTES_BF16),
        )


class TestSsmState(unittest.TestCase):
    def test_ssm_state_does_not_depend_on_context(self) -> None:
        """The whole point of the recurrence: state size is fixed, so the
        pure-SSM budget at 256k equals its budget at 8k."""
        pure_ssm = hybrid(attn_layers=0)
        short = (kv_cache_bytes(pure_ssm, 8_192, BYTES_BF16)
                 + ssm_state_bytes(pure_ssm, BYTES_BF16))
        long = (kv_cache_bytes(pure_ssm, CTX_256K, BYTES_BF16)
                + ssm_state_bytes(pure_ssm, BYTES_BF16))
        self.assertEqual(short, long)
        self.assertGreater(long, 0)

    def test_all_attention_hybrid_degenerates_to_a_transformer(self) -> None:
        """Switch every SSM layer off and the calculator must return exactly
        the pure-Transformer budget: no residual SSM state term."""
        transformer = hybrid(attn_layers=32)
        self.assertEqual(ssm_state_bytes(transformer, BYTES_BF16), 0)
        self.assertEqual(
            kv_cache_bytes(transformer, CTX_256K, BYTES_BF16)
            + ssm_state_bytes(transformer, BYTES_BF16),
            kv_cache_bytes(transformer, CTX_256K, BYTES_BF16),
        )

    def test_ssm_state_is_linear_in_ssm_layer_count(self) -> None:
        four_ssm = ssm_state_bytes(hybrid(attn_layers=28), BYTES_BF16)
        twenty_eight_ssm = ssm_state_bytes(hybrid(attn_layers=4), BYTES_BF16)
        self.assertEqual(twenty_eight_ssm, 7 * four_ssm)

    def test_ssm_state_is_negligible_next_to_a_long_context_cache(self) -> None:
        cfg = hybrid(attn_layers=4)
        kv = kv_cache_bytes(cfg, CTX_256K, BYTES_BF16)
        ssm = ssm_state_bytes(cfg, BYTES_BF16)
        self.assertLess(ssm / kv, 0.001)


class TestFormatting(unittest.TestCase):
    def test_units_step_up_every_1024(self) -> None:
        self.assertEqual(fmt_bytes(512), "512.00B")
        self.assertEqual(fmt_bytes(1024), "1.00KB")
        self.assertEqual(fmt_bytes(4 * 1024 ** 3), "4.00GB")

    def test_hybrid_at_256k_reported_in_gigabytes(self) -> None:
        cfg = hybrid(attn_layers=4)
        total = (kv_cache_bytes(cfg, CTX_256K, BYTES_BF16)
                 + ssm_state_bytes(cfg, BYTES_BF16))
        self.assertEqual(fmt_bytes(total), "16.00GB")


if __name__ == "__main__":
    unittest.main()

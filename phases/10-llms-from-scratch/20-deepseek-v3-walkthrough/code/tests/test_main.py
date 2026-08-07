"""Tests for the DeepSeek-V3 parameter / KV-cache calculator.

The lesson is a config walkthrough, so the tests pin the arithmetic it walks
through: that the per-component counts add up to the reported total, that the
only difference between total and active is the experts that did not fire,
that expert count and top-k move total and active independently, and that
MLA's latent cache is 4x smaller than the GQA reference at the same context.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    DEEPSEEK_V3,
    compute_components,
    compute_totals,
    fmt,
    mla_attention_params,
    rmsnorm_params,
    router_params,
    swiglu_mlp_params,
)

CTX_128K = 131_072


def variant(**overrides) -> dict:
    cfg = dict(DEEPSEEK_V3)
    cfg.update(overrides)
    return cfg


class TestComponentFormulas(unittest.TestCase):
    def test_swiglu_counts_three_projections(self) -> None:
        """gate + up + down, not two: this is where naive MoE counts go wrong."""
        self.assertEqual(swiglu_mlp_params(7168, 2048), 3 * 7168 * 2048)

    def test_mla_counts_both_low_rank_paths(self) -> None:
        expected = (
            7168 * 1536            # q down-projection
            + 1536 * 128 * 56      # q up-projection
            + 7168 * 512           # kv down-projection (the cached latent)
            + 512 * 128 * 56       # k up-projection
            + 512 * 128 * 56       # v up-projection
            + 128 * 56 * 7168      # output projection
        )
        self.assertEqual(
            mla_attention_params(7168, 128, 56, kv_lora=512, q_lora=1536), expected
        )

    def test_router_is_one_logit_row_per_expert(self) -> None:
        self.assertEqual(router_params(7168, 256), 7168 * 256)

    def test_rmsnorm_has_no_hidden_matrices(self) -> None:
        self.assertEqual(rmsnorm_params(7168), 2 * 7168)


class TestParameterAccounting(unittest.TestCase):
    def test_total_is_the_sum_of_its_components(self) -> None:
        c = compute_components(DEEPSEEK_V3)
        n_layers = DEEPSEEK_V3["num_hidden_layers"]
        first_dense = DEEPSEEK_V3["first_k_dense_layers"]
        n_moe = n_layers - first_dense
        rebuilt = (
            c.embedding
            + first_dense * (c.attention_per_layer + c.dense_mlp_per_layer
                             + c.rmsnorm_per_layer)
            + n_moe * (c.attention_per_layer
                       + DEEPSEEK_V3["num_experts"] * c.expert_mlp_each
                       + c.shared_expert
                       + c.router_per_layer
                       + c.rmsnorm_per_layer)
            + c.final_norm
            + c.mtp_module
        )
        self.assertEqual(compute_totals(DEEPSEEK_V3).total, rebuilt)

    def test_active_differs_from_total_only_by_idle_experts_and_mtp(self) -> None:
        report = compute_totals(DEEPSEEK_V3)
        c = compute_components(DEEPSEEK_V3)
        n_moe = DEEPSEEK_V3["num_hidden_layers"] - DEEPSEEK_V3["first_k_dense_layers"]
        idle = (DEEPSEEK_V3["num_experts"] - DEEPSEEK_V3["num_experts_per_tok"])
        self.assertEqual(
            report.total - report.active,
            n_moe * idle * c.expert_mlp_each + c.mtp_module,
        )

    def test_published_config_is_sparse(self) -> None:
        report = compute_totals(DEEPSEEK_V3)
        self.assertLess(report.active, report.total)
        self.assertGreater(report.active_ratio, 0.03)
        self.assertLess(report.active_ratio, 0.08)

    def test_all_dense_config_has_nothing_to_skip(self) -> None:
        """Edge case: no MoE layers at all, so total and active differ only by
        the MTP module that inference does not run."""
        cfg = variant(first_k_dense_layers=DEEPSEEK_V3["num_hidden_layers"])
        report = compute_totals(cfg)
        c = compute_components(cfg)
        self.assertEqual(report.total - report.active, c.mtp_module)
        self.assertGreater(report.active_ratio, 0.95)

    def test_doubling_experts_grows_total_and_leaves_active_flat(self) -> None:
        base = compute_totals(DEEPSEEK_V3)
        wide = compute_totals(variant(num_experts=512))
        self.assertGreater(wide.total, 1.9 * base.total)
        self.assertLess(abs(wide.active - base.active) / base.active, 0.01)
        self.assertLess(wide.active_ratio, base.active_ratio)

    def test_doubling_top_k_adds_exactly_eight_experts_of_active_mass(self) -> None:
        base = compute_totals(DEEPSEEK_V3)
        deep = compute_totals(variant(num_experts_per_tok=16))
        c = compute_components(DEEPSEEK_V3)
        n_moe = DEEPSEEK_V3["num_hidden_layers"] - DEEPSEEK_V3["first_k_dense_layers"]
        self.assertEqual(deep.active - base.active, n_moe * 8 * c.expert_mlp_each)
        self.assertEqual(deep.total, base.total)

    def test_experts_dominate_the_moe_block(self) -> None:
        report = compute_totals(DEEPSEEK_V3)
        c = compute_components(DEEPSEEK_V3)
        experts = DEEPSEEK_V3["num_experts"] * c.expert_mlp_each
        self.assertGreater(experts / report.per_layer_moe_block, 0.95)
        overhead = c.router_per_layer + c.rmsnorm_per_layer
        self.assertLess(overhead / report.per_layer_moe_block, 0.01)


class TestKvCache(unittest.TestCase):
    def test_mla_latent_cache_is_four_times_smaller_than_gqa(self) -> None:
        report = compute_totals(DEEPSEEK_V3, ctx=CTX_128K)
        self.assertEqual(report.gqa_kv_cache_bytes_ref, 4 * report.kv_cache_bytes)

    def test_cache_is_linear_in_context(self) -> None:
        short = compute_totals(DEEPSEEK_V3, ctx=1024)
        long = compute_totals(DEEPSEEK_V3, ctx=2048)
        self.assertEqual(2 * short.kv_cache_bytes, long.kv_cache_bytes)
        self.assertEqual(2 * short.gqa_kv_cache_bytes_ref, long.gqa_kv_cache_bytes_ref)

    def test_halving_the_mla_rank_halves_the_cache(self) -> None:
        base = compute_totals(DEEPSEEK_V3, ctx=CTX_128K)
        narrow = compute_totals(variant(kv_lora_rank=256), ctx=CTX_128K)
        self.assertEqual(2 * narrow.kv_cache_bytes, base.kv_cache_bytes)
        self.assertEqual(narrow.gqa_kv_cache_bytes_ref, base.gqa_kv_cache_bytes_ref)
        self.assertLess(narrow.total, base.total)

    def test_context_defaults_to_the_config_window(self) -> None:
        report = compute_totals(DEEPSEEK_V3)
        expected = (DEEPSEEK_V3["num_hidden_layers"] * DEEPSEEK_V3["kv_lora_rank"]
                    * DEEPSEEK_V3["max_position_embeddings"] * 2)
        self.assertEqual(report.kv_cache_bytes, expected)


class TestFormatting(unittest.TestCase):
    def test_magnitude_suffixes(self) -> None:
        self.assertEqual(fmt(999), "999")
        self.assertEqual(fmt(1_500), "1.5K")
        self.assertEqual(fmt(2_500_000), "2.5M")
        self.assertEqual(fmt(671_000_000_000), "671.0B")


if __name__ == "__main__":
    unittest.main()

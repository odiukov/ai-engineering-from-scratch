"""Tests for the open-model architecture calculator.

Pins the six knobs the lesson teaches: RMSNorm vs LayerNorm parameter cost,
SwiGLU's third matrix, the KV-cache formula and what GQA/MQA/MLA do to it,
and the total-vs-active parameter split that MoE creates.
"""

from __future__ import annotations

import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    CONFIGS,
    analyze,
    attention_params_per_layer,
    attention_scheme,
    fmt_billions,
    fmt_bytes,
    layer_norm_params_per_layer,
    mlp_params,
)


def variant(name: str, **overrides) -> dict:
    config = copy.deepcopy(CONFIGS[name])
    config.update(overrides)
    return config


class TestAttentionScheme(unittest.TestCase):
    def test_equal_q_and_kv_head_counts_is_mha(self) -> None:
        self.assertEqual(attention_scheme(CONFIGS["gpt2-small"]), "MHA")

    def test_fewer_kv_heads_than_q_heads_is_gqa_and_reports_the_ratio(self) -> None:
        self.assertEqual(attention_scheme(CONFIGS["llama3-8b"]), "GQA (32/8)")
        self.assertEqual(attention_scheme(CONFIGS["llama3-70b"]), "GQA (64/8)")

    def test_a_single_kv_head_is_mqa(self) -> None:
        self.assertEqual(
            attention_scheme(variant("llama3-8b", num_key_value_heads=1)), "MQA"
        )

    def test_a_latent_kv_projection_is_mla_regardless_of_head_counts(self) -> None:
        config = CONFIGS["deepseek-v3"]
        self.assertEqual(config["num_attention_heads"], config["num_key_value_heads"])
        self.assertEqual(attention_scheme(config), "MLA")


class TestComponentParams(unittest.TestCase):
    def test_swiglu_needs_three_matrices_where_gelu_needs_two(self) -> None:
        h, ff = 4096, 14336
        self.assertEqual(mlp_params(h, ff, "gelu"), 2 * h * ff)
        self.assertEqual(mlp_params(h, ff, "swiglu"), 3 * h * ff)

    def test_rmsnorm_costs_half_of_layernorm_per_block(self) -> None:
        rms = layer_norm_params_per_layer(variant("llama3-8b"))
        ln = layer_norm_params_per_layer(variant("gpt2-small"))
        self.assertEqual(rms, 2 * 4096)
        self.assertEqual(ln, 4 * 768)
        self.assertEqual(
            layer_norm_params_per_layer(variant("llama3-8b", norm="layernorm")),
            2 * rms,
        )

    def test_gqa_shrinks_only_the_kv_projections(self) -> None:
        h = 4096
        gqa = attention_params_per_layer(CONFIGS["llama3-8b"])
        mha = attention_params_per_layer(variant("llama3-8b", num_key_value_heads=32))
        # Q and O are untouched; K and V drop by the group factor of 4.
        self.assertEqual(mha, 4 * h * h)
        self.assertEqual(mha - gqa, 2 * h * h - 2 * h * (8 * 128))

    def test_mla_caches_a_latent_instead_of_one_tensor_per_head(self) -> None:
        # Same hidden size, same head count, only the KV path differs.
        mla = analyze("deepseek-v3", CONFIGS["deepseek-v3"])
        as_mha = analyze("x", variant("deepseek-v3", attention="mha"))
        self.assertEqual(attention_scheme(variant("deepseek-v3", attention="mha")),
                         "MHA")
        self.assertLess(mla.kv_cache_bytes_bf16, as_mha.kv_cache_bytes_bf16 / 10)
        self.assertLess(mla.attn_params_per_layer, as_mha.attn_params_per_layer)
        # The cache holds one 512-wide latent per layer per token, twice over.
        self.assertEqual(mla.kv_cache_bytes_bf16, 2 * 61 * 512 * 131072 * 2)


class TestKVCache(unittest.TestCase):
    def test_llama3_8b_matches_the_documented_formula(self) -> None:
        b = analyze("llama3-8b", CONFIGS["llama3-8b"])
        expected = 2 * 32 * 8 * 128 * 131072 * 2
        self.assertEqual(b.kv_cache_bytes_bf16, expected)
        self.assertEqual(expected, 17_179_869_184)

    def test_gqa_cache_is_the_mha_cache_divided_by_the_group_count(self) -> None:
        gqa = analyze("llama3-8b", CONFIGS["llama3-8b"])
        mha = analyze("mha", variant("llama3-8b", num_key_value_heads=32))
        self.assertEqual(mha.kv_cache_bytes_bf16, 4 * gqa.kv_cache_bytes_bf16)

    def test_mqa_cache_is_the_mha_cache_divided_by_the_head_count(self) -> None:
        mqa = analyze("mqa", variant("llama3-8b", num_key_value_heads=1))
        mha = analyze("mha", variant("llama3-8b", num_key_value_heads=32))
        self.assertEqual(mha.kv_cache_bytes_bf16, 32 * mqa.kv_cache_bytes_bf16)

    def test_cache_scales_linearly_with_context_length(self) -> None:
        short = analyze("s", variant("llama3-8b", max_position_embeddings=8192))
        long = analyze("l", variant("llama3-8b", max_position_embeddings=16384))
        self.assertEqual(2 * short.kv_cache_bytes_bf16, long.kv_cache_bytes_bf16)

    def test_llama3_8b_cache_at_max_context_outgrows_its_bf16_weights(self) -> None:
        b = analyze("llama3-8b", CONFIGS["llama3-8b"])
        weights_bytes = 2 * b.total_params
        self.assertGreater(b.kv_cache_bytes_bf16, weights_bytes)


class TestTotalAndActiveParams(unittest.TestCase):
    def test_dense_models_activate_every_parameter(self) -> None:
        for name in ("gpt2-small", "mistral-7b", "llama3-8b", "llama3-70b",
                     "qwen2.5-72b"):
            b = analyze(name, CONFIGS[name])
            self.assertEqual(b.total_params, b.active_params, name)

    def test_moe_activates_only_the_routed_experts(self) -> None:
        for name in ("mixtral-8x7b", "deepseek-v3"):
            b = analyze(name, CONFIGS[name])
            self.assertLess(b.active_params, b.total_params, name)

    def test_deepseek_is_sparser_than_mixtral(self) -> None:
        mixtral = analyze("mixtral-8x7b", CONFIGS["mixtral-8x7b"])
        deepseek = analyze("deepseek-v3", CONFIGS["deepseek-v3"])
        mixtral_ratio = mixtral.active_params / mixtral.total_params
        deepseek_ratio = deepseek.active_params / deepseek.total_params
        self.assertLess(deepseek_ratio, mixtral_ratio)
        self.assertLess(deepseek_ratio, 0.10)

    def test_published_totals_are_reproduced_from_config_alone(self) -> None:
        # The calculator omits the untied LM head, so it lands a few percent
        # under the published number. Anything wider than that is a bug.
        published = {
            "gpt2-small": 124e6,
            "mistral-7b": 7.2e9,
            "llama3-8b": 8.0e9,
            "llama3-70b": 70e9,
            "mixtral-8x7b": 47e9,
            "qwen2.5-72b": 72e9,
            "deepseek-v3": 671e9,
        }
        for name, target in published.items():
            b = analyze(name, CONFIGS[name])
            self.assertLess(
                abs(b.total_params - target) / target, 0.10,
                f"{name}: computed {b.total_params}, published {target}",
            )

    def test_mixtral_active_params_match_the_published_thirteen_billion(self) -> None:
        b = analyze("mixtral-8x7b", CONFIGS["mixtral-8x7b"])
        self.assertLess(abs(b.active_params - 13e9) / 13e9, 0.05)

    def test_moe_ratio_reports_the_expert_width_not_the_dense_width(self) -> None:
        b = analyze("deepseek-v3", CONFIGS["deepseek-v3"])
        self.assertAlmostEqual(b.mlp_ratio, 2048 / 7168, places=6)
        dense = analyze("llama3-8b", CONFIGS["llama3-8b"])
        self.assertAlmostEqual(dense.mlp_ratio, 14336 / 4096, places=6)
        self.assertAlmostEqual(dense.mlp_ratio, 3.5, places=6)

    def test_first_dense_layers_are_excluded_from_the_expert_count(self) -> None:
        with_dense = analyze("deepseek-v3", CONFIGS["deepseek-v3"])
        all_moe = analyze("x", variant("deepseek-v3", first_dense_layers=0))
        self.assertGreater(all_moe.total_params, with_dense.total_params)


class TestVerdictAndFormatting(unittest.TestCase):
    def test_verdict_names_every_knob_the_config_moved(self) -> None:
        verdict = analyze("deepseek-v3", CONFIGS["deepseek-v3"]).verdict
        for token in ("RMSNORM", "SWIGLU", "ROPE", "MLA", "MoE 256e/top-8"):
            self.assertIn(token, verdict)
        baseline = analyze("gpt2-small", CONFIGS["gpt2-small"]).verdict
        for token in ("LAYERNORM", "GELU", "LEARNED", "MHA"):
            self.assertIn(token, baseline)
        self.assertNotIn("MoE", baseline)

    def test_bytes_are_formatted_in_decimal_si_units(self) -> None:
        self.assertEqual(fmt_bytes(999), "999.0B")
        self.assertEqual(fmt_bytes(1000), "1.0KB")
        self.assertEqual(fmt_bytes(17_179_869_184), "17.2GB")

    def test_params_are_formatted_by_magnitude(self) -> None:
        self.assertEqual(fmt_billions(7_500_000_000), "7.5B")
        self.assertEqual(fmt_billions(123_600_000), "123.6M")
        self.assertEqual(fmt_billions(8192), "8,192")


if __name__ == "__main__":
    unittest.main()

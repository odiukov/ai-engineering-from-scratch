"""Tests for the end-to-end LLM pipeline orchestrator.

Pins the properties the lesson claims: a valid DAG, a content-addressed
artifact store, a hash chain linking each stage to its upstream outputs,
replay reproducibility under a fixed seed, and gates that hold the run.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    ArtifactStore,
    DEFAULT_GATES,
    Manifest,
    STAGES,
    gate,
    manifest_to_json,
    plan,
    run,
    simulate_stage,
)


PASSING_EVAL = {
    "mmlu": 68.4,
    "humaneval": 42.1,
    "truthfulqa": 53.7,
    "safety_refusal_rate": 0.03,
    "kl_from_reference": 18.5,
    "cost_total_usd": 1941.0,
}


def fresh(**kwargs) -> tuple[Manifest, ArtifactStore]:
    return Manifest(seed=42, **kwargs), ArtifactStore()


class TestDag(unittest.TestCase):
    def test_every_dependency_is_declared_before_its_consumer(self) -> None:
        seen: set[str] = set()
        for name, deps, _stage_type in STAGES:
            for d in deps:
                self.assertIn(d, seen, f"{name} depends on unresolved {d}")
            seen.add(name)

    def test_stage_names_are_unique(self) -> None:
        names = [name for name, _, _ in STAGES]
        self.assertEqual(len(names), len(set(names)))

    def test_dpo_and_ppo_share_the_sft_parent_and_rejoin_at_cai(self) -> None:
        deps = {name: set(d) for name, d, _ in STAGES}
        self.assertEqual(deps["07_reward_ppo_policy"], {"06_sft_checkpoint"})
        self.assertEqual(deps["08_dpo_policy"], {"06_sft_checkpoint"})
        self.assertEqual(
            deps["09_cai_grpo_policy"],
            {"07_reward_ppo_policy", "08_dpo_policy"},
        )

    def test_plan_lists_every_stage_and_gate_without_running_anything(self) -> None:
        manifest, store = fresh()
        text = plan(manifest)
        for name, _, _ in STAGES:
            self.assertIn(name, text)
        for metric in DEFAULT_GATES:
            self.assertIn(metric, text)
        self.assertEqual(len(store), 0)
        self.assertEqual(manifest.stages, [])


class TestArtifactStore(unittest.TestCase):
    def test_identical_bytes_get_one_address(self) -> None:
        store = ArtifactStore()
        first = store.put(b"checkpoint-bytes")
        second = store.put(b"checkpoint-bytes")
        self.assertEqual(first, second)
        self.assertEqual(len(store), 1)

    def test_different_bytes_get_different_addresses(self) -> None:
        store = ArtifactStore()
        a = store.put(b"checkpoint-a")
        b = store.put(b"checkpoint-b")
        self.assertNotEqual(a, b)
        self.assertEqual(len(store), 2)

    def test_get_returns_the_stored_blob(self) -> None:
        store = ArtifactStore()
        h = store.put(b"weights")
        self.assertEqual(store.get(h), b"weights")
        self.assertTrue(store.has(h))
        self.assertFalse(store.has("0" * 64))


class TestRun(unittest.TestCase):
    def test_all_twelve_stages_complete(self) -> None:
        manifest, store = fresh()
        run(manifest, store)
        self.assertEqual(len(manifest.stages), len(STAGES))
        self.assertTrue(all(s.status == "ok" for s in manifest.stages))
        self.assertEqual(len(store), len(STAGES))

    def test_input_hashes_equal_the_upstream_output_hashes(self) -> None:
        manifest, store = fresh()
        run(manifest, store)
        produced = {s.name: s.output_hash for s in manifest.stages}
        deps = {name: d for name, d, _ in STAGES}
        for record in manifest.stages:
            expected = [produced[d] for d in deps[record.name]]
            self.assertEqual(record.input_hashes, expected)
            for h in record.input_hashes:
                self.assertTrue(store.has(h))

    def test_replay_with_the_same_seed_reproduces_every_hash(self) -> None:
        first, store_a = fresh()
        second, store_b = fresh()
        run(first, store_a)
        run(second, store_b)
        self.assertEqual(
            [s.output_hash for s in first.stages],
            [s.output_hash for s in second.stages],
        )
        self.assertEqual(first.total_cost_usd, second.total_cost_usd)

    def test_replay_into_a_warm_store_adds_no_new_artifacts(self) -> None:
        first, store = fresh()
        run(first, store)
        size_after_first = len(store)
        second = Manifest(seed=42)
        run(second, store)
        self.assertEqual(len(store), size_after_first)

    def test_changing_the_seed_changes_every_artifact(self) -> None:
        default, store_a = fresh()
        reseeded, store_b = Manifest(seed=7), ArtifactStore()
        run(default, store_a)
        run(reseeded, store_b)
        for a, b in zip(default.stages, reseeded.stages):
            self.assertNotEqual(a.output_hash, b.output_hash, a.name)

    def test_total_cost_is_the_sum_of_stage_costs(self) -> None:
        manifest, store = fresh()
        run(manifest, store)
        self.assertAlmostEqual(
            manifest.total_cost_usd,
            sum(s.cost_usd for s in manifest.stages),
            places=6,
        )

    def test_run_halts_at_the_stage_that_breaches_the_budget(self) -> None:
        manifest, store = fresh(budget_usd=100.0)
        run(manifest, store)
        self.assertLess(len(manifest.stages), len(STAGES))
        self.assertEqual(manifest.stages[-1].status, "halted_over_budget")
        self.assertEqual(manifest.stages[-1].output_hash, "")
        self.assertTrue(all(s.status == "ok" for s in manifest.stages[:-1]))
        self.assertGreater(manifest.total_cost_usd, manifest.budget_usd)
        self.assertEqual(manifest.eval_metrics, {})

    def test_simulate_stage_is_a_pure_function_of_its_arguments(self) -> None:
        a = simulate_stage("04_pretrained_base", "checkpoint", ["abc"], 42)
        b = simulate_stage("04_pretrained_base", "checkpoint", ["abc"], 42)
        c = simulate_stage("04_pretrained_base", "checkpoint", ["def"], 42)
        self.assertEqual(a, b)
        self.assertNotEqual(a[0], c[0])

    def test_expensive_stage_types_cost_more_than_cheap_ones(self) -> None:
        _, _, cheap = simulate_stage("x", "server_spec", [], 1)
        _, _, pricey = simulate_stage("y", "checkpoint", [], 1)
        self.assertLess(cheap, pricey)


class TestGate(unittest.TestCase):
    def test_all_metrics_inside_thresholds_ships(self) -> None:
        manifest, store = fresh()
        run(manifest, store, injected_eval=dict(PASSING_EVAL))
        ok, reasons = gate(manifest)
        self.assertTrue(ok)
        self.assertTrue(manifest.shippable)
        self.assertEqual(len(reasons), len(DEFAULT_GATES))
        self.assertTrue(all(r.startswith("PASS") for r in reasons))

    def test_a_regression_below_a_lower_bound_holds_the_run(self) -> None:
        manifest, store = fresh()
        run(manifest, store, injected_eval={**PASSING_EVAL, "mmlu": 42.0})
        ok, reasons = gate(manifest)
        self.assertFalse(ok)
        self.assertFalse(manifest.shippable)
        self.assertEqual(sum(r.startswith("HOLD") for r in reasons), 1)
        self.assertTrue(any("mmlu" in r and r.startswith("HOLD") for r in reasons))

    def test_exceeding_the_kl_budget_holds_the_run(self) -> None:
        manifest, store = fresh()
        run(manifest, store, injected_eval={**PASSING_EVAL, "kl_from_reference": 40.0})
        ok, reasons = gate(manifest)
        self.assertFalse(ok)
        self.assertTrue(
            any("kl_from_reference" in r and r.startswith("HOLD") for r in reasons)
        )

    def test_a_metric_exactly_on_the_threshold_passes(self) -> None:
        manifest, store = fresh()
        on_the_line = {
            metric: g["value"] for metric, g in DEFAULT_GATES.items()
        }
        run(manifest, store, injected_eval=on_the_line)
        ok, _ = gate(manifest)
        self.assertTrue(ok)

    def test_a_missing_metric_holds_rather_than_silently_passing(self) -> None:
        manifest, store = fresh()
        partial = dict(PASSING_EVAL)
        del partial["safety_refusal_rate"]
        run(manifest, store, injected_eval=partial)
        ok, reasons = gate(manifest)
        self.assertFalse(ok)
        self.assertIn("HOLD: missing metric safety_refusal_rate", reasons)


class TestManifestSerialization(unittest.TestCase):
    def test_json_round_trip_keeps_the_hash_chain(self) -> None:
        manifest, store = fresh()
        run(manifest, store, injected_eval=dict(PASSING_EVAL))
        gate(manifest)
        parsed = json.loads(manifest_to_json(manifest))
        self.assertEqual(len(parsed["stages"]), len(STAGES))
        self.assertEqual(parsed["seed"], 42)
        self.assertTrue(parsed["shippable"])
        self.assertEqual(
            parsed["stages"][3]["input_hashes"],
            [parsed["stages"][2]["output_hash"]],
        )


if __name__ == "__main__":
    unittest.main()

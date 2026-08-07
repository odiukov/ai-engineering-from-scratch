"""Tests for the pipeline-schedule bubble simulator.

Pins the properties the lesson claims for DualPipe rather than the printed
percentages: bidirectional scheduling must beat 1F1B at the same pipeline
depth, the absolute bubble must not grow with the micro-batch count, and
DualPipeV must buy its single parameter copy with a slightly larger bubble.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    bubble_1f1b,
    bubble_dualpipe,
    bubble_dualpipev,
    bubble_zero_bubble,
    gpu_hours_recovered,
    summarize,
)

DEPTHS = (2, 4, 8, 16, 32, 64)
MICRO_BATCHES = (4, 8, 16, 32, 64, 128)


class TestScheduleOrdering(unittest.TestCase):
    def test_dualpipe_beats_1f1b_at_every_depth(self) -> None:
        for p in DEPTHS:
            for m in MICRO_BATCHES:
                with self.subTest(P=p, M=m):
                    self.assertLess(bubble_dualpipe(p, m), bubble_1f1b(p, m))

    def test_zero_bubble_sits_between_1f1b_and_dualpipe(self) -> None:
        for p in DEPTHS:
            for m in MICRO_BATCHES:
                with self.subTest(P=p, M=m):
                    self.assertLess(bubble_zero_bubble(p, m), bubble_1f1b(p, m))
                    self.assertLess(bubble_dualpipe(p, m), bubble_zero_bubble(p, m))

    def test_dualpipev_pays_a_bubble_premium_for_one_param_copy(self) -> None:
        rows = summarize(P=8, M=16)
        by_name = {name: (bubble, copies) for name, bubble, copies, _ in rows}
        dualpipe_bubble, dualpipe_copies = by_name["DualPipe"]
        dualpipev_bubble, dualpipev_copies = by_name["DualPipeV"]
        self.assertEqual(dualpipe_copies, 2)
        self.assertEqual(dualpipev_copies, 1)
        self.assertGreater(dualpipev_bubble, dualpipe_bubble)
        self.assertLess(dualpipev_bubble, by_name["1F1B"][0])

    def test_summarize_reports_every_schedule_once(self) -> None:
        names = [name for name, _, _, _ in summarize(P=8, M=16)]
        self.assertEqual(names, ["1F1B", "Zero Bubble", "DualPipe", "DualPipeV"])
        self.assertEqual(len(set(names)), 4)


class TestBubbleScaling(unittest.TestCase):
    def test_absolute_bubble_is_independent_of_micro_batch_count(self) -> None:
        """Every schedule's bubble is warmup + cooldown only: the fraction
        shrinks with M because total work grows, not because the bubble does."""
        for p in DEPTHS:
            chunks = set()
            for m in MICRO_BATCHES:
                total = 3 * m + (p - 1)
                chunks.add(round(bubble_dualpipe(p, m) * total, 9))
            with self.subTest(P=p):
                self.assertEqual(len(chunks), 1)

    def test_bubble_fraction_shrinks_as_micro_batches_grow(self) -> None:
        for fn in (bubble_1f1b, bubble_zero_bubble, bubble_dualpipe, bubble_dualpipev):
            for p in DEPTHS:
                fractions = [fn(p, m) for m in MICRO_BATCHES]
                with self.subTest(schedule=fn.__name__, P=p):
                    self.assertEqual(fractions, sorted(fractions, reverse=True))

    def test_bubble_fraction_grows_with_pipeline_depth(self) -> None:
        for fn in (bubble_1f1b, bubble_zero_bubble, bubble_dualpipe, bubble_dualpipev):
            fractions = [fn(p, 64) for p in DEPTHS]
            with self.subTest(schedule=fn.__name__):
                self.assertEqual(fractions, sorted(fractions))

    def test_single_stage_pipeline_has_no_bubble(self) -> None:
        for fn in (bubble_1f1b, bubble_zero_bubble, bubble_dualpipe, bubble_dualpipev):
            with self.subTest(schedule=fn.__name__):
                self.assertEqual(fn(1, 16), 0.0)

    def test_fractions_stay_in_the_unit_interval(self) -> None:
        for fn in (bubble_1f1b, bubble_zero_bubble, bubble_dualpipe, bubble_dualpipev):
            for p in DEPTHS:
                for m in MICRO_BATCHES:
                    with self.subTest(schedule=fn.__name__, P=p, M=m):
                        self.assertGreaterEqual(fn(p, m), 0.0)
                        self.assertLessEqual(fn(p, m), 1.0)


class TestRecoveredGpuHours(unittest.TestCase):
    def test_recovered_hours_equal_the_bubble_gap(self) -> None:
        report = gpu_hours_recovered(P=16, M=128, total_gpu_hours=2_800_000)
        gap = report["1F1B_bubble_frac"] - report["DualPipe_bubble_frac"]
        self.assertAlmostEqual(report["recovered_gpu_hours"], gap * 2_800_000, places=6)
        self.assertGreater(report["recovered_gpu_hours"], 0.0)

    def test_nothing_to_recover_without_a_pipeline(self) -> None:
        report = gpu_hours_recovered(P=1, M=128, total_gpu_hours=2_800_000)
        self.assertEqual(report["recovered_gpu_hours"], 0.0)

    def test_recovered_hours_scale_with_run_size(self) -> None:
        small = gpu_hours_recovered(P=16, M=128, total_gpu_hours=1_000)
        large = gpu_hours_recovered(P=16, M=128, total_gpu_hours=2_000)
        self.assertAlmostEqual(
            2 * small["recovered_gpu_hours"], large["recovered_gpu_hours"], places=6
        )


if __name__ == "__main__":
    unittest.main()

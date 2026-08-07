"""Tests for the Hogwild! shared-cache inference simulator.

Pins the bookkeeping the shared cache must never break — every worker step
lands in the cache exactly once, nothing is lost or double-counted — plus the
two claims the lesson makes about it: one worker is indistinguishable from
serial decoding, and coordination is what turns extra workers into progress
instead of redundancy.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import expected_speedup, run_hogwild  # noqa: E402

STEPS = 200
SEED = 7


def run(n_workers: int, coordination_weight: float) -> dict:
    return run_hogwild(
        n_workers=n_workers,
        step_budget=STEPS,
        target_per_category=100,
        coordination_weight=coordination_weight,
        seed=SEED,
    )


class TestSharedCacheBookkeeping(unittest.TestCase):
    def test_every_worker_step_writes_exactly_one_token(self) -> None:
        for n in (1, 2, 4, 8):
            with self.subTest(workers=n):
                result = run(n, 0.8)
                self.assertEqual(result["tokens_emitted"], n * STEPS)

    def test_no_token_is_lost_or_counted_twice(self) -> None:
        result = run(4, 0.8)
        counts = result["category_counts"]
        self.assertEqual(sum(counts.values()), result["tokens_emitted"])
        self.assertEqual(result["work_tokens"], counts["A"] + counts["B"])
        self.assertEqual(
            result["work_tokens"] + counts["noise"] + counts["coord"],
            result["tokens_emitted"],
        )

    def test_progress_never_exceeds_the_work_tokens_written(self) -> None:
        for n in (1, 2, 4, 8):
            with self.subTest(workers=n):
                result = run(n, 0.8)
                self.assertLessEqual(result["unique_progress"], result["work_tokens"])

    def test_at_most_one_unique_token_per_category_per_step(self) -> None:
        """Two work categories exist, so no step can add more than 2 progress
        however many workers race for the cache."""
        for n in (2, 4, 8):
            with self.subTest(workers=n):
                self.assertLessEqual(run(n, 1.0)["unique_progress"], 2 * STEPS)

    def test_same_seed_reproduces_the_run(self) -> None:
        first, second = run(2, 0.8), run(2, 0.8)
        self.assertEqual(first, second)

    def test_rates_are_per_step_averages(self) -> None:
        result = run(2, 0.8)
        self.assertAlmostEqual(result["tokens_per_step"], result["tokens_emitted"] / STEPS)
        self.assertAlmostEqual(result["work_per_step"], result["work_tokens"] / STEPS)
        self.assertAlmostEqual(
            result["progress_per_step"], result["unique_progress"] / STEPS
        )


class TestSingleWorkerIsSerial(unittest.TestCase):
    def test_one_worker_can_never_be_redundant(self) -> None:
        result = run(1, 0.8)
        self.assertEqual(result["unique_progress"], result["work_tokens"])
        self.assertEqual(result["progress_per_step"], result["work_per_step"])

    def test_one_worker_is_unaffected_by_the_coordination_weight_bookkeeping(self) -> None:
        for weight in (0.0, 0.5, 1.0):
            with self.subTest(coordination_weight=weight):
                result = run(1, weight)
                self.assertEqual(result["unique_progress"], result["work_tokens"])
                self.assertEqual(result["tokens_emitted"], STEPS)

    def test_adding_workers_never_loses_progress(self) -> None:
        serial = run(1, 0.8)["unique_progress"]
        for n in (2, 4):
            with self.subTest(workers=n):
                self.assertGreaterEqual(run(n, 0.8)["unique_progress"], serial)


class TestCoordinationDrivesTheSpeedup(unittest.TestCase):
    def test_stronger_coordination_produces_more_progress(self) -> None:
        progress = [run(2, w)["unique_progress"] for w in (0.0, 0.2, 0.5, 0.8, 1.0)]
        self.assertEqual(progress, sorted(progress))
        self.assertGreater(progress[-1], progress[0])

    def test_without_coordination_the_second_worker_is_redundant(self) -> None:
        """Both workers stay in category A, so most of the second worker's
        tokens duplicate the first's and the speedup stays near 1x."""
        uncoordinated = run(2, 0.0)
        self.assertLess(uncoordinated["unique_progress"], uncoordinated["work_tokens"])
        uncoordinated_speedup = (uncoordinated["unique_progress"]
                                 / run(1, 0.0)["unique_progress"])
        coordinated_speedup = (run(2, 1.0)["unique_progress"]
                               / run(1, 1.0)["unique_progress"])
        self.assertLess(uncoordinated_speedup, 1.3)
        self.assertGreater(coordinated_speedup - uncoordinated_speedup, 0.5)

    def test_full_coordination_nearly_doubles_a_two_worker_run(self) -> None:
        coordinated = run(2, 1.0)
        speedup = coordinated["unique_progress"] / run(1, 1.0)["unique_progress"]
        self.assertGreater(speedup, 1.5)
        self.assertLessEqual(coordinated["unique_progress"], 2 * STEPS)


class TestExpectedSpeedup(unittest.TestCase):
    def test_no_parallel_fraction_means_no_speedup(self) -> None:
        self.assertAlmostEqual(expected_speedup(10_000, p=0.0, c=0, N=4,
                                                steps_per_worker=2_500), 1.0)

    def test_free_coordination_reproduces_amdahl(self) -> None:
        for n in (2, 4, 8):
            with self.subTest(workers=n):
                self.assertAlmostEqual(
                    expected_speedup(10_000, p=0.7, c=0, N=n, steps_per_worker=1),
                    1.0 / (0.3 + 0.7 / n),
                )

    def test_amdahl_ceiling_is_never_exceeded(self) -> None:
        for n in (2, 4, 8, 64):
            with self.subTest(workers=n):
                self.assertLess(
                    expected_speedup(10_000, p=0.7, c=200, N=n, steps_per_worker=1),
                    1.0 / 0.3,
                )

    def test_coordination_overhead_can_make_parallel_slower_than_serial(self) -> None:
        for n in (2, 4, 8):
            with self.subTest(workers=n):
                self.assertLess(
                    expected_speedup(1_000, p=0.3, c=150, N=n, steps_per_worker=1), 1.0
                )

    def test_more_workers_stop_paying_once_overhead_dominates(self) -> None:
        speedups = [
            expected_speedup(10_000, p=0.7, c=200, N=n, steps_per_worker=1)
            for n in (2, 4, 8, 16, 32)
        ]
        best = speedups.index(max(speedups))
        self.assertGreater(best, 0)
        self.assertLess(best, len(speedups) - 1)
        self.assertLess(speedups[-1], speedups[best])


if __name__ == "__main__":
    unittest.main()

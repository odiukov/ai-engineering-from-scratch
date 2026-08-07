"""Тесты к уроку «AI Scientist v2: автономный научный цикл». Правь exercise.py."""

import random

import pytest

from exercise import (
    DEFAULT_CONFIG,
    REQUIRED_CHECKS,
    novelty_check,
    polish_figures,
    release_gate,
    review,
    run_experiment,
    run_loop,
    summarize,
    supports_conclusion,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ALL_CHECKS = dict.fromkeys(REQUIRED_CHECKS, True)


def paper(ok=True, flawed=False, effect=True, masked=False):
    """Собрать работу руками: результат эксперимента, вывод и полировка."""
    return {
        "claim": {"effect_observed": effect},
        "novelty": "novel",
        "experiment": {"ok": ok, "flawed": flawed, "retried": False},
        "masked": masked,
    }


# --------------------------------------------------------------- novelty_check
def test_a_genuinely_new_idea_is_always_called_novel():
    rng = random.Random(0)
    assert all(novelty_check(rng, False, 0.9) == "novel" for _ in range(100))


def test_a_perfect_search_never_mislabels_a_known_idea():
    rng = random.Random(0)
    assert all(novelty_check(rng, True, 0.0) == "known" for _ in range(100))


def test_mislabelling_happens_at_the_configured_rate():
    rng = random.Random(1)
    calls = [novelty_check(rng, True, 0.25) for _ in range(4000)]
    assert 0.22 < calls.count("novel") / len(calls) < 0.28


def test_the_error_is_one_sided_old_passes_for_new_never_the_reverse():
    """Конвейер завышает новизну, а не шумит вокруг правды."""
    rng = random.Random(2)
    assert "known" not in {novelty_check(rng, False, 1.0) for _ in range(200)}


def test_a_rate_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        novelty_check(random.Random(0), True, 1.7)


# -------------------------------------------------------------- run_experiment
def test_an_experiment_that_never_fails_is_clean():
    out = run_experiment(random.Random(0), 0.0, 0.0)
    assert out == {"ok": True, "flawed": False, "retried": False}


def test_an_unrecoverable_failure_stops_the_pipeline():
    out = run_experiment(random.Random(0), 1.0, 0.0)
    assert out["ok"] is False


def test_a_retry_recovered_experiment_keeps_a_residual_flaw():
    """Ретрай чинит падение, но не перепроверяет численный результат."""
    out = run_experiment(random.Random(0), 1.0, 1.0)
    assert out["ok"] is True
    assert out["flawed"] is True


def test_only_failed_experiments_are_ever_retried():
    rng = random.Random(3)
    runs = [run_experiment(rng, 0.42, 0.55) for _ in range(2000)]
    assert all(r["retried"] for r in runs if r["flawed"])
    assert not any(r["retried"] for r in runs if not r["flawed"])


def test_failure_rate_drives_the_share_of_flawed_results():
    rng = random.Random(4)
    low = [run_experiment(rng, 0.1, 0.9) for _ in range(2000)]
    high = [run_experiment(rng, 0.8, 0.9) for _ in range(2000)]
    assert sum(r["flawed"] for r in low) < sum(r["flawed"] for r in high)


# -------------------------------------------------------------- polish_figures
def test_nothing_to_mask_in_a_clean_experiment():
    out = polish_figures(random.Random(0), paper(flawed=False), 1.0)
    assert out["masked"] is False


def test_a_flawed_experiment_can_be_masked_completely():
    out = polish_figures(random.Random(0), paper(flawed=True), 1.0)
    assert out["masked"] is True


def test_polish_never_repairs_the_experiment_itself():
    """Меняется заметность изъяна, а не изъян."""
    out = polish_figures(random.Random(0), paper(flawed=True), 1.0)
    assert out["experiment"]["flawed"] is True


def test_polish_returns_a_copy_and_leaves_the_original_readable():
    original = paper(flawed=True)
    del original["masked"]
    polish_figures(random.Random(0), original, 1.0)
    assert "masked" not in original


def test_zero_mask_rate_leaves_the_flaw_visible():
    out = polish_figures(random.Random(0), paper(flawed=True), 0.0)
    assert out["masked"] is False


# --------------------------------------------------------- supports_conclusion
def test_a_clean_experiment_with_an_observed_effect_supports_the_claim():
    assert supports_conclusion(paper()) is True


def test_a_conclusion_without_an_observed_effect_is_rejected():
    """Научный цикл обязан отвергать вывод, которому нечего предъявить."""
    assert supports_conclusion(paper(effect=False)) is False


def test_a_flawed_experiment_cannot_support_anything():
    assert supports_conclusion(paper(flawed=True)) is False


def test_a_failed_experiment_cannot_support_anything():
    assert supports_conclusion(paper(ok=False, flawed=True)) is False


def test_polish_has_no_vote_in_the_substantive_check():
    """Проверка, которая смотрит на фигуры, перестаёт быть проверкой."""
    assert supports_conclusion(paper(flawed=True, masked=True)) is False
    assert supports_conclusion(paper(masked=True)) is True


# --------------------------------------------------------------------- review
def test_strict_reviewer_accepts_supported_work():
    assert review(paper()) == "accept"


def test_strict_reviewer_rejects_a_polished_broken_paper():
    assert review(paper(flawed=True, masked=True)) == "reject"


def test_weak_reviewer_accepts_the_same_polished_broken_paper():
    """Разница между проверкой результата и проверкой впечатления."""
    assert review(paper(flawed=True, masked=True), strict=False) == "accept"


def test_weak_reviewer_still_rejects_unpolished_broken_work():
    assert review(paper(flawed=True, masked=False), strict=False) == "reject"


def test_both_reviewers_agree_on_clean_work():
    assert review(paper()) == review(paper(), strict=False) == "accept"


# ---------------------------------------------------------------- release_gate
def test_clean_work_with_every_box_ticked_gets_out():
    assert release_gate(paper(), ALL_CHECKS) == (True, [])


def test_a_missing_box_counts_as_not_ticked():
    allowed, failed = release_gate(paper(), {})
    assert allowed is False
    assert failed == list(REQUIRED_CHECKS)


def test_broken_work_is_held_even_with_every_box_ticked():
    assert release_gate(paper(flawed=True), ALL_CHECKS) == (False, ["conclusion_supported"])


def test_extra_keys_invented_by_the_agent_are_ignored():
    """Гейт задаёт список проверок, а не тот, кого проверяют."""
    allowed, failed = release_gate(paper(), {**ALL_CHECKS, "looks_great": True})
    assert (allowed, failed) == (True, [])
    assert release_gate(paper(), {"looks_great": True})[0] is False


def test_failure_order_is_stable_across_calls():
    first = release_gate(paper(flawed=True), {"novelty_verified": True})[1]
    second = release_gate(paper(flawed=True), {"novelty_verified": True})[1]
    assert first == second == ["conclusion_supported", "experiment_reproduced",
                               "human_signoff"]


# ------------------------------------------------------------------- run_loop
def test_a_correctly_recognised_known_idea_is_dropped_at_novelty():
    config = {**DEFAULT_CONFIG, "novelty_mislabel": 0.0}
    out = run_loop(random.Random(0), config, is_known=True)
    assert out == {"submitted": False, "stage": "novelty", "paper": None,
                   "clean": False}


def test_an_unrecoverable_experiment_ends_the_run_there():
    config = {**DEFAULT_CONFIG, "experiment_failure": 1.0, "retry_recovery": 0.0}
    out = run_loop(random.Random(0), config, is_known=False)
    assert out["stage"] == "experiment"


def test_a_flawless_pipeline_submits_clean_work():
    config = {
        "novelty_mislabel": 0.0, "experiment_failure": 0.0,
        "retry_recovery": 0.0, "polish_masks_weakness": 0.0,
        "writeup_success": 1.0, "internal_review_accept": 1.0,
    }
    out = run_loop(random.Random(0), config, is_known=False)
    assert out["submitted"] is True
    assert out["clean"] is True


def test_a_polished_flaw_reaches_submission_uncleanly():
    """Ровно та категория, ради которой написан раздел про polish masking."""
    config = {
        "novelty_mislabel": 0.0, "experiment_failure": 1.0,
        "retry_recovery": 1.0, "polish_masks_weakness": 1.0,
        "writeup_success": 1.0, "internal_review_accept": 1.0,
    }
    out = run_loop(random.Random(0), config, is_known=False)
    assert out["submitted"] is True
    assert out["clean"] is False
    assert out["paper"]["masked"] is True


def test_an_unsupported_claim_never_counts_as_clean():
    config = {
        "novelty_mislabel": 0.0, "experiment_failure": 0.0,
        "retry_recovery": 0.0, "polish_masks_weakness": 0.0,
        "writeup_success": 1.0, "internal_review_accept": 1.0,
    }
    out = run_loop(random.Random(0), config, is_known=False, effect_observed=False)
    assert out["clean"] is False


def test_same_seed_reproduces_the_same_run():
    a = run_loop(random.Random(9), DEFAULT_CONFIG)
    b = run_loop(random.Random(9), DEFAULT_CONFIG)
    assert a == b


# ------------------------------------------------------------------ summarize
def test_the_two_submission_buckets_are_exhaustive():
    outs = [run_loop(random.Random(s), DEFAULT_CONFIG) for s in range(400)]
    report = summarize(outs)
    assert report["clean"] + report["flawed"] == report["submitted"]


def test_submit_rate_is_measured_against_all_trials():
    outs = [run_loop(random.Random(s), DEFAULT_CONFIG) for s in range(400)]
    report = summarize(outs)
    assert report["submit_rate"] == APPROX(report["submitted"] / report["trials"])


def test_zero_submissions_give_a_zero_share_not_a_crash():
    config = {**DEFAULT_CONFIG, "experiment_failure": 1.0, "retry_recovery": 0.0}
    outs = [run_loop(random.Random(s), config, is_known=False) for s in range(20)]
    report = summarize(outs)
    assert report["submitted"] == 0
    assert report["clean_share_of_submitted"] == APPROX(0.0)


def test_abandoned_runs_are_counted_by_the_stage_that_stopped_them():
    config = {**DEFAULT_CONFIG, "experiment_failure": 1.0, "retry_recovery": 0.0}
    outs = [run_loop(random.Random(s), config, is_known=False) for s in range(20)]
    assert summarize(outs)["abandoned_by_stage"] == {"experiment": 20}


def test_submitted_runs_do_not_appear_in_the_abandoned_breakdown():
    outs = [run_loop(random.Random(s), DEFAULT_CONFIG) for s in range(200)]
    report = summarize(outs)
    assert sum(report["abandoned_by_stage"].values()) == report["trials"] - report["submitted"]


def test_beel_style_defaults_leave_a_large_flawed_share():
    """Даже с дефолтными числами заметная часть поданного несёт изъян."""
    outs = [run_loop(random.Random(s), DEFAULT_CONFIG, is_known=False)
            for s in range(600)]
    report = summarize(outs)
    assert report["submitted"] > 0
    assert report["flawed"] / report["submitted"] > 0.2


def test_empty_trial_list_is_rejected():
    with pytest.raises(ValueError):
        summarize([])

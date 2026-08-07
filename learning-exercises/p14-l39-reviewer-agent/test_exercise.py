"""Тесты к уроку «Агент-ревьюер: строитель и приёмщик — разные роли». Правь exercise.py."""

import pytest

from exercise import (
    CALIBRATION_FLOOR,
    CONFIDENCE_FLOOR,
    DIMENSIONS,
    calibration_agreement,
    consistent_pairwise_winner,
    re_review,
    review_report,
    reviewer_view,
    score_rubric,
    verdict_from_scores,
)

ACCEPT = "pytest -q tests/test_signup.py"


def inputs(**over):
    """Безупречное закрытие задачи; любой кусок подменяется именованным аргументом."""
    base = {
        "task": {
            "required_behaviors": ["reject short password", "return 422"],
            "acceptance": [ACCEPT],
        },
        "diff": {
            "touched_files": ["app/signup.py"],
            "added_tests": ["tests/test_signup.py"],
            "behaviors_covered": ["reject short password", "return 422"],
        },
        "scope": {"allowed_files": ["app/*.py", "tests/*.py"], "declared_growth": []},
        "assumptions": [{"text": "пароль приходит в теле", "recorded_in": "docs/notes.md"}],
        "feedback": [{"command": ACCEPT, "exit_code": 0}],
        "handoff": {"next_action": "прогнать нагрузочный тест", "clean_state": []},
    }
    base.update(over)
    return base


def scores_of(**over):
    return score_rubric(inputs(**over))


# ------------------------------------------------------------ reviewer_view
def test_the_reviewer_cannot_patch_the_builders_diff():
    """Умеет править дифф — значит роли схлопнулись, и зазор исчез."""
    original = inputs()
    view = reviewer_view(original)
    view["diff"]["touched_files"].append("app/secretly_added.py")
    assert original["diff"]["touched_files"] == ["app/signup.py"]


def test_the_view_carries_every_artifact_the_reviewer_reads():
    view = reviewer_view(inputs())
    assert set(view) >= {"diff", "feedback", "handoff", "scope", "task"}


def test_the_view_is_equal_to_the_original_at_the_moment_it_is_taken():
    original = inputs()
    assert reviewer_view(original) == original


def test_editing_the_original_does_not_leak_into_an_already_taken_view():
    original = inputs()
    view = reviewer_view(original)
    original["feedback"].append({"command": "ruff", "exit_code": 1})
    assert view["feedback"] == [{"command": ACCEPT, "exit_code": 0}]


# -------------------------------------------------------------- score_rubric
def test_a_clean_close_out_scores_two_everywhere():
    scores = scores_of()
    assert [scores[n]["score"] for n in DIMENSIONS] == [2, 2, 2, 2, 2]


def test_solving_a_nearby_task_zeroes_problem_fit():
    scores = scores_of(
        diff={"touched_files": ["app/login.py"], "added_tests": [], "behaviors_covered": []}
    )
    assert scores["problem_fit"]["score"] == 0


def test_partial_coverage_scores_one_not_zero():
    diff = dict(inputs()["diff"], behaviors_covered=["return 422"])
    assert scores_of(diff=diff)["problem_fit"]["score"] == 1


def test_declared_scope_growth_costs_a_point_not_the_dimension():
    scores = scores_of(
        diff=dict(inputs()["diff"], touched_files=["app/signup.py", "docs/api.md"]),
        scope={"allowed_files": ["app/*.py", "tests/*.py"], "declared_growth": ["docs/api.md"]},
    )
    assert scores["scope_discipline"]["score"] == 1


def test_undeclared_scope_growth_zeroes_the_dimension():
    scores = scores_of(
        diff=dict(inputs()["diff"], touched_files=["app/signup.py", "scripts/release.sh"])
    )
    assert scores["scope_discipline"]["score"] == 0


def test_an_unrecorded_assumption_zeroes_the_dimension():
    assert scores_of(assumptions=[{"text": "почта уникальна", "recorded_in": None}])[
        "assumptions"
    ]["score"] == 0


def test_a_null_exit_in_the_log_zeroes_verification_quality():
    """Гейт бы это заблокировал; ревьюер обязан увидеть то же самое."""
    scores = scores_of(feedback=[{"command": ACCEPT, "exit_code": None}])
    assert scores["verification_quality"]["score"] == 0


def test_a_handoff_without_next_action_zeroes_handoff_readiness():
    scores = scores_of(handoff={"next_action": None, "clean_state": []})
    assert scores["handoff_readiness"]["score"] == 0


def test_a_dirty_workbench_costs_a_point():
    scores = scores_of(
        handoff={"next_action": "дописать тест", "clean_state": ["незакоммиченный дифф"]}
    )
    assert scores["handoff_readiness"]["score"] == 1


def test_evidence_follows_touched_files_not_the_builders_own_claim():
    """Отпечаток берётся с твёрдых улик — иначе самоотчёт строителя его сдвинет."""
    claimed = dict(inputs()["diff"], behaviors_covered=["reject short password", "return 422"])
    empty_claim = dict(claimed, behaviors_covered=[])
    assert (
        scores_of(diff=claimed)["problem_fit"]["evidence"]
        == scores_of(diff=empty_claim)["problem_fit"]["evidence"]
    )


# ------------------------------------------------------- verdict_from_scores
def flat_scores(*values):
    return {
        name: {"score": v, "confidence": 0.9, "reason": "", "evidence": ()}
        for name, v in zip(DIMENSIONS, values)
    }


def test_a_full_house_passes():
    assert verdict_from_scores(flat_scores(2, 2, 2, 2, 2)) == "pass"


def test_a_middling_run_is_a_soft_fail():
    assert verdict_from_scores(flat_scores(1, 1, 2, 1, 1)) == "soft_fail"


def test_a_zero_outranks_a_high_total():
    """Девять из десяти при нуле в problem_fit — отличная работа над не той задачей."""
    assert verdict_from_scores(flat_scores(0, 2, 2, 2, 2)) == "hard_fail"


def test_the_worst_zero_free_run_is_still_only_a_soft_fail():
    """Пять единиц дают ровно 5: сумма ниже порога hard_fail недостижима без нуля."""
    assert verdict_from_scores(flat_scores(1, 1, 1, 1, 1)) == "soft_fail"


def test_a_missing_dimension_is_refused_not_scored_as_zero():
    partial = flat_scores(2, 2, 2, 2, 2)
    partial.pop(DIMENSIONS[-1])
    with pytest.raises(ValueError):
        verdict_from_scores(partial)


def test_a_score_above_the_cap_is_refused():
    with pytest.raises(ValueError):
        verdict_from_scores(flat_scores(3, 2, 2, 2, 2))


# ------------------------------------------------------------ review_report
def test_a_clean_close_out_ships_a_pass():
    report = review_report(inputs())
    assert (report["verdict"], report["ship"], report["total"]) == ("pass", True, 10)


def test_the_report_names_the_dimensions_it_rejected():
    report = review_report(
        inputs(diff={"touched_files": ["app/login.py"], "added_tests": [], "behaviors_covered": []})
    )
    assert report["grounds"] == ["problem_fit"]
    assert report["verdict"] == "hard_fail"


def test_low_confidence_refuses_a_verdict_instead_of_guessing():
    report = review_report(inputs(feedback=[]))
    assert report["verdict"] == "needs_evidence"
    assert report["ship"] is False
    assert "verification_quality" in report["blocked_by"]


def test_the_confidence_floor_is_a_parameter_not_a_law():
    report = review_report(inputs(feedback=[]), confidence_floor=0.1)
    assert report["ship"] is True
    assert report["verdict"] != "needs_evidence"


def test_the_report_records_the_clock_it_was_given():
    assert review_report(inputs(), now=99)["generated_at"] == 99


def test_the_weakest_dimension_sets_the_reported_confidence():
    report = review_report(inputs(feedback=[]))
    assert report["min_confidence"] < CONFIDENCE_FLOOR


# ---------------------------------------------------------------- re_review
def rejected_inputs():
    """Строитель починил не ту половину: поведения не покрыты, файлы не тронуты."""
    return inputs(
        diff={"touched_files": ["app/login.py"], "added_tests": [], "behaviors_covered": []}
    )


def test_a_relabelled_diff_does_not_buy_approval():
    """Главное свойство: то же основание — тот же отказ, что бы ни дописал строитель."""
    first = review_report(rejected_inputs())
    relabelled = inputs(
        diff={
            "touched_files": ["app/login.py"],
            "added_tests": [],
            "behaviors_covered": ["reject short password", "return 422"],
        },
        resolved_claims=["problem_fit"],
    )
    second = re_review(first, relabelled)
    assert second["verdict"] == "hard_fail"
    assert second["sticky_grounds"] == ["problem_fit"]
    assert second["rejected_claims"] == ["problem_fit"]


def test_real_work_on_the_ground_clears_it():
    first = review_report(rejected_inputs())
    second = re_review(first, inputs(resolved_claims=["problem_fit"]))
    assert second["verdict"] == "pass"
    assert second["sticky_grounds"] == []


def test_a_ground_that_was_never_raised_is_not_made_sticky():
    first = review_report(inputs())
    second = re_review(first, rejected_inputs())
    assert second["sticky_grounds"] == []
    assert second["grounds"] == ["problem_fit"]


def test_the_sticky_score_is_reflected_in_the_total():
    first = review_report(rejected_inputs())
    relabelled = inputs(
        diff={
            "touched_files": ["app/login.py"],
            "added_tests": [],
            "behaviors_covered": ["reject short password", "return 422"],
        }
    )
    assert re_review(first, relabelled)["total"] == 8


def test_a_claim_without_a_sticky_ground_is_not_rejected():
    first = review_report(rejected_inputs())
    second = re_review(first, inputs(resolved_claims=["problem_fit"]))
    assert second["rejected_claims"] == []


# ----------------------------------------------- consistent_pairwise_winner
def test_a_judge_that_always_says_first_produces_no_winner():
    assert consistent_pairwise_winner(lambda x, y: "first", "A", "B") is None


def test_a_judge_consistent_across_both_orderings_produces_a_winner():
    judge = lambda x, y: "first" if x == "A" else "second"
    assert consistent_pairwise_winner(judge, "A", "B") == "A"


def test_the_winner_does_not_depend_on_the_argument_order():
    judge = lambda x, y: "first" if x == "A" else "second"
    assert consistent_pairwise_winner(judge, "B", "A") == "A"


def test_a_judge_that_always_says_second_is_also_inconsistent():
    assert consistent_pairwise_winner(lambda x, y: "second", "A", "B") is None


def test_an_answer_outside_the_two_allowed_words_is_refused():
    with pytest.raises(ValueError):
        consistent_pairwise_winner(lambda x, y: "tie", "A", "B")


# ------------------------------------------------------ calibration_agreement
def test_full_agreement_ships_the_rubric():
    cases = [
        {"id": "c1", "inputs": inputs(), "verdict": "pass"},
        {"id": "c2", "inputs": rejected_inputs(), "verdict": "hard_fail"},
    ]
    result = calibration_agreement(review_report, cases)
    assert result == {"agreement": 1.0, "ships": True, "disagreements": []}


def test_disagreements_are_named_so_the_rubric_can_be_fixed():
    cases = [
        {"id": "c1", "inputs": inputs(), "verdict": "pass"},
        {"id": "c2", "inputs": rejected_inputs(), "verdict": "pass"},
    ]
    result = calibration_agreement(review_report, cases)
    assert result["disagreements"] == ["c2"]
    assert result["agreement"] == 0.5


def test_agreement_below_the_floor_blocks_the_rollout():
    cases = [{"id": f"c{i}", "inputs": rejected_inputs(), "verdict": "pass"} for i in range(5)]
    cases[0]["verdict"] = "hard_fail"
    result = calibration_agreement(review_report, cases)
    assert result["agreement"] < CALIBRATION_FLOOR
    assert result["ships"] is False


def test_an_empty_calibration_set_is_refused_rather_than_read_as_perfect():
    with pytest.raises(ValueError):
        calibration_agreement(review_report, [])

"""Тесты к уроку «Специализация ролей». Правь exercise.py."""

import pytest

from exercise import (
    DEFAULT_REVISION_BUDGET,
    ROLE_MATRIX,
    assign_tasks,
    best_role,
    competence,
    critic_review,
    generalist,
    run_pipeline,
    team_quality,
    verifier_run,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

SPEC = {
    "name": "add_two",
    "must_contain": ["def ", "return"],
    "tests": [((1, 2), 3), ((10, 20), 30), ((-5, 5), 0)],
}

GOOD = {"code": "def add_two(a, b):\n    return a + b\n", "fn": lambda a, b: a + b}
PLAUSIBLE_BUG = {"code": "def add_two(a, b):\n    return a * b\n", "fn": lambda a, b: a * b}
UGLY = {"code": "add_two = 42\n", "fn": lambda a, b: a + b}
CRASHES = {"code": "def add_two(a, b):\n    return a / 0\n", "fn": lambda a, b: a / 0}

HETEROGENEOUS = [("spec", "plan"), ("impl", "code"), ("rev", "review"), ("qa", "test")]
HOMOGENEOUS = [("impl1", "code"), ("impl2", "code"), ("impl3", "code")]


def skills_of(tasks):
    return sorted({skill for _, skill in tasks})


def scripted_executor(*artifacts):
    """Исполнитель, выдающий заранее заданные артефакты по кругам правок."""
    calls = []

    def executor(spec, feedback):
        calls.append(list(feedback))
        return artifacts[min(len(calls) - 1, len(artifacts) - 1)]

    executor.calls = calls
    return executor


# -------------------------------------------------------------- competence
def test_competence_reads_the_matrix_cell():
    assert competence(ROLE_MATRIX, "executor", "code") == APPROX(0.9)


def test_competence_of_an_unknown_skill_is_zero():
    assert competence(ROLE_MATRIX, "executor", "dance") == APPROX(0.0)


def test_competence_of_an_unknown_role_is_an_error_not_zero():
    """Опечатку в имени роли нельзя молча превращать в нулевого исполнителя."""
    with pytest.raises(ValueError):
        competence(ROLE_MATRIX, "exectuor", "code")


def test_every_role_is_the_best_at_exactly_one_skill():
    """Специализация — это пик, а не ровный уровень."""
    for role, row in ROLE_MATRIX.items():
        top = max(row.values())
        peaks = [skill for skill, value in row.items() if value == top]
        assert len(peaks) == 1, role
        assert competence(ROLE_MATRIX, role, peaks[0]) == APPROX(top)
        assert best_role(ROLE_MATRIX, peaks[0]) == role


# --------------------------------------------------------------- best_role
def test_best_role_for_code_is_the_executor():
    assert best_role(ROLE_MATRIX, "code") == "executor"


def test_best_role_for_test_is_the_verifier():
    assert best_role(ROLE_MATRIX, "test") == "verifier"


def test_best_role_breaks_ties_alphabetically():
    tie = {"zeta": {"x": 0.5}, "alpha": {"x": 0.5}}
    assert best_role(tie, "x") == "alpha"


def test_best_role_falls_back_to_any_role_when_nobody_knows_the_skill():
    """Все нули — всё равно нужен детерминированный ответ, а не исключение."""
    assert best_role(ROLE_MATRIX, "dance") == "critic"


# ------------------------------------------------------------ assign_tasks
def test_assign_tasks_sends_each_task_to_its_specialist():
    assert assign_tasks(ROLE_MATRIX, HETEROGENEOUS) == {
        "spec": "planner",
        "impl": "executor",
        "rev": "critic",
        "qa": "verifier",
    }


def test_assign_tasks_rejects_duplicate_task_names():
    with pytest.raises(ValueError):
        assign_tasks(ROLE_MATRIX, [("impl", "code"), ("impl", "test")])


def test_assign_tasks_gives_identical_tasks_the_same_role():
    assignment = assign_tasks(ROLE_MATRIX, HOMOGENEOUS)
    assert set(assignment.values()) == {"executor"}


# ------------------------------------------------------------ team_quality
def test_team_quality_averages_the_assigned_competences():
    assert team_quality(ROLE_MATRIX, HETEROGENEOUS) == APPROX(0.9125)


def test_team_quality_rejects_an_empty_task_list():
    with pytest.raises(ValueError):
        team_quality(ROLE_MATRIX, [])


def test_specialization_beats_a_generalist_on_heterogeneous_tasks():
    flat_team = generalist(skills_of(HETEROGENEOUS), 0.9)
    assert team_quality(ROLE_MATRIX, HETEROGENEOUS) > team_quality(flat_team, HETEROGENEOUS)


def test_specialization_gains_nothing_on_homogeneous_tasks():
    """Однородные задачи — универсал того же уровня не хуже команды."""
    flat_team = generalist(skills_of(HOMOGENEOUS), 0.9)
    assert team_quality(ROLE_MATRIX, HOMOGENEOUS) == APPROX(team_quality(flat_team, HOMOGENEOUS))


def test_generalist_rejects_an_impossible_level():
    with pytest.raises(ValueError):
        generalist(["code"], 1.4)


# ----------------------------------------------------------- critic_review
def test_critic_approves_code_that_looks_right():
    assert critic_review(SPEC, GOOD) == (True, [])


def test_critic_rejects_an_artifact_missing_a_required_form():
    approved, notes = critic_review(SPEC, UGLY)
    assert approved is False
    assert notes


def test_critic_is_fooled_by_a_plausible_semantic_bug():
    """Ключевая сцена урока: код выглядит правильным, критик пропускает."""
    approved, notes = critic_review(SPEC, PLAUSIBLE_BUG)
    assert (approved, notes) == (True, [])


# ------------------------------------------------------------ verifier_run
def test_verifier_passes_correct_code():
    assert verifier_run(SPEC, GOOD) == (True, [])


def test_verifier_catches_the_bug_the_critic_missed():
    passed, failures = verifier_run(SPEC, PLAUSIBLE_BUG)
    assert passed is False
    assert len(failures) == 3


def test_verifier_survives_an_artifact_that_raises():
    """Исключение внутри артефакта — провал теста, а не крах верификатора."""
    passed, failures = verifier_run(SPEC, CRASHES)
    assert passed is False
    assert "ZeroDivisionError" in failures[0]


# ------------------------------------------------------------- run_pipeline
def test_pipeline_ships_correct_code_without_revisions():
    result = run_pipeline(SPEC, scripted_executor(GOOD))
    assert result["status"] == "shipped"
    assert result["revisions"] == 0


def test_pipeline_escalates_when_the_verifier_keeps_failing():
    result = run_pipeline(SPEC, scripted_executor(PLAUSIBLE_BUG))
    assert result["status"] == "escalated"
    assert result["revisions"] == DEFAULT_REVISION_BUDGET


def test_verification_gap_ships_the_bug_when_the_verifier_is_removed():
    """All-LLM анти-паттерн: без верификатора тот же баг уезжает в прод."""
    with_check = run_pipeline(SPEC, scripted_executor(PLAUSIBLE_BUG))
    without_check = run_pipeline(SPEC, scripted_executor(PLAUSIBLE_BUG), use_verifier=False)
    assert with_check["status"] == "escalated"
    assert without_check["status"] == "shipped"


def test_pipeline_ships_after_one_revision_round():
    result = run_pipeline(SPEC, scripted_executor(PLAUSIBLE_BUG, GOOD))
    assert result["status"] == "shipped"
    assert result["revisions"] == 1


def test_pipeline_feeds_the_failures_back_to_the_executor():
    """Исполнитель второго круга получает ровно замечания первого."""
    executor = scripted_executor(PLAUSIBLE_BUG, GOOD)
    run_pipeline(SPEC, executor)
    assert executor.calls[0] == []
    assert any("expected 3" in note for note in executor.calls[1])


def test_pipeline_runs_the_critic_before_the_verifier():
    """Критик отклонил — верификатор в этом круге не запускается."""
    executor = scripted_executor(UGLY, GOOD)
    run_pipeline(SPEC, executor)
    assert all("expected" not in note for note in executor.calls[1])


def test_pipeline_rejects_a_negative_budget():
    with pytest.raises(ValueError):
        run_pipeline(SPEC, scripted_executor(GOOD), max_revisions=-1)

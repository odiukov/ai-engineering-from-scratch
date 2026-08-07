"""Тесты к уроку «Self-Refine и CRITIC». Правь exercise.py."""

import pytest

from exercise import (
    external_verify,
    format_issues,
    loop_report,
    refine_loop,
    refine_prompt,
    scripted_generate,
    self_feedback,
    should_stop,
)

# Три версии одного черновика: полностью неверная, наполовину исправленная и
# верная. Форма у всех трёх правильная, различаются только факты — так и
# ловится разница между самокритикой и внешним верификатором.
BAD = "- Paris is the capital of Germany\n- Mt Everest is in Europe\n- Water boils at 100C"
HALF = "- Paris is the capital of France\n- Mt Everest is in Europe\n- Water boils at 100C"
GOOD = "- Paris is the capital of France\n- Mt Everest is in Asia\n- Water boils at 100C"

WRONG_FACTS = (("paris", "germany"), ("everest", "europe"))

# Заглушка модели реагирует на содержимое промпта: пока в истории нет второй
# попытки, она чинит только столицу.
SCRIPT = (
    ("attempt 2", GOOD),
    ("germany", HALF),
    ("", BAD),
)

# Скрипт, который каждый раз выдаёт новый, но всё равно кривой по форме
# черновик: залипания нет, петля упирается в бюджет.
BUDGET_SCRIPT = (
    ("attempt 3", "- x1\n- x2"),
    ("attempt 2", "- y1"),
    ("attempt 1", "- z1\n- z2\n- z3\n- z4"),
    ("", "- w1\n- w2\n- w3\n- w4\n- w5"),
)

# Скрипт без прогресса: модель раз за разом выдаёт один и тот же черновик.
STALL_SCRIPT = (("", BAD),)


def critic(output):
    """CRITIC-верификатор с заземлением на WRONG_FACTS."""
    return external_verify(output, WRONG_FACTS)


# ----------------------------------------------------------- format_issues
def test_format_issues_empty_for_wellformed_draft():
    assert format_issues(GOOD) == []


def test_format_issues_counts_bullets():
    assert format_issues("- a\n- b") == ["expected 3 bullets, got 2"]


def test_format_issues_flags_too_long_bullet():
    long_bullet = "- " + "x" * 70
    issues = format_issues(long_bullet + "\n- b\n- c")
    assert issues == ["bullet too long: 72 > 60"]


def test_format_issues_ignores_trailing_blank_lines():
    """Хвостовой перевод строки не должен превращаться в четвёртый пункт."""
    assert format_issues("- a\n- b\n- c\n\n") == []


def test_format_issues_flags_non_bullet_line():
    issues = format_issues("Here you go:\n- a\n- b\n- c")
    assert "non-bullet line present" in issues


# ----------------------------------------------------------- self_feedback
def test_self_feedback_accepts_wellformed_draft():
    assert self_feedback(GOOD) == ("no issues", True)


def test_self_feedback_reports_format_problem():
    critique, ok = self_feedback("- a\n- b")
    assert ok is False
    assert "expected 3 bullets" in critique


def test_self_feedback_is_blind_to_hallucinations():
    """Самокритик принимает уверенно звучащую чушь — ради этого и есть CRITIC."""
    assert self_feedback(BAD) == ("no issues", True)


# --------------------------------------------------------- external_verify
def test_external_verify_accepts_grounded_draft():
    assert external_verify(GOOD, WRONG_FACTS) == ("verifier: ok", True)


def test_external_verify_rejects_what_self_feedback_accepted():
    """Тот же черновик: самокритик доволен, внешний верификатор — нет."""
    assert self_feedback(BAD)[1] is True
    assert external_verify(BAD, WRONG_FACTS)[1] is False


def test_external_verify_needs_every_part_of_a_fact():
    """Одно слово «paris» ошибкой не является — нужен весь кортеж целиком."""
    assert external_verify(HALF, [("paris", "germany")]) == ("verifier: ok", True)


def test_external_verify_matches_facts_case_insensitively():
    critique, ok = external_verify(BAD, [("PARIS", "GERMANY")])
    assert ok is False
    assert "PARIS + GERMANY" in critique


def test_external_verify_without_tools_degenerates_to_self_feedback():
    """Нет внешних фактов — CRITIC становится обычным Self-Refine."""
    for draft in (BAD, GOOD, "- a\n- b"):
        assert external_verify(draft, [])[1] == self_feedback(draft)[1]


# ------------------------------------------------------------ refine_prompt
def test_refine_prompt_without_history_asks_for_first_draft():
    assert refine_prompt("bullets", []) == "TASK: bullets\nWrite the first draft."


def test_refine_prompt_carries_every_prior_output():
    """Абляция из статьи: выкинь историю — и модель повторит свою ошибку."""
    history = [
        {"iteration": 1, "output": BAD, "critique": "c1"},
        {"iteration": 2, "output": HALF, "critique": "c2"},
    ]
    prompt = refine_prompt("world facts", history)
    assert BAD in prompt
    assert HALF in prompt


def test_refine_prompt_carries_every_prior_critique():
    history = [
        {"iteration": 1, "output": BAD, "critique": "capital is wrong"},
        {"iteration": 2, "output": HALF, "critique": "continent is wrong"},
    ]
    prompt = refine_prompt("world facts", history)
    assert "CRITIQUE 1: capital is wrong" in prompt
    assert "CRITIQUE 2: continent is wrong" in prompt


def test_refine_prompt_keeps_the_original_task_at_every_depth():
    """Задача не должна теряться под грудой истории."""
    history = [{"iteration": i, "output": f"o{i}", "critique": f"c{i}"} for i in range(1, 6)]
    assert refine_prompt("summarize the docs", history).startswith("TASK: summarize the docs")


# -------------------------------------------------------- scripted_generate
def test_scripted_generate_picks_first_matching_keyword():
    script = (("germany", "fixed"), ("", "first draft"))
    assert scripted_generate("CRITIQUE 1: germany is wrong", script) == "fixed"


def test_scripted_generate_falls_back_to_the_empty_keyword():
    script = (("germany", "fixed"), ("", "first draft"))
    assert scripted_generate("TASK: facts", script) == "first draft"


def test_scripted_generate_is_a_pure_function_of_the_prompt():
    """Одинаковый промпт — одинаковый черновик, иначе тест петли ничего не значит."""
    script = (("germany", "fixed"), ("", "first draft"))
    assert scripted_generate("germany", script) == scripted_generate("germany", script)
    assert scripted_generate("germany", script) != scripted_generate("nothing", script)


def test_scripted_generate_rejects_empty_script():
    with pytest.raises(ValueError):
        scripted_generate("TASK: facts", ())


# --------------------------------------------------------------- should_stop
def test_should_stop_when_verifier_passes():
    assert should_stop(1, True, False, 4) == (True, "verified")


def test_should_stop_when_model_repeats_a_rejected_draft():
    assert should_stop(2, False, True, 4) == (True, "stalled")


def test_should_stop_when_budget_is_exhausted():
    assert should_stop(4, False, False, 4) == (True, "budget")


def test_should_continue_while_budget_is_left():
    assert should_stop(1, False, False, 4) == (False, "continue")


def test_verified_wins_over_exhausted_budget():
    """Удачная последняя итерация — это 'verified', а не 'budget'."""
    assert should_stop(4, True, False, 4) == (True, "verified")


# --------------------------------------------------------------- refine_loop
def test_refine_loop_converges_when_verifier_passes():
    history = refine_loop("world facts", SCRIPT, critic, max_iterations=4)
    assert len(history) == 3
    assert history[-1]["output"] == GOOD
    assert history[-1]["stop_reason"] == "verified"


def test_self_refine_settles_on_a_hallucination_that_critic_catches():
    """Головной результат урока: заземление меняет исход, а не число итераций."""
    self_run = refine_loop("world facts", SCRIPT, self_feedback, max_iterations=4)
    critic_run = refine_loop("world facts", SCRIPT, critic, max_iterations=4)
    assert self_run[-1]["output"] == BAD
    assert critic_run[-1]["output"] == GOOD


def test_refine_loop_stops_when_the_model_repeats_a_rejected_draft():
    """Критик не принимает уже отклонённый вариант, и петля обязана закончиться."""
    history = refine_loop("world facts", STALL_SCRIPT, critic, max_iterations=99)
    assert len(history) == 2
    assert history[-1]["stop_reason"] == "stalled"
    assert history[-1]["verified"] is False


def test_refine_loop_respects_the_iteration_budget():
    history = refine_loop("bullets", BUDGET_SCRIPT, self_feedback, max_iterations=4)
    assert len(history) == 4
    assert history[-1]["stop_reason"] == "budget"


def test_refine_loop_marks_only_the_last_attempt_as_stopping():
    history = refine_loop("bullets", BUDGET_SCRIPT, self_feedback, max_iterations=4)
    assert [a["stop_reason"] for a in history] == ["continue", "continue", "continue", "budget"]


def test_refine_loop_with_one_iteration_never_refines():
    """max_iterations=1 — это просто generate плюс проверка."""
    history = refine_loop("world facts", SCRIPT, critic, max_iterations=1)
    assert len(history) == 1
    assert history[0]["output"] == BAD
    assert history[0]["stop_reason"] == "budget"


# --------------------------------------------------------------- loop_report
def test_loop_report_marks_a_converged_run():
    report = loop_report(refine_loop("world facts", SCRIPT, critic, max_iterations=4))
    assert report["converged"] is True
    assert report["reason"] == "verified"
    assert report["iterations"] == 3
    assert report["final_output"] == GOOD


def test_loop_report_does_not_call_an_exhausted_budget_a_success():
    report = loop_report(refine_loop("bullets", BUDGET_SCRIPT, self_feedback, max_iterations=4))
    assert report["converged"] is False
    assert report["reason"] == "budget"
    assert len(report["critiques"]) == 4


def test_loop_report_rejects_empty_history():
    with pytest.raises(ValueError):
        loop_report([])

"""Тесты к уроку «Workflow-паттерны Anthropic». Правь exercise.py."""

import pytest

from exercise import (
    PATTERNS,
    evaluator_optimizer,
    orchestrator_workers,
    parallel_sections,
    parallel_vote,
    pick_pattern,
    prompt_chain,
    route,
)


def echo_llm():
    """«Модель», которая приписывает к промпту метку. Возвращает (llm, журнал)."""
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return f"<{prompt}>"

    return llm, calls


def sequence_llm(answers):
    """«Модель», выдающая заготовленные ответы по порядку."""
    calls = []
    box = {"i": 0}

    def llm(prompt):
        calls.append(prompt)
        answer = answers[min(box["i"], len(answers) - 1)]
        box["i"] += 1
        return answer

    return llm, calls


def worker(name, handles, output):
    """Воркер оркестратора: имя, условие, готовый ответ."""
    return {"name": name, "handles": handles, "run": lambda task: output}


# -------------------------------------------------------------- prompt_chain
def test_chain_feeds_each_output_into_the_next_step():
    llm, calls = echo_llm()
    final, _ = prompt_chain("raw", llm, (("a", "A:{text}"), ("b", "B:{text}")))
    assert calls == ["A:raw", "B:<A:raw>"]
    assert final == "<B:<A:raw>>"


def test_chain_without_steps_returns_the_input_unchanged():
    llm, calls = echo_llm()
    assert prompt_chain("raw", llm, ()) == ("raw", [])
    assert calls == []


def test_chain_records_every_step_in_the_trace():
    llm, _ = echo_llm()
    _, trace = prompt_chain("raw", llm, (("a", "A:{text}"), ("b", "B:{text}")))
    assert [label for label, _ in trace] == ["a", "b"]


def test_chain_stops_calling_the_model_after_a_failed_gate():
    """Ради этой программной проверки цепочку и разбивают на шаги."""
    llm, calls = echo_llm()
    gate = lambda label, output: (label != "a", "step a produced junk")
    prompt_chain("raw", llm, (("a", "A:{text}"), ("b", "B:{text}")), gate=gate)
    assert calls == ["A:raw"]


def test_chain_returns_the_last_validated_value_not_the_rejected_one():
    """Отдать наружу забракованное — значит поставить gate и игнорировать его."""
    llm, _ = echo_llm()
    gate = lambda label, output: (label != "a", "step a produced junk")
    final, trace = prompt_chain("raw", llm, (("a", "A:{text}"),), gate=gate)
    assert final == "raw"
    assert trace[-1] == ("a:gate", "step a produced junk")


# --------------------------------------------------------------------- route
def test_route_dispatches_to_the_matching_handler():
    classifier = lambda text: ("refund", 0.95)
    handlers = {"refund": lambda t: "refund filed", "bug": lambda t: "bug logged"}
    assert route("I want my money back", classifier, handlers) == \
        ("refund", "refund filed")


def test_route_falls_back_to_the_default_handler():
    classifier = lambda text: ("astrology", 0.9)
    handlers = {"refund": lambda t: "refund filed", "default": lambda t: "generic"}
    assert route("what is my sign", classifier, handlers) == ("astrology", "generic")


def test_route_refuses_an_unknown_label_without_a_default():
    """Тихий пустой ответ позволил бы классификатору выдумывать категории."""
    with pytest.raises(KeyError):
        route("x", lambda t: ("astrology", 0.9), {"refund": lambda t: "filed"})


def test_route_below_the_threshold_never_calls_the_specialist():
    """Дешёвый неверный ответ в поддержке дороже эскалации."""
    called = []
    handlers = {
        "refund": lambda t: called.append("refund") or "refund filed",
        "escalate": lambda t: "to human",
    }
    label, out = route("maybe a refund?", lambda t: ("refund", 0.4), handlers,
                       threshold=0.9)
    assert (label, out) == ("escalate", "to human")
    assert called == []


def test_route_with_the_default_threshold_never_escalates():
    handlers = {"refund": lambda t: "refund filed", "escalate": lambda t: "to human"}
    assert route("x", lambda t: ("refund", 0.0), handlers)[0] == "refund"


# ------------------------------------------------------------- parallel_vote
def test_vote_returns_the_majority_answer():
    llm, _ = sequence_llm(["yes", "yes", "no", "yes", "no"])
    winner, counts = parallel_vote("safe to ship?", llm, n=5)
    assert winner == "yes"
    assert counts == {"yes": 3, "no": 2}


def test_vote_calls_the_model_exactly_n_times():
    llm, calls = sequence_llm(["yes"])
    parallel_vote("safe to ship?", llm, n=4)
    assert len(calls) == 4


def test_vote_breaks_a_tie_in_favour_of_the_earlier_answer():
    """Без правила один и тот же набор голосов давал бы разный вердикт."""
    llm, _ = sequence_llm(["no", "yes", "yes", "no"])
    assert parallel_vote("safe to ship?", llm, n=4)[0] == "no"


def test_vote_of_a_single_run_returns_that_run():
    llm, _ = sequence_llm(["maybe"])
    assert parallel_vote("safe to ship?", llm, n=1) == ("maybe", {"maybe": 1})


def test_vote_refuses_to_run_with_no_votes():
    llm, _ = sequence_llm(["yes"])
    with pytest.raises(ValueError):
        parallel_vote("safe to ship?", llm, n=0)


# --------------------------------------------------------- parallel_sections
def test_sections_call_the_model_once_per_section():
    llm, calls = echo_llm()
    parallel_sections((("a", "check a"), ("b", "check b")), llm, dict)
    assert calls == ["check a", "check b"]


def test_sections_keep_the_declared_order():
    """aggregate часто склеивает секции в документ — перестановка там заметна."""
    llm, _ = echo_llm()
    _, outputs = parallel_sections((("intro", "p1"), ("body", "p2")), llm, dict)
    assert [name for name, _ in outputs] == ["intro", "body"]


def test_sections_pass_everything_to_the_aggregator():
    llm, _ = echo_llm()
    joined, _ = parallel_sections(
        (("a", "p1"), ("b", "p2")), llm,
        lambda pairs: " | ".join(f"{n}={v}" for n, v in pairs))
    assert joined == "a=<p1> | b=<p2>"


def test_sections_with_nothing_to_do_still_call_the_aggregator():
    llm, calls = echo_llm()
    assert parallel_sections((), llm, dict) == ({}, [])
    assert calls == []


# ------------------------------------------------------ orchestrator_workers
def test_orchestrator_runs_only_the_workers_that_handle_the_task():
    ran = []
    workers = (
        {"name": "python", "handles": lambda t: "python" in t,
         "run": lambda t: ran.append("python") or "python ok"},
        {"name": "sql", "handles": lambda t: "sql" in t,
         "run": lambda t: ran.append("sql") or "sql ok"},
    )
    _, outputs = orchestrator_workers("review this python change", workers, list)
    assert ran == ["python"]
    assert outputs == [("python", "python ok")]


def test_orchestrator_keeps_the_declared_worker_order():
    workers = (worker("security", lambda t: True, "sec ok"),
               worker("style", lambda t: True, "style ok"))
    _, outputs = orchestrator_workers("review", workers, list)
    assert [name for name, _ in outputs] == ["security", "style"]


def test_orchestrator_hands_the_outputs_to_the_synthesizer():
    workers = (worker("a", lambda t: True, "one"), worker("b", lambda t: True, "two"))
    final, _ = orchestrator_workers(
        "review", workers, lambda pairs: " | ".join(v for _, v in pairs))
    assert final == "one | two"


def test_orchestrator_with_no_matching_worker_does_not_crash():
    """Иначе каждый новый тип задачи ронял бы роутер."""
    workers = (worker("a", lambda t: False, "one"),)
    assert orchestrator_workers("unknown task", workers, list) == ([], [])


# ------------------------------------------------------- evaluator_optimizer
def test_evaluator_optimizer_stops_on_the_first_pass():
    proposals = []

    def propose(task, feedback):
        proposals.append(feedback)
        return "draft"

    final, trace = evaluator_optimizer("t", propose, lambda t, c: (True, "PASS"))
    assert (final, len(trace)) == ("draft", 1)
    assert proposals == [None]


def test_evaluator_optimizer_feeds_the_verdict_back_into_the_next_proposal():
    """Без этого это не рефайнмент, а повтор одного промпта в надежде на чудо."""
    seen = []

    def propose(task, feedback):
        seen.append(feedback)
        return f"draft{len(seen)}"

    def evaluate(task, candidate):
        return candidate == "draft2", f"FAIL: fix {candidate}"

    evaluator_optimizer("t", propose, evaluate)
    assert seen == [None, "FAIL: fix draft1"]


def test_evaluator_optimizer_gives_up_after_max_iter():
    """Бесконечный цикл — самый дорогой способ узнать, что оценщик строг."""
    final, trace = evaluator_optimizer(
        "t", lambda task, fb: "draft", lambda t, c: (False, "FAIL"), max_iter=3)
    assert (final, len(trace)) == ("draft", 3)


def test_evaluator_optimizer_records_the_verdict_of_every_iteration():
    _, trace = evaluator_optimizer(
        "t", lambda task, fb: "draft", lambda t, c: (False, "FAIL: too long"),
        max_iter=2)
    assert trace == [("draft", False, "FAIL: too long"),
                     ("draft", False, "FAIL: too long")]


def test_evaluator_optimizer_refuses_a_budget_of_zero():
    with pytest.raises(ValueError):
        evaluator_optimizer("t", lambda task, fb: "draft",
                            lambda t, c: (True, "PASS"), max_iter=0)


# -------------------------------------------------------------- pick_pattern
def test_unpredictable_steps_always_mean_an_agent():
    """Неперечислимые шаги нельзя выразить предопределённым графом."""
    spec = {"steps_known": False, "categories": 5, "has_evaluator": True,
            "workers_chosen_at_runtime": True, "parallel_units": 9}
    assert pick_pattern(spec)[0] == "agent"


def test_runtime_worker_choice_beats_a_fixed_category_split():
    spec = {"steps_known": True, "workers_chosen_at_runtime": True, "categories": 4}
    assert pick_pattern(spec)[0] == "orchestrator-workers"


def test_a_machine_checkable_answer_means_evaluator_optimizer():
    spec = {"steps_known": True, "has_evaluator": True, "categories": 4}
    assert pick_pattern(spec)[0] == "evaluator-optimizer"


def test_distinct_input_categories_mean_routing():
    assert pick_pattern({"steps_known": True, "categories": 3,
                         "parallel_units": 5})[0] == "routing"


def test_independent_units_of_one_category_mean_parallelization():
    assert pick_pattern({"steps_known": True, "categories": 1,
                         "parallel_units": 5})[0] == "parallelization"


def test_the_default_answer_is_the_cheapest_pattern():
    """По умолчанию выбирают линейную цепочку, а не фреймворк."""
    pattern, reason = pick_pattern({"steps_known": True})
    assert pattern == "prompt-chaining"
    assert reason


def test_pick_pattern_only_ever_names_a_known_pattern():
    specs = ({"steps_known": False},
             {"steps_known": True},
             {"steps_known": True, "categories": 2},
             {"steps_known": True, "parallel_units": 2},
             {"steps_known": True, "has_evaluator": True},
             {"steps_known": True, "workers_chosen_at_runtime": True})
    assert all(pick_pattern(s)[0] in PATTERNS for s in specs)


def test_pick_pattern_refuses_to_guess_whether_the_steps_are_known():
    """Угадав этот ответ, функция вернёт красивый паттерн для неподходящей задачи."""
    with pytest.raises(KeyError):
        pick_pattern({"categories": 3})

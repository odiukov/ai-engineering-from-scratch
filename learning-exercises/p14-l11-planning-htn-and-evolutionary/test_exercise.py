"""Тесты к уроку «Планирование: HTN и эволюционный поиск». Правь exercise.py."""

import random

import pytest

from exercise import (
    applicable,
    apply_operator,
    decompose,
    evolve,
    execute_plan,
    fitness_linear,
    mutate,
    plan,
)


def domain():
    """Домен «выкатить изменение»: четыре оператора и один метод."""
    return {
        "operators": {
            "open_editor": {"pre": ("logged_in",), "add": ("editor_open",),
                            "remove": ()},
            "write_tests": {"pre": ("editor_open",), "add": ("tests_written",),
                            "remove": ()},
            "run_tests": {"pre": ("tests_written",), "add": ("tests_passing",),
                          "remove": ()},
            "open_pr": {"pre": ("tests_passing",), "add": ("pr_open",),
                        "remove": ()},
        },
        "methods": {
            "ship_change": (
                {"name": "m1", "pre": ("logged_in",),
                 "subtasks": ("open_editor", "write_tests", "run_tests", "open_pr")},
            ),
        },
    }


def scripted_llm(script):
    """«LLM» из словаря задача -> подзадачи. Возвращает (функция, журнал вызовов)."""
    calls = []

    def llm(task, state):
        calls.append(task)
        return script.get(task)

    return llm, calls


SAMPLES = tuple((x, 3 * x + 7) for x in range(-5, 6))


def sse(individual):
    """Fitness для прямой 3x + 7 на фиксированной выборке."""
    return fitness_linear(individual, SAMPLES)


# ---------------------------------------------------------------- applicable
def test_applicable_when_every_precondition_holds():
    op = {"pre": ("logged_in",), "add": ("editor_open",), "remove": ()}
    assert applicable(op, {"logged_in"}) is True


def test_applicable_is_false_when_a_precondition_is_missing():
    op = {"pre": ("logged_in", "on_branch"), "add": (), "remove": ()}
    assert applicable(op, {"logged_in"}) is False


def test_an_operator_without_preconditions_is_always_applicable():
    """Пустое «все» истинно — так выглядит первый шаг любого плана."""
    assert applicable({"pre": (), "add": ("x",), "remove": ()}, set()) is True


# ------------------------------------------------------------ apply_operator
def test_apply_operator_adds_its_effect():
    op = {"pre": ("logged_in",), "add": ("editor_open",), "remove": ()}
    assert apply_operator(op, {"logged_in"}) == frozenset({"logged_in", "editor_open"})


def test_apply_operator_removes_before_it_adds():
    """Оператор, обновляющий факт на месте, при обратном порядке потерял бы его."""
    op = {"pre": (), "add": ("city_lisbon",), "remove": ("city_lisbon", "city_berlin")}
    assert apply_operator(op, {"city_berlin"}) == frozenset({"city_lisbon"})


def test_apply_operator_refuses_when_preconditions_are_unmet():
    """На этом отказе держится заявление HTN о корректности плана."""
    op = {"pre": ("logged_in",), "add": ("editor_open",), "remove": ()}
    with pytest.raises(ValueError):
        apply_operator(op, set())


def test_apply_operator_leaves_the_input_state_alone():
    state = {"logged_in"}
    apply_operator({"pre": (), "add": ("editor_open",), "remove": ()}, state)
    assert state == {"logged_in"}


# ----------------------------------------------------------------- decompose
def test_decompose_returns_the_subtasks_of_an_applicable_method():
    assert decompose(domain()["methods"], "ship_change", {"logged_in"}) == (
        "open_editor", "write_tests", "run_tests", "open_pr")


def test_decompose_refuses_a_task_whose_preconditions_are_unmet():
    """Метод описывает, как делать задачу в подходящей обстановке, а не всегда."""
    assert decompose(domain()["methods"], "ship_change", set()) is None


def test_decompose_of_an_unknown_task_is_none():
    assert decompose(domain()["methods"], "deploy_to_mars", {"logged_in"}) is None


def test_decompose_takes_the_first_declared_applicable_method():
    """Порядок объявления методов — это приоритет."""
    methods = {
        "t": (
            {"name": "specific", "pre": ("fast_path",), "subtasks": ("a",)},
            {"name": "general", "pre": (), "subtasks": ("b",)},
        )
    }
    assert decompose(methods, "t", {"fast_path"}) == ("a",)
    assert decompose(methods, "t", set()) == ("b",)


# ---------------------------------------------------------------------- plan
def test_plan_of_a_primitive_task_is_a_single_step():
    assert plan(domain(), "open_editor", {"logged_in"}) == ["open_editor"]


def test_plan_expands_a_compound_task_into_primitives():
    assert plan(domain(), "ship_change", {"logged_in"}) == [
        "open_editor", "write_tests", "run_tests", "open_pr"]


def test_plan_refuses_a_task_whose_preconditions_are_unmet():
    """HTN не раскрывает задачу в неподходящем состоянии — плана просто нет."""
    assert plan(domain(), "ship_change", set()) is None


def test_plan_advances_the_state_between_subtasks():
    """write_tests требует editor_open, который появляется только после первого шага."""
    dom = domain()
    assert plan(dom, "write_tests", {"logged_in"}) is None
    assert plan(dom, "ship_change", {"logged_in"})[1] == "write_tests"


def test_plan_asks_the_llm_only_when_no_method_matches():
    llm, calls = scripted_llm({})
    plan(domain(), "ship_change", {"logged_in"}, llm=llm)
    assert calls == []


def test_plan_accepts_a_valid_llm_decomposition():
    llm, calls = scripted_llm({
        "ship_with_migration": ("open_editor", "write_tests", "run_tests"),
    })
    assert plan(domain(), "ship_with_migration", {"logged_in"}, llm=llm) == [
        "open_editor", "write_tests", "run_tests"]
    assert calls == ["ship_with_migration"]


def test_plan_rejects_an_llm_step_that_is_not_in_the_schema():
    """Модель предлагает кандидата; схема — единственный фильтр от галлюцинаций."""
    llm, _ = scripted_llm({"ship_with_migration": ("open_editor", "fly_to_mars")})
    cache = {}
    assert plan(domain(), "ship_with_migration", {"logged_in"},
                llm=llm, cache=cache) is None
    assert cache == {}


def test_plan_reuses_the_cache_instead_of_calling_the_llm_twice():
    """Online method learning: за ту же декомпозицию второй раз не платим."""
    llm, calls = scripted_llm({"ship_with_migration": ("open_editor", "write_tests")})
    dom, cache = domain(), {}
    plan(dom, "ship_with_migration", {"logged_in"}, llm=llm, cache=cache)
    plan(dom, "ship_with_migration", {"logged_in"}, llm=llm, cache=cache)
    assert calls == ["ship_with_migration"]
    assert cache == {"ship_with_migration": ("open_editor", "write_tests")}


# -------------------------------------------------------------- execute_plan
def test_execute_plan_applies_every_step_in_order():
    assert execute_plan(domain(), ["open_editor", "write_tests"], {"logged_in"}) == \
        frozenset({"logged_in", "editor_open", "tests_written"})


def test_execute_plan_of_an_empty_plan_changes_nothing():
    assert execute_plan(domain(), [], {"logged_in"}) == frozenset({"logged_in"})


def test_execute_plan_refuses_a_step_the_environment_no_longer_supports():
    """План и исполнение разнесены во времени — проверка нужна и здесь."""
    with pytest.raises(ValueError):
        execute_plan(domain(), ["run_tests"], {"logged_in"})


def test_execute_plan_refuses_a_step_that_is_not_in_the_schema():
    with pytest.raises(KeyError):
        execute_plan(domain(), ["fly_to_mars"], {"logged_in"})


# ------------------------------------------------------------ fitness_linear
def test_fitness_of_a_perfect_fit_is_zero():
    assert fitness_linear((3, 7), ((0, 7), (1, 10))) == pytest.approx(0.0)


def test_fitness_sums_squared_errors():
    assert fitness_linear((0, 0), ((0, 7), (1, 10))) == pytest.approx(149.0)


def test_fitness_grows_as_the_line_moves_away():
    assert sse((3, 7)) < sse((3, 8)) < sse((3, 12))


# -------------------------------------------------------------------- mutate
def test_mutate_with_zero_step_is_the_identity():
    """Вырожденный случай отделяет вклад мутаций от вклада отбора."""
    assert mutate((3, 7), random.Random(0), step=0) == (3, 7)


def test_mutate_is_reproducible_for_the_same_seed():
    assert mutate((3, 7), random.Random(42)) == mutate((3, 7), random.Random(42))


def test_mutate_stays_within_the_step_around_the_parent():
    child = mutate((3, 7), random.Random(1), step=2)
    assert abs(child[0] - 3) <= 2 and abs(child[1] - 7) <= 2


# -------------------------------------------------------------------- evolve
def test_evolve_history_has_one_entry_per_generation_plus_the_start():
    _, history = evolve([(0, 0)], sse, mutate, random.Random(0), generations=5)
    assert len(history) == 6


def test_evolve_never_lets_the_best_get_worse():
    """Элитизм: родители переходят дальше, поэтому лучшее не может ухудшиться."""
    _, history = evolve([(0, 0), (-8, 4)], sse, mutate, random.Random(0),
                        generations=25)
    assert all(later <= earlier for earlier, later in zip(history, history[1:]))


def test_evolve_is_reproducible_for_the_same_seed():
    first = evolve([(0, 0)], sse, mutate, random.Random(7), generations=20)
    second = evolve([(0, 0)], sse, mutate, random.Random(7), generations=20)
    assert first == second


def test_evolve_converges_on_the_target_line():
    best, history = evolve([(0, 0)], sse, mutate, random.Random(0), generations=40)
    assert best == (3, 7)
    assert history[-1] == pytest.approx(0.0)


def test_evolve_without_real_mutations_stands_still():
    """Отбор без мутаций ничего не находит — история плоская."""
    frozen = lambda ind, rng: mutate(ind, rng, step=0)
    best, history = evolve([(0, 0)], sse, frozen, random.Random(0), generations=10)
    assert best == (0, 0)
    assert len(set(history)) == 1

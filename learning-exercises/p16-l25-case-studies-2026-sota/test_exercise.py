"""Тесты к уроку «Разборы кейсов и состояние дел 2026». Правь exercise.py."""

import pytest

from exercise import (
    SUBAGENTS,
    CyclicRouting,
    critical_path,
    linear_fit,
    r_squared,
    relative_improvement,
    retirable_versions,
    subagent_budget,
    topological_order,
    verification_budget,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Прогоны BrowseComp: тысячи токенов и балл. Подобрано так, что объём
# токенов объясняет ровно 80% разброса — цифра из инженерного разбора
# системы Research у Anthropic.
TOKENS = [10, 10, 30, 30]
SCORES = [4, 2, 8, 6]

# Роли MetaGPT как DAG передач: PM -> архитектор -> инженер -> QA,
# и параллельная ветка на техписателя.
SOP = {
    "pm": ["architect", "writer"],
    "architect": ["engineer"],
    "engineer": ["qa"],
    "qa": [],
    "writer": [],
}


def sse(xs, ys, slope, intercept):
    """Сумма квадратов остатков. Своя реализация, из exercise ничего не берёт."""
    return sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))


# -------------------------------------------------------------- linear_fit
def test_linear_fit_recovers_an_exact_line():
    assert linear_fit([0, 1, 2], [1, 3, 5]) == (APPROX(2.0), APPROX(1.0))


def test_linear_fit_of_a_flat_cloud_has_zero_slope():
    assert linear_fit([0, 1], [5, 5]) == (APPROX(0.0), APPROX(5.0))


def test_linear_fit_sits_at_a_zero_gradient_of_the_squared_error():
    """Аналитический оптимум обязан совпасть с численным: обе производные ~0.

    Считаем центральную разность SSE по наклону и по свободному члену
    в найденной точке. Если формула наименьших квадратов реализована
    правильно, обе близки к нулю.
    """
    slope, intercept = linear_fit(TOKENS, SCORES)
    h = 1e-5
    d_slope = (sse(TOKENS, SCORES, slope + h, intercept)
               - sse(TOKENS, SCORES, slope - h, intercept)) / (2 * h)
    d_intercept = (sse(TOKENS, SCORES, slope, intercept + h)
                   - sse(TOKENS, SCORES, slope, intercept - h)) / (2 * h)
    assert d_slope == pytest.approx(0.0, abs=1e-6)
    assert d_intercept == pytest.approx(0.0, abs=1e-6)


def test_linear_fit_beats_any_nearby_line():
    """Наименьшие квадраты — значит меньше не бывает."""
    slope, intercept = linear_fit(TOKENS, SCORES)
    best = sse(TOKENS, SCORES, slope, intercept)
    for ds, di in ((0.1, 0.0), (-0.1, 0.0), (0.0, 0.5), (0.0, -0.5)):
        assert sse(TOKENS, SCORES, slope + ds, intercept + di) > best


def test_linear_fit_rejects_a_vertical_cloud():
    """По одинаковым x прямую не провести — это ValueError, а не нулевой наклон."""
    with pytest.raises(ValueError):
        linear_fit([3, 3, 3], [1, 2, 3])


def test_linear_fit_needs_at_least_two_points():
    with pytest.raises(ValueError):
        linear_fit([1], [1])


# --------------------------------------------------------------- r_squared
def test_r_squared_of_points_on_a_line_is_one():
    assert r_squared([0, 1, 2], [1, 3, 5]) == APPROX(1.0)


def test_token_usage_explains_eighty_percent_of_the_variance():
    """Та самая цифра из разбора Research: 80% разброса — это объём токенов."""
    assert r_squared(TOKENS, SCORES) == APPROX(0.8)


def test_r_squared_is_low_when_the_line_explains_little():
    assert r_squared([0, 1, 2, 3], [0, 3, 1, 2]) == APPROX(0.16)


def test_r_squared_ignores_a_shift_of_the_predictor():
    """Объяснённая доля не зависит от того, в каких единицах меряют токены."""
    shifted = [t * 1000 + 7 for t in TOKENS]
    assert r_squared(shifted, SCORES) == APPROX(r_squared(TOKENS, SCORES))


def test_r_squared_needs_variance_in_the_outcome():
    with pytest.raises(ValueError):
        r_squared([1, 2, 3], [5, 5, 5])


# ------------------------------------------------------ relative_improvement
def test_relative_improvement_reads_as_a_ratio_to_the_baseline():
    assert relative_improvement(1.0, 1.902) == pytest.approx(0.902, abs=1e-9)


def test_relative_improvement_is_negative_on_a_regression():
    assert relative_improvement(0.5, 0.25) == APPROX(-0.5)


def test_relative_improvement_is_not_a_difference_in_percentage_points():
    """0.30 -> 0.57 это +90%, а не +27. Путаница удваивает заявленный результат."""
    assert relative_improvement(0.30, 0.57) == pytest.approx(0.9, abs=1e-9)


def test_relative_improvement_rejects_a_zero_baseline():
    with pytest.raises(ValueError):
        relative_improvement(0.0, 0.3)


# ----------------------------------------------------- verification_budget
def test_verification_budget_splits_the_token_pool():
    assert verification_budget(100000, 0.25) == (APPROX(75000.0), APPROX(25000.0))


def test_verification_budget_conserves_the_total():
    work, check = verification_budget(123456, 0.3)
    assert work + check == APPROX(123456)


def test_zero_tax_leaves_the_whole_budget_for_work():
    assert verification_budget(100000, 0.0) == (APPROX(100000.0), APPROX(0.0))


def test_verification_budget_rejects_a_full_tax():
    """tax=1 — система, которая только проверяет и ничего не делает."""
    with pytest.raises(ValueError):
        verification_budget(100000, 1.0)


def test_verification_budget_rejects_a_negative_tax():
    with pytest.raises(ValueError):
        verification_budget(100000, -0.1)


# --------------------------------------------------------- subagent_budget
def test_subagent_budget_scales_with_complexity():
    assert subagent_budget("simple") == 1
    assert subagent_budget("medium") == 3
    assert subagent_budget("complex") == 10


def test_subagent_budget_is_monotone_in_complexity():
    assert subagent_budget("simple") < subagent_budget("medium") < subagent_budget("complex")


def test_subagent_budget_matches_the_published_table():
    assert {k: subagent_budget(k) for k in SUBAGENTS} == SUBAGENTS


def test_unknown_complexity_is_an_error_not_a_silent_default():
    """Тихий дефолт «пусть будет один» превращает сложные запросы в плохие ответы."""
    with pytest.raises(ValueError):
        subagent_budget("huge")


# -------------------------------------------------------- topological_order
def test_topological_order_follows_a_chain():
    assert topological_order({"pm": ["arch"], "arch": ["eng"], "eng": []}) == [
        "pm", "arch", "eng"
    ]


def test_topological_order_puts_every_predecessor_first():
    order = topological_order(SOP)
    position = {node: i for i, node in enumerate(order)}
    for node, successors in SOP.items():
        for s in successors:
            assert position[node] < position[s]


def test_topological_order_includes_nodes_without_their_own_key():
    """Узел может встречаться только среди преемников — он всё равно участвует."""
    assert topological_order({"pm": ["qa"]}) == ["pm", "qa"]


def test_topological_order_is_deterministic():
    """Одинаковый маршрут при каждом запуске: иначе трассы не сравнить."""
    assert topological_order(SOP) == topological_order(dict(reversed(list(SOP.items()))))


def test_a_cycle_raises_its_own_exception_type():
    with pytest.raises(CyclicRouting):
        topological_order({"a": ["b"], "b": ["a"]})


def test_a_self_loop_is_a_cycle_too():
    with pytest.raises(CyclicRouting):
        topological_order({"a": ["a"]})


# ------------------------------------------------------------ critical_path
def test_critical_path_of_a_chain_counts_every_node():
    assert critical_path({"pm": ["arch"], "arch": ["eng"], "eng": []}) == 3


def test_a_wide_fan_out_stays_shallow():
    """Аргумент MacNet: ширина растёт до тысяч узлов, глубина — почти нет."""
    wide = {"root": ["n%d" % i for i in range(1000)]}
    wide.update({"n%d" % i: [] for i in range(1000)})
    assert critical_path(wide) == 2


def test_critical_path_takes_the_longest_branch_not_the_first():
    assert critical_path(SOP) == 4


def test_critical_path_of_an_empty_graph_is_zero():
    assert critical_path({}) == 0


# ------------------------------------------------------- retirable_versions
def test_a_version_with_in_flight_runs_stays_alive():
    runs = [{"version": "v1", "done": True}, {"version": "v1", "done": False},
            {"version": "v2", "done": True}]
    assert retirable_versions(runs, "v3") == ["v2"]


def test_the_current_version_is_never_retired():
    """Даже без активных прогонов: новые придут через секунду."""
    runs = [{"version": "v2", "done": True}]
    assert retirable_versions(runs, "v2") == []


def test_every_finished_version_is_retirable():
    runs = [{"version": "v1", "done": True}, {"version": "v2", "done": True}]
    assert retirable_versions(runs, "v3") == ["v1", "v2"]


def test_retirable_versions_ignores_versions_without_runs():
    assert retirable_versions([], "v1") == []

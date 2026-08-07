"""Тесты к уроку «Библиотеки навыков и пожизненное обучение (Voyager)». Правь exercise.py."""

import pytest

from exercise import (
    compose_skill,
    dependency_order,
    execute_skill,
    make_skill,
    propose_next_task,
    register_skill,
    search_skills,
)


def always_ok(skill):
    """Верификатор, который всё пропускает."""
    return True, "ok"


def always_bad(skill):
    """Верификатор, который всё отвергает."""
    return False, "crashed on empty inventory"


def primitives():
    """Библиотека из трёх примитивов, собранная через публичный API."""
    library = {}
    for name, description, code, tags in (
        ("mine_ore", "mine iron ore from rock", "mine(3)", ("gather", "ore")),
        ("gather_sticks", "gather sticks from a tree", "gather(2)", ("gather",)),
        ("place_table", "place a crafting table here", "place()", ("setup",)),
    ):
        library, _ = register_skill(
            library, make_skill(name, description, code, tags=tags), always_ok)
    return library


# --------------------------------------------------------------- make_skill
def test_make_skill_starts_at_version_one_with_empty_history():
    skill = make_skill("mine_ore", "mine iron ore from rock", "mine(3)")
    assert (skill["version"], skill["history"], skill["depends_on"]) == (1, (), ())


def test_make_skill_normalizes_tags_and_deps_to_tuples():
    skill = make_skill("craft", "craft a pickaxe", "craft()",
                       tags=["tool"], depends_on=["mine_ore"])
    assert (skill["tags"], skill["depends_on"]) == (("tool",), ("mine_ore",))


def test_make_skill_refuses_an_empty_description():
    """Навык без описания не найдётся никогда — только займёт место."""
    with pytest.raises(ValueError):
        make_skill("mine_ore", "   ", "mine(3)")


# ----------------------------------------------------------- register_skill
def test_register_puts_a_verified_skill_into_the_library():
    library, message = register_skill(
        {}, make_skill("mine_ore", "mine iron ore", "mine(3)"), always_ok)
    assert "mine_ore" in library
    assert message == "registered mine_ore v1"


def test_register_keeps_an_unverified_skill_out_of_the_library():
    """Непроверенный код в библиотеке — это отладка на уровне «почему агент чудит»."""
    library, message = register_skill(
        {}, make_skill("mine_ore", "mine iron ore", "mine(3)"), always_bad)
    assert library == {}
    assert message.startswith("rejected mine_ore:")


def test_register_of_an_existing_name_is_a_refinement_not_a_duplicate():
    library = primitives()
    library, message = register_skill(
        library, make_skill("mine_ore", "mine iron ore from rock", "mine(5)"),
        always_ok)
    assert library["mine_ore"]["version"] == 2
    assert library["mine_ore"]["code"] == "mine(5)"
    assert message == "refined mine_ore -> v2"


def test_register_keeps_the_previous_code_in_history():
    library = primitives()
    library, _ = register_skill(
        library, make_skill("mine_ore", "mine iron ore from rock", "mine(5)"),
        always_ok)
    assert library["mine_ore"]["history"] == ("mine(3)",)


def test_register_leaves_the_input_library_alone():
    library = primitives()
    register_skill(library, make_skill("brew", "brew a potion", "brew()"), always_ok)
    assert "brew" not in library


# ------------------------------------------------------------- search_skills
def test_search_finds_the_skill_by_its_description():
    hits = search_skills(primitives(), "mine iron ore")
    assert hits[0][1]["name"] == "mine_ore"


def test_search_returns_nothing_when_no_word_overlaps():
    """Подсунуть агенту случайный код опаснее, чем ответить «такого нет»."""
    assert search_skills(primitives(), "brew potion") == []


def test_search_tag_filter_is_a_hard_constraint():
    """Похожесть деградирует на сотнях навыков — теги сужают область."""
    hits = search_skills(primitives(), "gather from rock or tree", tag="ore")
    assert [h[1]["name"] for h in hits] == ["mine_ore"]


def test_search_breaks_ties_by_skill_name():
    library = {}
    for name in ("zeta_tool", "alpha_tool"):
        library, _ = register_skill(
            library, make_skill(name, "craft a shiny pickaxe", "x()"), always_ok)
    hits = search_skills(library, "craft a shiny pickaxe")
    assert [h[1]["name"] for h in hits] == ["alpha_tool", "zeta_tool"]


# --------------------------------------------------------- dependency_order
def test_dependency_order_puts_dependencies_first():
    library = primitives()
    library, _ = register_skill(library, compose_skill(
        library, "craft_pickaxe", "craft an iron pickaxe",
        ("mine_ore", "gather_sticks")), always_ok)
    order = dependency_order(library, "craft_pickaxe")
    assert order.index("mine_ore") < order.index("craft_pickaxe")
    assert order.index("gather_sticks") < order.index("craft_pickaxe")


def test_dependency_order_is_reproducible():
    """Разные трассы на одном DAG превращают «иногда падает» в неуловимое."""
    library = primitives()
    library, _ = register_skill(library, compose_skill(
        library, "craft_pickaxe", "craft an iron pickaxe",
        ("mine_ore", "gather_sticks")), always_ok)
    assert dependency_order(library, "craft_pickaxe") == [
        "gather_sticks", "mine_ore", "craft_pickaxe"]


def test_dependency_order_visits_a_shared_dependency_once():
    library = primitives()
    library, _ = register_skill(library, make_skill(
        "a", "step a", "a()", depends_on=("mine_ore",)), always_ok)
    library, _ = register_skill(library, make_skill(
        "b", "step b", "b()", depends_on=("mine_ore",)), always_ok)
    library, _ = register_skill(library, make_skill(
        "top", "step top", "top()", depends_on=("a", "b")), always_ok)
    order = dependency_order(library, "top")
    assert order.count("mine_ore") == 1


def test_dependency_order_raises_on_a_missing_skill():
    with pytest.raises(KeyError):
        dependency_order(primitives(), "brew_potion")


def test_dependency_order_refuses_a_cycle_instead_of_running_half_the_dag():
    """У цикла нет «первого» навыка: частичный запуск оставит среду неизвестной."""
    library = {}
    library, _ = register_skill(library, make_skill(
        "a", "step a", "a()", depends_on=("b",)), always_ok)
    library, _ = register_skill(library, make_skill(
        "b", "step b", "b()", depends_on=("a",)), always_ok)
    with pytest.raises(ValueError):
        dependency_order(library, "a")


# ------------------------------------------------------------ compose_skill
def test_compose_records_the_subskills_as_dependencies():
    library = primitives()
    composed = compose_skill(library, "craft_pickaxe", "craft an iron pickaxe",
                             ("mine_ore", "gather_sticks"))
    assert composed["depends_on"] == ("mine_ore", "gather_sticks")


def test_compose_keeps_the_order_the_author_wrote():
    """Сначала добыть руду, потом крафтить — это смысл, а не деталь."""
    library = primitives()
    composed = compose_skill(library, "craft_pickaxe", "craft an iron pickaxe",
                             ("mine_ore", "place_table"))
    assert composed["code"] == "mine_ore(); place_table()"


def test_compose_raises_on_a_missing_subskill():
    with pytest.raises(KeyError):
        compose_skill(primitives(), "craft", "craft a pickaxe", ("brew_potion",))


def test_compose_does_not_register_anything_by_itself():
    """Составной навык проходит тот же verify: из рабочих кусков бывает брак."""
    library = primitives()
    compose_skill(library, "craft_pickaxe", "craft an iron pickaxe", ("mine_ore",))
    assert "craft_pickaxe" not in library


# ------------------------------------------------------------ execute_skill
def test_execute_runs_dependencies_before_the_skill():
    library = primitives()
    library, _ = register_skill(library, compose_skill(
        library, "craft_pickaxe", "craft an iron pickaxe",
        ("mine_ore", "gather_sticks")), always_ok)

    def note(tag):
        def run(env):
            env.setdefault("steps", []).append(tag)
            return tag
        return run

    runtime = {"mine_ore": note("ore"), "gather_sticks": note("stick"),
               "craft_pickaxe": note("craft")}
    env, log, ok = execute_skill(library, "craft_pickaxe", runtime)
    assert ok is True
    assert env["steps"] == ["stick", "ore", "craft"]
    assert len(log) == 3


def test_execute_does_not_rebind_keys_in_the_environment_it_was_given():
    env = {"ore": 0}
    _, _, ok = execute_skill(primitives(), "mine_ore",
                             {"mine_ore": lambda e: e.update(ore=3) or "+3 ore"}, env)
    assert ok is True
    assert env == {"ore": 0}


def test_execute_turns_a_crash_into_feedback_naming_the_skill_and_version():
    """Именно этот текст уедет в промпт следующей итерации Voyager."""
    library = primitives()

    def boom(env):
        raise RuntimeError("need 3 ore, have 0")

    _, log, ok = execute_skill(library, "mine_ore", {"mine_ore": boom})
    assert ok is False
    assert log == ["error in mine_ore v1: RuntimeError: need 3 ore, have 0"]


def test_execute_stops_at_the_first_failure():
    """Следующие навыки рассчитаны на состояние, которого теперь нет."""
    library = primitives()
    library, _ = register_skill(library, compose_skill(
        library, "craft_pickaxe", "craft an iron pickaxe",
        ("mine_ore", "gather_sticks")), always_ok)

    def boom(env):
        raise RuntimeError("no rock nearby")

    runtime = {"mine_ore": boom, "gather_sticks": lambda e: "stick",
               "craft_pickaxe": lambda e: "craft"}
    _, log, ok = execute_skill(library, "craft_pickaxe", runtime)
    assert ok is False
    assert len(log) == 2


def test_execute_reports_a_skill_with_no_runtime_as_a_failure():
    _, log, ok = execute_skill(primitives(), "mine_ore", {})
    assert ok is False
    assert "mine_ore" in log[0]


# -------------------------------------------------------- propose_next_task
def test_propose_returns_the_first_capability_gap():
    """Задача чуть выше текущего уровня, а не самая далёкая."""
    wanted = ("mine iron ore from rock", "brew a healing potion",
              "tame a wolf in the forest")
    assert propose_next_task(primitives(), wanted) == "brew a healing potion"


def test_propose_returns_none_when_everything_is_covered():
    wanted = ("mine iron ore from rock", "gather sticks from a tree")
    assert propose_next_task(primitives(), wanted) is None


def test_propose_on_an_empty_library_returns_the_very_first_item():
    assert propose_next_task({}, ("mine iron ore from rock",)) == \
        "mine iron ore from rock"


def test_propose_threshold_decides_what_counts_as_covered():
    """Низкий порог объявит умение закрытым по одному случайному слову."""
    wanted = ("mine a diamond from deep rock",)
    assert propose_next_task(primitives(), wanted, threshold=0.9) == wanted[0]
    assert propose_next_task(primitives(), wanted, threshold=0.1) is None

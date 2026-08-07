"""Тесты к уроку «Модель чужого сознания и координация». Правь exercise.py."""

import random

import pytest

from exercise import (
    BASKET_A,
    BASKET_B,
    MARBLE,
    belief_of,
    choose_box,
    new_agent,
    observe,
    predict_search,
    sally_anne,
    simulate_collection,
    update_belief,
)


def valued(order, **values):
    """Агент, который сам оценил ящики и приписал ту же оценку соседу a0."""
    agent = new_agent("me", order)
    for box, value in values.items():
        observe(agent, box, value, witnesses=("a0",))
    return agent


# --------------------------------------------------------------- new_agent
def test_new_agent_starts_with_no_beliefs_at_all():
    agent = new_agent("observer", 1)
    assert agent["beliefs"] == {} and agent["models"] == {}


def test_every_supported_order_can_be_created():
    assert [new_agent("x", o)["order"] for o in (0, 1, 2)] == [0, 1, 2]


def test_an_unsupported_order_is_rejected():
    for bad in (-1, 3):
        with pytest.raises(ValueError):
            new_agent("x", bad)


# --------------------------------------------- update_belief / belief_of
def test_own_belief_round_trips():
    agent = update_belief(new_agent("me", 0), (), MARBLE, BASKET_B)
    assert belief_of(agent, (), MARBLE) == BASKET_B


def test_a_model_of_someone_else_is_kept_apart_from_my_own_belief():
    agent = new_agent("me", 1)
    update_belief(agent, (), MARBLE, BASKET_B)
    update_belief(agent, ("sally",), MARBLE, BASKET_A)
    assert belief_of(agent, (), MARBLE) == BASKET_B
    assert belief_of(agent, ("sally",), MARBLE) == BASKET_A


def test_second_order_records_what_they_think_i_know():
    agent = new_agent("me", 2)
    update_belief(agent, ("sally", "anne"), MARBLE, BASKET_A)
    assert belief_of(agent, ("sally", "anne"), MARBLE) == BASKET_A


def test_reading_a_model_that_was_never_built_returns_the_default():
    agent = new_agent("me", 2)
    assert belief_of(agent, ("klaus",), MARBLE, default="нет данных") == "нет данных"


def test_a_zeroth_order_agent_cannot_hold_a_model_of_anyone():
    with pytest.raises(ValueError):
        update_belief(new_agent("me", 0), ("sally",), MARBLE, BASKET_A)


def test_a_first_order_agent_cannot_go_two_levels_deep():
    with pytest.raises(ValueError):
        update_belief(new_agent("me", 1), ("sally", "anne"), MARBLE, BASKET_A)


# ----------------------------------------------------------------- observe
def test_an_observation_updates_the_agents_own_belief():
    agent = observe(new_agent("me", 0), MARBLE, BASKET_A)
    assert belief_of(agent, (), MARBLE) == BASKET_A


def test_a_first_order_agent_credits_the_witnesses_with_what_they_saw():
    agent = observe(new_agent("me", 1), MARBLE, BASKET_A, witnesses=("sally", "anne"))
    assert belief_of(agent, ("anne",), MARBLE) == BASKET_A


def test_a_zeroth_order_agent_has_nowhere_to_put_the_witnesses():
    agent = observe(new_agent("me", 0), MARBLE, BASKET_A, witnesses=("sally",))
    assert agent["models"] == {}


def test_a_second_order_agent_knows_that_they_know_that_it_knows():
    agent = observe(new_agent("me", 2), MARBLE, BASKET_A, witnesses=("anne",))
    assert belief_of(agent, ("anne", "me"), MARBLE) == BASKET_A


# ---------------------------------------------------------- predict_search
def test_a_zeroth_order_agent_projects_its_own_belief_onto_others():
    agent = observe(new_agent("me", 0), MARBLE, BASKET_B)
    assert predict_search(agent, "sally", MARBLE) == BASKET_B


def test_a_first_order_agent_reads_the_model_of_the_other():
    agent = new_agent("me", 1)
    update_belief(agent, (), MARBLE, BASKET_B)
    update_belief(agent, ("sally",), MARBLE, BASKET_A)
    assert predict_search(agent, "sally", MARBLE) == BASKET_A


def test_prediction_falls_back_to_own_belief_for_an_unmodelled_agent():
    agent = observe(new_agent("me", 1), MARBLE, BASKET_B)
    assert predict_search(agent, "klaus", MARBLE) == BASKET_B


# -------------------------------------------------------------- sally_anne
def test_without_a_model_of_others_the_agent_fails_sally_anne():
    """Ответ про шарик, а не про убеждение Салли — ошибка трёхлетнего."""
    assert sally_anne(0) == BASKET_B


def test_a_first_order_model_passes_sally_anne():
    assert sally_anne(1) == BASKET_A


def test_a_second_order_model_passes_it_too():
    assert sally_anne(2) == BASKET_A


def test_passing_the_test_does_not_mean_being_confused_about_the_marble():
    """Агент отлично знает, где шарик; он просто не путает это с знанием Салли."""
    observer = new_agent("observer", 1)
    observe(observer, MARBLE, BASKET_A, witnesses=("sally", "anne"))
    observe(observer, MARBLE, BASKET_B, witnesses=("anne",))
    assert belief_of(observer, (), MARBLE) == BASKET_B
    assert belief_of(observer, ("sally",), MARBLE) == BASKET_A


# -------------------------------------------------------------- choose_box
def test_alone_the_agent_takes_its_favourite_box():
    agent = valued(1, box0=0.9, box1=0.2)
    assert choose_box(agent, ["box0", "box1"]) == "box0"


def test_a_first_order_agent_steps_aside_from_the_predicted_pick():
    agent = valued(1, box0=0.9, box1=0.2)
    assert choose_box(agent, ["box0", "box1"], others=("a0",)) == "box1"


def test_a_zeroth_order_agent_walks_into_the_occupied_box():
    agent = valued(0, box0=0.9, box1=0.2)
    assert choose_box(agent, ["box0", "box1"], others=("a0",)) == "box0"


def test_when_every_box_is_predicted_taken_the_agent_still_picks_one():
    agent = valued(1, box0=0.9, box1=0.2)
    assert choose_box(agent, ["box0", "box1"], others=("a0", "a1")) in ("box0", "box1")


def test_choose_box_without_boxes_raises():
    with pytest.raises(ValueError):
        choose_box(valued(1, box0=0.9), [])


# ------------------------------------------------------ simulate_collection
def test_theory_of_mind_cuts_the_duplication_rate():
    blind = simulate_collection(3, 3, 0, random.Random(0), trials=300)
    aware = simulate_collection(3, 3, 1, random.Random(0), trials=300)
    assert aware["duplication_rate"] < blind["duplication_rate"] / 2


def test_theory_of_mind_raises_the_completion_rate():
    blind = simulate_collection(3, 3, 0, random.Random(1), trials=300)
    aware = simulate_collection(3, 3, 1, random.Random(1), trials=300)
    assert aware["completion_rate"] > blind["completion_rate"]


def test_blind_agents_collide_most_of_the_time():
    blind = simulate_collection(5, 5, 0, random.Random(2), trials=200)
    assert blind["duplication_rate"] > 0.5


def test_both_rates_stay_inside_the_unit_interval():
    for order in (0, 1, 2):
        rates = simulate_collection(4, 4, order, random.Random(3), trials=100)
        assert all(0.0 <= v <= 1.0 for v in rates.values())


def test_the_measurement_is_reproducible_for_a_given_seed():
    first = simulate_collection(3, 3, 1, random.Random(4), trials=100)
    second = simulate_collection(3, 3, 1, random.Random(4), trials=100)
    assert first == second

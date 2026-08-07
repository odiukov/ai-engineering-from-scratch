"""Тесты к уроку «Мультимодальные агенты и computer-use». Правь exercise.py."""

import pytest

from exercise import (
    ACTION_SCHEMA,
    agent_loop,
    apply_action,
    compress_history,
    parse_action,
    recover,
    scale_click,
    success_rate,
    validate_action,
)


def booking_page():
    """Мок страницы бронирования: кнопка поиска, поле города, ссылка на рейс."""
    return {
        "url": "https://booking.example/search",
        "elements": [
            {"desc": "Search button", "bbox": (300, 40, 400, 80),
             "goto": "https://booking.example/results"},
            {"desc": "City field", "bbox": (100, 40, 280, 80), "goto": None},
            {"desc": "Flight NH106", "bbox": (100, 200, 500, 240),
             "goto": "https://booking.example/checkout"},
        ],
        "fields": {},
        "error": None,
    }


# ---------------------------------------------------------- validate_action
def test_a_well_formed_click_has_no_complaints():
    assert validate_action({"action": "click", "x": 10, "y": 20}) == []


def test_a_missing_required_field_is_named():
    assert validate_action({"action": "click", "x": 10}) == ["missing field: y"]


def test_an_action_outside_the_schema_is_rejected():
    assert validate_action({"action": "fly", "x": 1}) == ["unknown action: fly"]


def test_a_reply_without_an_action_key_is_rejected():
    assert validate_action({"x": 1, "y": 2}) == ["missing key: action"]


def test_extra_fields_are_allowed_because_recovery_needs_them():
    """element_desc не в схеме, но именно он спасает после промаха."""
    action = {"action": "click", "x": 10, "y": 20, "element_desc": "Search button"}
    assert validate_action(action) == []


def test_validation_reports_problems_instead_of_raising():
    """VLM ломает схему регулярно; агент показывает список претензий и просит ещё раз."""
    errors = validate_action({"action": "drag"})
    assert len(errors) == len(ACTION_SCHEMA["drag"])


# ------------------------------------------------------------- parse_action
def test_parse_action_reads_a_bare_json_object():
    assert parse_action('{"action": "click", "x": 1, "y": 2}') == {
        "action": "click", "x": 1, "y": 2,
    }


def test_parse_action_ignores_the_markdown_fence():
    reply = 'Сначала нажму поиск.\n```json\n{"action": "wait", "ms": 5}\n```'
    assert parse_action(reply) == {"action": "wait", "ms": 5}


def test_parse_action_ignores_reasoning_around_the_json():
    reply = 'Мне нужен город. {"action": "type", "text": "Tokyo"} Дальше — поиск.'
    assert parse_action(reply) == {"action": "type", "text": "Tokyo"}


def test_a_reply_without_json_is_an_error_not_an_empty_action():
    """Пустой словарь агент выполнил бы как «ничего» и решил, что шаг удался."""
    with pytest.raises(ValueError):
        parse_action("Не понимаю, что тут делать.")


def test_broken_json_is_an_error():
    with pytest.raises(ValueError):
        parse_action('{"action": "click", "x": }')


# -------------------------------------------------------------- scale_click
def test_scale_click_at_the_same_resolution_changes_nothing():
    action = {"action": "click", "x": 100, "y": 50}
    assert scale_click(action, (1000, 500), (1000, 500)) == action


def test_scale_click_maps_model_coordinates_onto_the_real_screen():
    assert scale_click(
        {"action": "click", "x": 100, "y": 50}, (1000, 500), (2000, 1000)
    ) == {"action": "click", "x": 200, "y": 100}


def test_each_axis_gets_its_own_factor():
    """Пропорции при ресайзе почти никогда не сохраняются."""
    assert scale_click(
        {"action": "click", "x": 100, "y": 100}, (1000, 1000), (2560, 1440)
    ) == {"action": "click", "x": 256, "y": 144}


def test_scale_click_rescales_every_corner_of_a_drag():
    scaled = scale_click(
        {"action": "drag", "x0": 10, "y0": 20, "x1": 30, "y1": 40},
        (100, 100), (200, 400),
    )
    assert scaled == {"action": "drag", "x0": 20, "y0": 80, "x1": 60, "y1": 160}


def test_scale_click_leaves_non_coordinate_fields_and_the_input_alone():
    action = {"action": "click", "x": 100, "y": 50, "element_desc": "Search button"}
    scaled = scale_click(action, (1000, 500), (2000, 1000))
    assert scaled["element_desc"] == "Search button"
    assert action["x"] == 100


# ------------------------------------------------------------- apply_action
def test_clicking_a_link_navigates():
    state = apply_action(booking_page(), {"action": "click", "x": 350, "y": 60})
    assert state["url"] == "https://booking.example/results"
    assert state["error"] is None


def test_clicking_empty_space_records_an_error_and_keeps_the_page():
    state = apply_action(booking_page(), {"action": "click", "x": 900, "y": 900})
    assert state["error"] is not None
    assert state["url"] == "https://booking.example/search"


def test_typing_fills_a_field():
    state = apply_action(
        booking_page(), {"action": "type", "text": "Tokyo", "field": "city"}
    )
    assert state["fields"]["city"] == "Tokyo"


def test_a_successful_step_clears_the_previous_error():
    """Иначе одна старая ошибка тянется через весь эпизод."""
    failed = apply_action(booking_page(), {"action": "click", "x": 900, "y": 900})
    fixed = apply_action(failed, {"action": "click", "x": 350, "y": 60})
    assert fixed["error"] is None


def test_apply_action_does_not_mutate_the_state_it_was_given():
    """Бенчмарк гоняет один стартовый state по десяти задачам подряд."""
    state = booking_page()
    apply_action(state, {"action": "type", "text": "Tokyo", "field": "city"})
    apply_action(state, {"action": "click", "x": 350, "y": 60})
    assert state["fields"] == {}
    assert state["url"] == "https://booking.example/search"


# ------------------------------------------------------------------ recover
def test_recover_regrounds_a_missed_click_by_its_description():
    missed = {"action": "click", "x": 0, "y": 0, "element_desc": "Search button"}
    fixed = recover(missed, booking_page())
    assert (fixed["x"], fixed["y"]) == (350, 60)


def test_the_regrounded_click_actually_works():
    """Проверка не на координаты, а на результат: перецеленный клик уводит на нужный url."""
    missed = {"action": "click", "x": 0, "y": 0, "element_desc": "Flight NH106"}
    state = apply_action(booking_page(), recover(missed, booking_page()))
    assert state["url"] == "https://booking.example/checkout"
    assert state["error"] is None


def test_recover_gives_up_when_the_description_matches_nothing():
    missed = {"action": "click", "x": 0, "y": 0, "element_desc": "Buy crypto"}
    assert recover(missed, booking_page()) is None


def test_recover_gives_up_without_a_semantic_hint():
    """Клик без element_desc перецелить не по чему — надо перепланировать."""
    assert recover({"action": "click", "x": 0, "y": 0}, booking_page()) is None


# --------------------------------------------------------- compress_history
def test_compression_keeps_only_the_last_screenshots_live():
    history = [{"screenshot": f"img{i}", "action": {"action": "wait", "ms": i}}
               for i in range(20)]
    compressed = compress_history(history, keep_live=4)
    assert sum(1 for s in compressed if s["screenshot"] is not None) == 4
    assert [s["screenshot"] for s in compressed[-4:]] == ["img16", "img17", "img18", "img19"]


def test_compression_never_drops_an_action():
    """Скриншоты снимаем, лог действий — нет: он и есть память агента."""
    history = [{"screenshot": f"img{i}", "action": {"action": "wait", "ms": i}}
               for i in range(20)]
    compressed = compress_history(history, keep_live=4)
    assert [s["action"]["ms"] for s in compressed] == list(range(20))


def test_a_short_episode_is_left_alone():
    history = [{"screenshot": "img0", "action": {"action": "wait", "ms": 0}}]
    assert compress_history(history, keep_live=4) == history


def test_compression_does_not_mutate_the_history():
    history = [{"screenshot": f"img{i}", "action": {"action": "wait", "ms": i}}
               for i in range(10)]
    compress_history(history, keep_live=2)
    assert all(s["screenshot"] is not None for s in history)


# --------------------------------------------------------------- agent_loop
def test_the_loop_stops_when_the_agent_reports_done():
    plan = [
        {"action": "click", "x": 350, "y": 60},
        {"action": "done", "success": True},
        {"action": "click", "x": 1, "y": 1},
    ]
    steps = iter(plan)
    _, trace = agent_loop(booking_page(), lambda s: next(steps), max_steps=10)
    assert trace == plan[:2]


def test_the_loop_respects_the_step_budget():
    """Агент, не понявший, что задача решена, иначе выжжет весь бюджет вызовов VLM."""
    _, trace = agent_loop(
        booking_page(), lambda s: {"action": "click", "x": 1, "y": 1}, max_steps=5
    )
    assert len(trace) == 5


def test_the_loop_feeds_each_action_back_into_the_state():
    plan = iter([
        {"action": "type", "text": "Tokyo", "field": "city"},
        {"action": "click", "x": 350, "y": 60},
        {"action": "done", "success": True},
    ])
    final, _ = agent_loop(booking_page(), lambda s: next(plan), max_steps=10)
    assert final["fields"]["city"] == "Tokyo"
    assert final["url"] == "https://booking.example/results"


def test_done_is_reported_but_not_executed():
    """"done" — это отчёт агента, а не событие в браузере."""
    plan = iter([{"action": "done", "success": True}])
    final, trace = agent_loop(booking_page(), lambda s: next(plan), max_steps=10)
    assert trace == [{"action": "done", "success": True}]
    assert final == booking_page()


# ------------------------------------------------------------- success_rate
def test_success_rate_of_a_mixed_run():
    assert success_rate([{"success": True}, {"success": False}]) == pytest.approx(0.5)


def test_success_rate_of_an_empty_run_is_zero_not_one():
    """Ноль задач — это сломанный харнесс, а не идеальный результат."""
    assert success_rate([]) == pytest.approx(0.0)


def test_success_rate_matches_the_benchmark_numbers_from_the_lesson():
    """VisualWebArena: открытая SOTA ~20%, Gemini 3 Pro ~27%."""
    results = [{"success": i < 2} for i in range(10)]
    assert success_rate(results) == pytest.approx(0.2)

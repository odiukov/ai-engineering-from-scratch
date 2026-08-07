"""Тесты к уроку «Полный vision-pipeline: капстоун». Правь exercise.py."""

import pytest

from exercise import (
    attach_classifications,
    bottleneck_stage,
    build_result,
    clamp_box,
    is_classifiable,
    select_crops,
    validate_box,
    validate_detection,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(boxes):
    """Развернуть список боксов в плоский список: pytest.approx не умеет вложенность."""
    return [v for b in boxes for v in b]


# ------------------------------------------------------------- validate_box
def test_validate_box_returns_four_floats():
    box = validate_box([10, 20, 30, 40])
    assert box == APPROX((10.0, 20.0, 30.0, 40.0))
    assert all(isinstance(v, float) for v in box)


def test_validate_box_rejects_wrong_number_of_coordinates():
    with pytest.raises(ValueError):
        validate_box([10, 20, 30])


def test_validate_box_rejects_center_width_height_format():
    """(cx, cy, w, h) = (50, 50, 10, 10) прикидывается боксом, но x2 < x1."""
    with pytest.raises(ValueError):
        validate_box((50, 50, 10, 10))


def test_validate_box_accepts_zero_area_box():
    """Схлопнутый бокс — легальное значение, а не ошибка."""
    assert validate_box((0, 0, 0, 0)) == APPROX((0.0, 0.0, 0.0, 0.0))


# ------------------------------------------------------- validate_detection
def test_validate_detection_builds_the_contract_dict():
    det = validate_detection([0, 0, 10, 10], 0.9, 3)
    assert det["box"] == APPROX((0.0, 0.0, 10.0, 10.0))
    assert det["score"] == APPROX(0.9)
    assert det["class_id"] == 3


def test_validate_detection_accepts_score_exactly_one():
    """После softmax уверенность честно бывает 1.0 — падать на этом нельзя."""
    assert validate_detection([0, 0, 1, 1], 1.0, 0)["score"] == APPROX(1.0)


def test_validate_detection_rejects_score_outside_unit_range():
    with pytest.raises(ValueError):
        validate_detection([0, 0, 1, 1], 1.4, 0)


def test_validate_detection_rejects_negative_class_id():
    with pytest.raises(ValueError):
        validate_detection([0, 0, 1, 1], 0.5, -1)


def test_validate_detection_also_enforces_the_box_contract():
    with pytest.raises(ValueError):
        validate_detection([50, 50, 10, 10], 0.5, 0)


# --------------------------------------------------------------- clamp_box
def test_clamp_box_pulls_negative_corner_to_zero():
    assert clamp_box((-5, -5, 50, 50), 100, 100) == APPROX((0.0, 0.0, 50.0, 50.0))


def test_clamp_box_caps_at_image_size():
    assert clamp_box((10, 10, 999, 999), 100, 80) == APPROX((10.0, 10.0, 100.0, 80.0))


def test_clamp_box_leaves_an_inside_box_untouched():
    assert clamp_box((10, 20, 30, 40), 100, 100) == APPROX((10.0, 20.0, 30.0, 40.0))


def test_clamp_box_of_an_offscreen_box_collapses_to_zero_area():
    """Бокс целиком за кадром схлопывается, а не превращается в исключение."""
    x1, y1, x2, y2 = clamp_box((200, 200, 300, 300), 100, 100)
    assert (x2 - x1) == APPROX(0.0)
    assert (y2 - y1) == APPROX(0.0)


def test_clamp_box_preserves_corner_order():
    """Зажатие монотонно: результат обязан снова пройти validate_box."""
    clamped = clamp_box((-90, -90, -10, -10), 100, 100)
    assert validate_box(clamped) == APPROX(clamped)


# ---------------------------------------------------------- is_classifiable
def test_is_classifiable_accepts_a_large_enough_box():
    assert is_classifiable((0, 0, 64, 64), 32) is True


def test_is_classifiable_needs_both_sides_not_just_the_area():
    """64x8 по площади больше, чем 32x32, но это полоса, а не изображение."""
    assert is_classifiable((0, 0, 64, 8), 32) is False


def test_is_classifiable_boundary_is_inclusive():
    assert is_classifiable((0, 0, 32, 32), 32) is True


# ------------------------------------------------------------- select_crops
def test_select_crops_returns_one_clamped_box_per_input():
    boxes = [(0, 0, 50, 50), (0, 0, 4, 4), (-10, -10, 200, 200)]
    clamped, valid = select_crops(boxes, 100, 100, 32)
    assert len(clamped) == len(boxes)
    assert flat(clamped) == APPROX(
        [0.0, 0.0, 50.0, 50.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 100.0, 100.0]
    )
    assert valid == [0, 2]


def test_select_crops_indices_point_at_original_positions():
    """Мост между двумя нумерациями: мелкие боксы не сдвигают индексы крупных."""
    boxes = [(0, 0, 2, 2), (0, 0, 2, 2), (0, 0, 90, 90)]
    _, valid = select_crops(boxes, 100, 100, 32)
    assert valid == [2]


def test_select_crops_on_an_empty_image_returns_two_empty_lists():
    assert select_crops([], 100, 100, 32) == ([], [])


def test_select_crops_disqualifies_a_box_that_shrinks_after_clamping():
    """До зажатия бокс 40x40, после — 10x40: порядок «сначала зажать» важен."""
    _, valid = select_crops([(90, 10, 130, 50)], 100, 100, 32)
    assert valid == []


# ------------------------------------------------- attach_classifications
def test_attach_classifications_resolves_names_from_the_label_map():
    out = attach_classifications([0, 2], [(1, 0.8), (0, 0.6)], ["cat", "dog"])
    assert [c["class_name"] for c in out] == ["dog", "cat"]
    assert [c["detection_index"] for c in out] == [0, 2]
    assert [c["score"] for c in out] == APPROX([0.8, 0.6])


def test_attach_classifications_keeps_detection_index_not_crop_index():
    """Индекс в ответе — номер ДЕТЕКЦИИ, а не позиция кропа в батче."""
    out = attach_classifications([5, 9], [(0, 0.5), (0, 0.5)], ["cat"])
    assert [c["detection_index"] for c in out] == [5, 9]


def test_attach_classifications_rejects_length_mismatch():
    """zip молча обрезал бы по короткому и приклеил чужой класс."""
    with pytest.raises(ValueError):
        attach_classifications([0, 1, 2], [(0, 0.5)], ["cat"])


def test_attach_classifications_rejects_class_id_outside_the_label_map():
    with pytest.raises(ValueError):
        attach_classifications([0], [(7, 0.5)], ["cat", "dog"])


def test_attach_classifications_on_no_crops_returns_empty_list():
    assert attach_classifications([], [], ["cat"]) == []


# ------------------------------------------------------------ build_result
def test_build_result_with_no_detections_is_a_valid_answer():
    res = build_result("demo", [], [], 12.5)
    assert res == {
        "image_id": "demo",
        "detections": [],
        "classifications": [],
        "inference_ms": APPROX(12.5),
    }


def test_build_result_rejects_a_dangling_detection_index():
    det = [validate_detection([0, 0, 40, 40], 0.9, 0)]
    cls = attach_classifications([3], [(0, 0.5)], ["cat"])
    with pytest.raises(ValueError):
        build_result("demo", det, cls, 1.0)


def test_build_result_rejects_negative_inference_time():
    with pytest.raises(ValueError):
        build_result("demo", [], [], -0.1)


def test_build_result_does_not_alias_the_input_lists():
    """Ответ ушёл клиенту — дальнейшая работа с исходным списком его не портит."""
    detections = []
    res = build_result("demo", detections, [], 1.0)
    detections.append(validate_detection([0, 0, 40, 40], 0.9, 0))
    assert res["detections"] == []


# --------------------------------------------------------- bottleneck_stage
def test_bottleneck_stage_names_the_slowest_stage():
    name, share = bottleneck_stage(
        {"preprocess": [3.0, 3.0], "detect": [400.0, 400.0], "classify": [97.0, 97.0]}
    )
    assert name == "detect"
    assert share == APPROX(0.8)


def test_bottleneck_stage_share_is_relative_to_the_whole_pipeline():
    """Две одинаковые стадии дают 0.5, а не 1.0."""
    _, share = bottleneck_stage({"a": [10.0], "b": [10.0]})
    assert share == APPROX(0.5)


def test_bottleneck_stage_uses_median_so_one_outlier_does_not_win():
    """У 'a' среднее 25.75, у 'b' — 10. По среднему победила бы 'a'."""
    name, _ = bottleneck_stage({"a": [1.0, 1.0, 1.0, 100.0], "b": [10.0] * 4})
    assert name == "b"


def test_bottleneck_stage_rejects_empty_input():
    with pytest.raises(ValueError):
        bottleneck_stage({})


def test_bottleneck_stage_rejects_a_stage_without_measurements():
    with pytest.raises(ValueError):
        bottleneck_stage({"detect": [10.0], "classify": []})

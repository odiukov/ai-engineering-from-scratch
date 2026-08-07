"""Тесты к уроку «Аудио-языковые модели». Правь exercise.py."""

import pytest

from exercise import (
    accuracy_by_category,
    build_lm_sequence,
    gate_on_silence,
    gelu,
    is_above_chance,
    linear,
    project,
    trainable_parameter_count,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """Разворачивает список списков в плоский: pytest.approx не умеет вложенные."""
    return [x for row in M for x in row]


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def fake_embed(text):
    """Заглушка вместо токенайзера: один вектор на символ."""
    return [[float(ord(ch))] for ch in text]


# ----------------------------------------------------------------- linear
def test_linear_with_identity_matrix_only_adds_the_bias():
    assert linear([1.0, 2.0], eye(2), [10.0, 20.0]) == APPROX([11.0, 22.0])


def test_linear_changes_the_dimension():
    """Ради этого проектор и существует: 2 признака аудио → 3 измерения LLM."""
    W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert len(linear([3.0, 4.0], W, [0.0, 0.0, 0.0])) == 3


def test_linear_computes_a_dot_product_per_row():
    assert linear([1.0, 2.0], [[1.0, 1.0]], [5.0]) == APPROX([8.0])


def test_linear_rejects_shapes_that_do_not_fit():
    with pytest.raises(ValueError):
        linear([1.0, 2.0], [[1.0, 0.0]], [0.0, 0.0])
    with pytest.raises(ValueError):
        linear([1.0, 2.0], [[1.0, 0.0, 0.0]], [0.0])


# ------------------------------------------------------------------- gelu
def test_gelu_of_zero_is_zero():
    assert gelu([0.0]) == APPROX([0.0])


def test_gelu_passes_large_positives_almost_unchanged():
    assert gelu([10.0])[0] == pytest.approx(10.0, abs=1e-6)


def test_gelu_almost_kills_large_negatives():
    assert gelu([-10.0])[0] == pytest.approx(0.0, abs=1e-6)


def test_gelu_dips_below_zero_unlike_relu():
    """Отрицательный провал — то, чем GELU отличается от ReLU."""
    y = gelu([-1.0])[0]
    assert -0.2 < y < 0.0


# ---------------------------------------------------------------- project
def test_project_output_dimension_comes_from_the_last_layer():
    """Проектор — мост: выход всегда в размерности LLM, вход какой угодно."""
    W1 = [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]      # 3 признака аудио -> 2
    W2 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]  # 2 -> 4 (dim LLM)
    out = project([[1.0, 2.0, 3.0]], [(W1, [0.0, 0.0]), (W2, [0.0] * 4)])
    assert len(out[0]) == 4


def test_project_keeps_the_number_of_frames():
    """Сколько кадров дал энкодер — столько аудио-токенов увидит LLM."""
    frames = [[1.0], [2.0], [3.0]]
    layers = [([[1.0]], [0.0]), ([[1.0]], [0.0])]
    assert len(project(frames, layers)) == 3


def test_project_applies_gelu_between_layers():
    """Два единичных слоя без нелинейности вернули бы вход как есть."""
    out = project([[-1.0]], [([[1.0]], [0.0]), ([[1.0]], [0.0])])
    assert out[0][0] == APPROX(gelu([-1.0])[0])
    assert out[0][0] != APPROX(-1.0)


def test_project_does_not_apply_gelu_after_the_last_layer():
    """Последний выход уходит в LLM как есть, иначе отрицательные срежутся."""
    out = project([[-1.0]], [([[1.0]], [0.0])])
    assert flat(out) == APPROX([-1.0])


def test_project_rejects_an_empty_stack():
    with pytest.raises(ValueError):
        project([[1.0]], [])


# ------------------------------------------------------- build_lm_sequence
def test_build_lm_sequence_keeps_the_order_of_modalities():
    parts = [("text", "a"), ("audio", [[7.0]]), ("text", "b")]
    assert flat(build_lm_sequence(parts, fake_embed, 1)) == APPROX(
        [float(ord("a")), 7.0, float(ord("b"))]
    )


def test_build_lm_sequence_makes_audio_indistinguishable_from_text():
    """Декодер видит один плоский список векторов одной размерности."""
    parts = [("text", "ab"), ("audio", [[1.0], [2.0]])]
    seq = build_lm_sequence(parts, fake_embed, 1)
    assert {len(v) for v in seq} == {1}


def test_build_lm_sequence_rejects_a_vector_of_the_wrong_size():
    with pytest.raises(ValueError):
        build_lm_sequence([("audio", [[1.0, 2.0]])], fake_embed, 1)


def test_build_lm_sequence_rejects_an_unknown_modality():
    with pytest.raises(ValueError):
        build_lm_sequence([("video", [[1.0]])], fake_embed, 1)


# ------------------------------------------------ trainable_parameter_count
def test_stage_one_trains_only_the_projector():
    modules = {"encoder": 6.4e8, "projector": 5.0e6, "llm": 7.0e9}
    assert trainable_parameter_count(modules, ["projector"]) == APPROX(5.0e6)


def test_stage_one_is_a_tiny_fraction_of_the_model():
    modules = {"encoder": 6.4e8, "projector": 5.0e6, "llm": 7.0e9}
    total = trainable_parameter_count(modules, modules)
    assert trainable_parameter_count(modules, ["projector"]) / total < 0.01


def test_freezing_everything_leaves_nothing_to_train():
    assert trainable_parameter_count({"llm": 7.0e9}, []) == APPROX(0.0)


def test_trainable_parameter_count_rejects_a_typo_in_a_module_name():
    """Опечатка молча оставила бы модуль замороженным — пусть падает."""
    with pytest.raises(ValueError):
        trainable_parameter_count({"projector": 5.0e6}, ["projecter"])


# ------------------------------------------------------ accuracy_by_category
def _item(cat, ok):
    return {"category": cat, "predicted": "a", "correct": "a" if ok else "b"}


def test_accuracy_by_category_scores_each_category():
    res = accuracy_by_category([_item("speech", True), _item("multi", False)])
    assert res["speech"] == APPROX(1.0)
    assert res["multi"] == APPROX(0.0)


def test_overall_counts_questions_not_categories():
    """Среднее средних соврало бы: у категорий разное число вопросов."""
    items = [_item("speech", True)] * 9 + [_item("multi", False)]
    res = accuracy_by_category(items)
    assert res["overall"] == APPROX(0.9)


def test_aggregate_hides_a_failing_subset():
    """Главная мысль урока: общий балл приличный, а multi-audio на нуле."""
    items = [_item("speech", True)] * 8 + [_item("multi", False)] * 2
    res = accuracy_by_category(items)
    assert res["overall"] > 0.75 and res["multi"] == APPROX(0.0)


def test_accuracy_by_category_rejects_a_category_named_overall():
    with pytest.raises(ValueError):
        accuracy_by_category([_item("overall", True)])


def test_accuracy_by_category_rejects_an_empty_run():
    with pytest.raises(ValueError):
        accuracy_by_category([])


# --------------------------------------------------------- is_above_chance
def test_multi_audio_score_is_indistinguishable_from_guessing():
    """22% при четырёх вариантах — это ниже случайных 25%."""
    assert is_above_chance(0.22, 4) is False


def test_a_real_result_beats_chance():
    assert is_above_chance(0.60, 4) is True


def test_exactly_at_chance_is_not_above_chance():
    assert is_above_chance(0.25, 4, margin=0.0) is False


def test_chance_level_depends_on_the_number_of_choices():
    """40% — успех на 10 вариантах и провал на двух."""
    assert is_above_chance(0.40, 10) is True
    assert is_above_chance(0.40, 2) is False


def test_is_above_chance_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        is_above_chance(0.5, 1)
    with pytest.raises(ValueError):
        is_above_chance(1.5, 4)


# --------------------------------------------------------- gate_on_silence
def test_silence_produces_no_answer():
    assert gate_on_silence("dog barks", [0.0, 0.0, 0.0], 0.01) == ""


def test_gate_uses_rms_not_the_plain_mean():
    """Ловушка: среднее у [-1, 1, -1, 1] равно нулю, а запись громкая."""
    assert gate_on_silence("speech", [-1.0, 1.0, -1.0, 1.0], 0.5) == "speech"


def test_exactly_at_the_threshold_is_gated():
    assert gate_on_silence("speech", [0.2, -0.2], 0.2) == ""


def test_gate_on_silence_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        gate_on_silence("x", [], 0.1)
    with pytest.raises(ValueError):
        gate_on_silence("x", [0.1], -1.0)

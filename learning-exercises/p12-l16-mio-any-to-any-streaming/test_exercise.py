"""Тесты к уроку «MIO: any-to-any и потоковая генерация». Правь exercise.py."""

import pytest

from exercise import (
    CURRICULUM,
    TOKENIZERS,
    allocate_vocab,
    curriculum_gap,
    embedding_params,
    latency_trace,
    latency_verdict,
    modality_of,
    residual_vq_tokens,
    route_modality,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Раскладка словаря MIO из урока: четыре модальности плюс служебные теги.
PLAN = [
    ("text", 32000),
    ("image", 4096),
    ("speech", 4096),
    ("music", 8192),
    ("sep", 10),
]
VOCAB_SIZE = 48394  # сумма PLAN

SMALL_PLAN = [("text", 4), ("image", 3), ("speech", 2)]


# ----------------------------------------------------------- allocate_vocab
def test_allocate_vocab_worked_example():
    assert allocate_vocab([("text", 3), ("image", 2)]) == [
        ("text", 0, 3),
        ("image", 3, 5),
    ]


def test_allocated_slots_leave_no_gap_and_no_overlap():
    """Дыра тратит embedding-матрицу, нахлёст молча ломает смысл id."""
    slots = allocate_vocab(PLAN)
    assert slots[0][1] == 0
    for (_, _, end), (_, start, _) in zip(slots, slots[1:]):
        assert end == start


def test_allocated_vocab_size_is_the_sum_of_the_plan():
    slots = allocate_vocab(PLAN)
    assert slots[-1][2] == VOCAB_SIZE == sum(size for _, size in PLAN)


def test_allocate_vocab_rejects_an_empty_slot():
    with pytest.raises(ValueError):
        allocate_vocab([("text", 3), ("image", 0)])


def test_allocate_vocab_rejects_a_duplicated_slot_name():
    with pytest.raises(ValueError):
        allocate_vocab([("text", 3), ("text", 2)])


# -------------------------------------------------------------- modality_of
def test_modality_of_the_first_id_of_a_slot():
    assert modality_of(allocate_vocab(PLAN), 0) == "text"


def test_slot_boundary_belongs_to_the_next_modality():
    """Диапазон половинчатый: 32000 это уже картинка, а не последний текст."""
    slots = allocate_vocab(PLAN)
    assert modality_of(slots, 31999) == "text"
    assert modality_of(slots, 32000) == "image"


def test_every_id_belongs_to_exactly_one_modality():
    slots = allocate_vocab(SMALL_PLAN)
    total = sum(size for _, size in SMALL_PLAN)
    owners = [modality_of(slots, i) for i in range(total)]
    assert owners == ["text"] * 4 + ["image"] * 3 + ["speech"] * 2


def test_modality_of_rejects_an_id_past_the_vocabulary():
    with pytest.raises(ValueError):
        modality_of(allocate_vocab(PLAN), VOCAB_SIZE)


# ----------------------------------------------------------- route_modality
def test_route_modality_sends_images_to_the_image_tokenizer():
    assert route_modality("image") == "SEED-Tokenizer"


def test_voice_and_speech_share_one_tokenizer():
    """Разные слова продуктовой команды, один residual-VQ под капотом."""
    assert route_modality("voice") == route_modality("speech")


def test_every_declared_kind_has_a_tokenizer():
    assert all(route_modality(kind) == TOKENIZERS[kind] for kind in TOKENIZERS)


def test_route_modality_rejects_an_unknown_kind():
    """Fallback на BPE дал бы правдоподобный мусор вместо явной ошибки."""
    with pytest.raises(ValueError):
        route_modality("video")


# --------------------------------------------------------- embedding_params
def test_embedding_params_worked_example():
    assert embedding_params(VOCAB_SIZE, 4096) == 198221824


def test_untying_the_output_projection_doubles_the_cost():
    tied = embedding_params(VOCAB_SIZE, 4096, tied=True)
    untied = embedding_params(VOCAB_SIZE, 4096, tied=False)
    assert untied == 2 * tied


def test_embedding_cost_is_linear_in_every_new_modality():
    """Добавили 8k музыки — цена выросла ровно на 8k * hidden, не больше."""
    before = embedding_params(40202, 4096)
    after = embedding_params(40202 + 8192, 4096)
    assert after - before == 8192 * 4096


def test_embedding_params_rejects_an_empty_vocabulary():
    with pytest.raises(ValueError):
        embedding_params(0, 4096)


# ------------------------------------------------------- residual_vq_tokens
def test_residual_vq_worked_example():
    assert residual_vq_tokens(1.0, 20, 8) == 160


def test_base_layer_alone_is_one_token_per_frame():
    assert residual_vq_tokens(1.0, 20, 1) == 20


def test_each_extra_codebook_costs_a_full_extra_stream():
    """Восемь уровней это восьмикратный объём — потому их и декодируют параллельно."""
    one = residual_vq_tokens(2.5, 20, 1)
    eight = residual_vq_tokens(2.5, 20, 8)
    assert eight == 8 * one


def test_an_incomplete_frame_produces_no_tokens():
    assert residual_vq_tokens(0.04, 20, 8) == 0


# ------------------------------------------------------------ latency_trace
def test_latency_trace_accumulates():
    assert latency_trace([("mic", 40.0), ("prefill", 80.0)]) == [
        ("mic", 40.0, 40.0),
        ("prefill", 80.0, 120.0),
    ]


def test_the_last_cumulative_value_is_the_ttfab():
    stages = [("mic", 40.0), ("prefill", 80.0), ("first token", 40.0),
              ("residual VQ", 30.0), ("vocoder", 80.0)]
    trace = latency_trace(stages)
    assert trace[-1][2] == APPROX(sum(ms for _, ms in stages))


def test_latency_trace_never_goes_backwards():
    stages = [("a", 10.0), ("b", 0.0), ("c", 5.0)]
    cumulative = [row[2] for row in latency_trace(stages)]
    assert all(a <= b for a, b in zip(cumulative, cumulative[1:]))


def test_latency_trace_rejects_a_negative_stage():
    with pytest.raises(ValueError):
        latency_trace([("mic", 40.0), ("magic", -10.0)])


# ---------------------------------------------------------- latency_verdict
def test_three_hundred_milliseconds_feels_conversational():
    assert latency_verdict(300.0) == "conversational"


def test_the_five_hundred_millisecond_boundary_is_exclusive():
    assert latency_verdict(499.0) == "conversational"
    assert latency_verdict(500.0) == "acceptable"


def test_past_eight_hundred_the_user_talks_over_the_model():
    assert latency_verdict(799.0) == "acceptable"
    assert latency_verdict(800.0) == "sluggish"


def test_latency_verdict_rejects_negative_time():
    with pytest.raises(ValueError):
        latency_verdict(-1.0)


# ----------------------------------------------------------- curriculum_gap
def test_a_full_curriculum_leaves_no_gap():
    assert curriculum_gap([name for name, _ in CURRICULUM]) == ()


def test_skipping_the_interleaved_stage_costs_cross_modal_context():
    gap = curriculum_gap(["alignment", "speech", "sft"])
    assert gap == ("кросс-модальный контекст",)


def test_gaps_are_reported_in_training_order_not_input_order():
    gap = curriculum_gap(["sft", "alignment"])
    expected = tuple(effect for name, effect in CURRICULUM if name not in ("sft", "alignment"))
    assert gap == expected


def test_curriculum_gap_rejects_a_misspelled_stage():
    """Опечатка иначе выглядит как честно пропущенная стадия."""
    with pytest.raises(ValueError):
        curriculum_gap(["alignement"])

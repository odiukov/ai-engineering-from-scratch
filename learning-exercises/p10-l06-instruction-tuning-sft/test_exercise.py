"""Тесты к уроку «Instruction tuning (SFT)». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    SPECIAL_TOKENS,
    create_loss_mask,
    d_masked_cross_entropy,
    dataset_quality,
    masked_cross_entropy,
    mix_pretraining_data,
    shift_for_training,
    tokenize_conversation,
    tokenize_instruction_pair,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

DATASET = [
    {"instruction": "What is the capital of France?", "response": "The capital is Paris."},
    {"instruction": "Explain gravity in one sentence.", "response": "Gravity attracts mass."},
    {"instruction": "Name three programming languages.", "response": "Python, Rust, TypeScript."},
    {"instruction": "What year did World War II end?", "response": "It ended in 1945."},
]

RAW = [
    "The transformer processes sequences through self-attention.",
    "Residual connections stabilise very deep networks.",
]


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу."""
    return [x for row in M for x in row]


# ------------------------------------------------ tokenize_instruction_pair
def test_tokenize_wraps_instruction_and_response_in_markers():
    assert tokenize_instruction_pair("a", "b") == [256, 97, 257, 258, 98, 259]


def test_tokenize_ends_every_example_with_eos():
    """Без EOS модель не научится замолкать и будет писать до лимита."""
    tokens = tokenize_instruction_pair("hello", "world")
    assert tokens[-1] == SPECIAL_TOKENS["EOS"]


def test_tokenize_keeps_special_ids_out_of_the_byte_range():
    """Урок зажимает байты в 0..252 — так текст перестаёт восстанавливаться."""
    tokens = tokenize_instruction_pair("ыъ", "яё")
    assert all(t < 256 for t in tokens if t not in SPECIAL_TOKENS.values())
    body = [t for t in tokens if t < 256]
    assert bytes(body).decode("utf-8") == "ыъяё"


def test_tokenize_length_is_content_bytes_plus_four_markers():
    tokens = tokenize_instruction_pair("abc", "de")
    assert len(tokens) == 3 + 2 + 4


# ------------------------------------------------------------ create_loss_mask
def test_loss_mask_covers_the_response_only():
    assert create_loss_mask([256, 97, 257, 258, 98, 259]) == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_loss_mask_excludes_the_response_marker_itself():
    tokens = tokenize_instruction_pair("a", "b")
    mask = create_loss_mask(tokens)
    assert mask[tokens.index(SPECIAL_TOKENS["RESP_START"])] == 0.0


def test_loss_mask_includes_the_eos_token():
    """Ставить точку в конце — тоже поведение, которому надо научить."""
    tokens = tokenize_instruction_pair("a", "b")
    mask = create_loss_mask(tokens)
    assert mask[-1] == 1.0


def test_loss_mask_sums_to_the_number_of_response_tokens():
    example = DATASET[0]
    tokens = tokenize_instruction_pair(example["instruction"], example["response"])
    expected = len(example["response"].encode("utf-8")) + 1  # +1 на EOS
    assert sum(create_loss_mask(tokens)) == APPROX(expected)


# ------------------------------------------------------- tokenize_conversation
def test_single_exchange_matches_the_instruction_pair_form():
    turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert tokenize_conversation(turns) == tokenize_instruction_pair("a", "b")


def test_conversation_masks_every_assistant_turn_not_just_the_last():
    """Иначе две трети размеченного диалога пропадут впустую."""
    turns = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    mask = create_loss_mask(tokenize_conversation(turns))
    assert sum(mask) == APPROX(3 * (2 + 1))


def test_conversation_mask_switches_off_on_the_next_user_turn():
    turns = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "z"},
    ]
    tokens = tokenize_conversation(turns)
    mask = create_loss_mask(tokens)
    assert mask[-2:] == [0.0, 0.0]  # байт "z" и INST_END уже вне ответа


def test_conversation_of_nothing_is_nothing():
    assert tokenize_conversation([]) == []
    assert create_loss_mask([]) == []


# --------------------------------------------------------- shift_for_training
def test_shift_moves_inputs_and_targets_apart_by_one():
    assert shift_for_training([1, 2, 3], [0.0, 1.0, 1.0]) == ([1, 2], [2, 3], [1.0, 1.0])


def test_shift_keeps_all_three_pieces_the_same_length():
    tokens = tokenize_instruction_pair("hello", "world")
    inputs, targets, mask = shift_for_training(tokens, create_loss_mask(tokens))
    assert len(inputs) == len(targets) == len(mask) == len(tokens) - 1


def test_shifted_mask_follows_the_targets_not_the_inputs():
    """Сдвинешь маску вместе со входами — потеря съедет на токен."""
    tokens = tokenize_instruction_pair("a", "bc")
    mask = create_loss_mask(tokens)
    _, targets, target_mask = shift_for_training(tokens, mask)
    scored = [t for t, m in zip(targets, target_mask) if m]
    assert scored == list("bc".encode("utf-8")) + [SPECIAL_TOKENS["EOS"]]


# ------------------------------------------------------ masked_cross_entropy
def test_masked_loss_counts_only_the_marked_position():
    assert masked_cross_entropy(
        [[0.0, 0.0], [0.0, 0.0]], [0, 1], [0.0, 1.0]
    ) == pytest.approx(math.log(2))


def test_masked_loss_with_a_full_mask_is_plain_cross_entropy():
    rows = [[0.5, -1.2, 2.0], [0.1, 0.3, -0.7]]
    targets = [2, 0]
    manual = 0.0
    for row, target in zip(rows, targets):
        manual += math.log(sum(math.exp(s) for s in row)) - row[target]
    assert masked_cross_entropy(rows, targets, [1.0, 1.0]) == pytest.approx(manual / 2)


def test_masked_loss_of_an_all_zero_mask_is_zero():
    assert masked_cross_entropy([[0.0, 0.0]], [0], [0.0]) == APPROX(0.0)


def test_logits_on_masked_positions_cannot_change_the_loss():
    """Инструкция участвует в forward, но не в потере — вот проверка этого."""
    mask = [0.0, 1.0]
    quiet = masked_cross_entropy([[0.0, 0.0], [1.0, 2.0]], [0, 1], mask)
    loud = masked_cross_entropy([[99.0, -99.0], [1.0, 2.0]], [0, 1], mask)
    assert quiet == APPROX(loud)


def test_long_instruction_does_not_dilute_the_response_loss():
    """Делим на сумму маски, а не на длину: иначе один и тот же ответ стоит по-разному."""
    response_row = [0.4, 0.1, -0.3]
    short = masked_cross_entropy([[0.0] * 3, response_row], [0, 2], [0.0, 1.0])
    long = masked_cross_entropy([[0.0] * 3] * 5 + [response_row], [0] * 5 + [2], [0.0] * 5 + [1.0])
    assert short == APPROX(long)


def test_masked_loss_survives_huge_logits():
    assert masked_cross_entropy([[1000.0, 999.0]], [0], [1.0]) == pytest.approx(
        math.log(1 + math.e ** -1)
    )


# ---------------------------------------------------- d_masked_cross_entropy
def test_masked_rows_of_the_gradient_are_exactly_zero():
    grad = d_masked_cross_entropy([[0.0, 0.0], [0.0, 0.0]], [0, 1], [0.0, 1.0])
    assert flat(grad) == APPROX([0.0, 0.0, 0.5, -0.5])


def test_live_gradient_rows_sum_to_zero():
    grad = d_masked_cross_entropy([[0.5, -1.2, 2.0], [0.1, 0.3, -0.7]], [2, 0], [1.0, 1.0])
    for row in grad:
        assert sum(row) == pytest.approx(0.0, abs=1e-12)


def test_masked_gradient_matches_the_numeric_gradient():
    """Сошедшаяся формула и правильная формула — разные вещи."""
    logits = [[0.5, -1.2, 2.0], [0.1, 0.3, -0.7], [1.5, 0.2, -0.4]]
    targets = [2, 0, 1]
    mask = [0.0, 1.0, 1.0]
    analytic = d_masked_cross_entropy(logits, targets, mask)
    h = 1e-5
    for i in range(len(logits)):
        for j in range(len(logits[0])):
            up = [row[:] for row in logits]
            down = [row[:] for row in logits]
            up[i][j] += h
            down[i][j] -= h
            numeric = (
                masked_cross_entropy(up, targets, mask)
                - masked_cross_entropy(down, targets, mask)
            ) / (2 * h)
            assert analytic[i][j] == pytest.approx(numeric, abs=1e-6)


# -------------------------------------------------------------- dataset_quality
def test_quality_counts_instruction_and_response_bytes():
    stats = dataset_quality({"instruction": "abc", "response": "de"})
    assert stats["instruction_tokens"] == 3
    assert stats["response_tokens"] == 2
    assert stats["response_ratio"] == APPROX(0.4)


def test_quality_diversity_falls_on_a_repetitive_answer():
    assert dataset_quality({"instruction": "ab", "response": "cd"})["diversity"] == APPROX(1.0)
    assert dataset_quality({"instruction": "ab", "response": "aaaa"})["diversity"] == APPROX(0.25)


def test_quality_of_an_empty_response_does_not_divide_by_zero():
    stats = dataset_quality({"instruction": "abc", "response": ""})
    assert stats["response_tokens"] == 0
    assert stats["diversity"] == APPROX(0.0)


def test_quality_filter_drops_stub_answers_and_keeps_real_ones():
    """Типовой порог: response_tokens >= 10 и diversity >= 0.3."""
    dataset = DATASET + [{"instruction": "Is it ok?", "response": "Yes."}]
    kept = [
        e
        for e in dataset
        if dataset_quality(e)["response_tokens"] >= 10
        and dataset_quality(e)["diversity"] >= 0.3
    ]
    assert kept == DATASET


# ----------------------------------------------------- mix_pretraining_data
def test_mixing_keeps_the_dataset_size():
    mixed = mix_pretraining_data(DATASET, RAW, 0.25, random.Random(0))
    assert len(mixed) == len(DATASET)


def test_zero_fraction_leaves_only_masked_instruction_examples():
    mixed = mix_pretraining_data(DATASET, RAW, 0.0, random.Random(0))
    assert all(0.0 in mask for _, mask in mixed)


def test_raw_text_comes_in_completely_unmasked():
    """Сырой текст — обычное предобучение, маскировать там нечего."""
    mixed = mix_pretraining_data(DATASET, RAW, 0.5, random.Random(1))
    unmasked = [tokens for tokens, mask in mixed if all(m == 1.0 for m in mask)]
    assert len(unmasked) == 2


def test_mixing_is_reproducible_for_a_given_seed():
    first = mix_pretraining_data(DATASET, RAW, 0.5, random.Random(7))
    second = mix_pretraining_data(DATASET, RAW, 0.5, random.Random(7))
    assert first == second


def test_a_different_seed_gives_a_different_shuffle():
    runs = {
        tuple(tuple(tokens) for tokens, _ in mix_pretraining_data(
            DATASET, RAW, 0.5, random.Random(seed)
        ))
        for seed in range(8)
    }
    assert len(runs) > 1

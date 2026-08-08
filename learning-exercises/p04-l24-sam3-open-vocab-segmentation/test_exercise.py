"""Тесты к уроку «SAM 3 и open-vocabulary сегментация». Правь exercise.py."""

import pytest

from exercise import (
    mask_area,
    mask_iou,
    mask_to_box,
    merge_concept_results,
    presence_gate,
    rle_decode,
    rle_encode,
    split_concepts,
)


def flat(nested):
    """Развернуть вложенные списки в плоский — pytest.approx не умеет вложенность."""
    return [value for row in nested for value in row]


# ---------------------------------------------------------- split_concepts
def test_split_concepts_on_a_comma_and_the_word_and():
    assert split_concepts("cats, dogs and balloons") == ["cats", "dogs", "balloons"]


def test_split_concepts_keeps_a_multiword_noun_phrase_whole():
    assert split_concepts("yellow school bus") == ["yellow school bus"]


def test_split_concepts_does_not_cut_inside_a_word():
    """"sandwich" содержит "and", но это не разделитель."""
    assert split_concepts("sandwich") == ["sandwich"]
    assert split_concepts("orange") == ["orange"]


def test_split_concepts_handles_semicolon_and_ampersand():
    assert split_concepts("mug; spoon & plate") == ["mug", "spoon", "plate"]


def test_split_concepts_drops_empty_pieces():
    assert split_concepts("cat,,  , dog") == ["cat", "dog"]


# --------------------------------------------------------------- rle_encode
def test_rle_encode_splits_a_row_into_runs():
    assert rle_encode([[0, 0, 1]]) == "0x2;1x1"


def test_rle_encode_merges_runs_across_row_boundaries():
    """Обход построчный: маска 2x2 из единиц — это один run длины 4."""
    assert rle_encode([[1, 1], [1, 1]]) == "1x4"


def test_rle_encode_serializes_an_empty_mask_as_an_empty_string():
    assert rle_encode([]) == ""
    assert rle_encode([[]]) == ""


def test_rle_encode_rejects_non_binary_values():
    with pytest.raises(ValueError):
        rle_encode([[0, 255]])


# --------------------------------------------------------------- rle_decode
def test_rle_decode_restores_a_row():
    assert flat(rle_decode("0x2;1x1", 3)) == [0, 0, 1]


def test_rle_decode_is_the_exact_inverse_of_encode():
    mask = [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]]
    assert flat(rle_decode(rle_encode(mask), 4)) == flat(mask)


def test_rle_decode_rejects_a_width_that_does_not_divide_the_mask():
    with pytest.raises(ValueError):
        rle_decode("1x3", 2)


# ----------------------------------------------------------------- mask_area
def test_mask_area_counts_only_the_ones():
    assert mask_area("0x2;1x1") == 1


def test_mask_area_sums_every_positive_run():
    assert mask_area("1x4;0x10;1x6") == 10


def test_mask_area_agrees_with_the_decoded_mask():
    """Быстрый путь по RLE обязан совпасть с честным подсчётом по пикселям."""
    mask = [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]]
    rle = rle_encode(mask)
    assert mask_area(rle) == sum(flat(mask))


# --------------------------------------------------------------- mask_to_box
def test_mask_to_box_of_a_single_pixel():
    assert mask_to_box([[0, 0], [0, 1]]) == (1, 1, 1, 1)


def test_mask_to_box_wraps_a_rectangle_tightly():
    mask = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    assert mask_to_box(mask) == (1, 1, 2, 2)


def test_mask_to_box_covers_scattered_pixels():
    """Бокс габаритный: он покрывает и дырки между кусками маски."""
    mask = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ]
    assert mask_to_box(mask) == (0, 0, 2, 2)


def test_mask_to_box_of_an_empty_mask_is_none():
    assert mask_to_box([[0, 0], [0, 0]]) is None


# ------------------------------------------------------------------ mask_iou
def test_mask_iou_of_identical_masks_is_one():
    mask = [[1, 0], [1, 1]]
    assert mask_iou(mask, mask) == pytest.approx(1.0)


def test_mask_iou_of_disjoint_masks_is_zero():
    assert mask_iou([[1, 0]], [[0, 1]]) == pytest.approx(0.0)


def test_mask_iou_of_a_half_overlap():
    a = [[1, 1, 0, 0]]
    b = [[0, 1, 1, 0]]
    assert mask_iou(a, b) == pytest.approx(1 / 3)


def test_mask_iou_is_symmetric():
    a = [[1, 1, 0], [0, 1, 0]]
    b = [[0, 1, 1], [0, 1, 1]]
    assert mask_iou(a, b) == pytest.approx(mask_iou(b, a))


def test_mask_iou_of_two_empty_masks_is_zero_not_one():
    assert mask_iou([[0, 0]], [[0, 0]]) == pytest.approx(0.0)


def test_mask_iou_rejects_masks_of_different_size():
    with pytest.raises(ValueError):
        mask_iou([[1, 0]], [[1, 0, 0]])


# ------------------------------------------------------------- presence_gate
def test_presence_gate_keeps_detections_when_the_concept_is_present():
    detections = [{"score": 0.99}, {"score": 0.6}]
    assert presence_gate(detections, 0.9) == detections


def test_presence_gate_drops_confident_detections_when_the_concept_is_absent():
    """Главное свойство presence head: score 0.99 не спасает отсутствующий концепт."""
    assert presence_gate([{"score": 0.99}], 0.1) == []


def test_presence_gate_uses_the_threshold_it_was_given():
    detections = [{"score": 0.5}]
    assert presence_gate(detections, 0.4, threshold=0.3) == detections
    assert presence_gate(detections, 0.4, threshold=0.7) == []


def test_presence_gate_does_not_mutate_the_input_list():
    detections = [{"score": 0.99}]
    presence_gate(detections, 0.9).clear()
    assert len(detections) == 1


# ------------------------------------------------------ merge_concept_results
def test_merge_numbers_instances_by_descending_score():
    merged = merge_concept_results({"cat": [{"score": 0.4}, {"score": 0.9}]})
    assert [(d["score"], d["instance_id"]) for d in merged] == [(0.9, 0), (0.4, 1)]


def test_merge_restarts_instance_ids_for_every_concept():
    """id отвечает на вопрос «какая это по счёту кошка», а не «какая это строка»."""
    merged = merge_concept_results(
        {"cat": [{"score": 0.9}, {"score": 0.8}], "dog": [{"score": 0.7}]}
    )
    ids = {(d["concept"], d["instance_id"]) for d in merged}
    assert ids == {("cat", 0), ("cat", 1), ("dog", 0)}


def test_merge_sorts_the_whole_list_by_score():
    merged = merge_concept_results(
        {"cat": [{"score": 0.4}], "dog": [{"score": 0.95}, {"score": 0.6}]}
    )
    scores = [d["score"] for d in merged]
    assert scores == sorted(scores, reverse=True)


def test_merge_keeps_the_original_fields():
    merged = merge_concept_results({"mug": [{"score": 0.5, "mask_rle": "1x4"}]})
    assert merged[0]["mask_rle"] == "1x4"
    assert merged[0]["concept"] == "mug"


def test_merge_of_nothing_is_an_empty_list():
    assert merge_concept_results({}) == []


def test_merge_does_not_mutate_the_input_detections():
    detection = {"score": 0.5}
    merge_concept_results({"cat": [detection]})
    assert detection == {"score": 0.5}

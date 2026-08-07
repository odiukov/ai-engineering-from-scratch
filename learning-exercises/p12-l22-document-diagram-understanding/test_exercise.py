"""Тесты к уроку «Понимание документов и диаграмм». Правь exercise.py."""

import pytest

from exercise import (
    STACKS,
    anyres_tokens,
    donut_parse,
    donut_serialize,
    iou,
    layoutlm_input,
    normalize_bbox,
    pick_stack,
    reading_order,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Кусок счёта из урока: заголовок, две колонки таблицы, итог внизу справа.
PAGE = [
    ("INVOICE", (100, 50, 300, 80)),
    ("ACME", (100, 100, 250, 130)),
    ("Item", (100, 200, 200, 230)),
    ("Price", (400, 200, 500, 230)),
    ("Widget", (100, 240, 250, 270)),
    ("$120.00", (400, 240, 500, 270)),
]


# ----------------------------------------------------------- normalize_bbox
def test_normalize_bbox_on_a_unit_page_is_the_identity():
    assert normalize_bbox((100, 50, 300, 80), 1000, 1000) == (100, 50, 300, 80)


def test_normalize_bbox_divides_by_the_page_size():
    assert normalize_bbox((100, 50, 300, 80), 2000, 1000) == (50, 50, 150, 80)


def test_normalize_bbox_is_resolution_independent():
    """Тот же лист при 150 и 300 DPI обязан дать один и тот же поток bbox."""
    low = normalize_bbox((100, 50, 300, 80), 1000, 1400)
    high = normalize_bbox((200, 100, 600, 160), 2000, 2800)
    assert low == high


def test_normalize_bbox_clamps_boxes_that_spill_off_the_page():
    """Эмбеддинга на позицию 1004 в LayoutLM нет, а OCR такие рамки выдаёт."""
    assert normalize_bbox((-20, -5, 1100, 1050), 1000, 1000) == (0, 0, 1000, 1000)


# ------------------------------------------------------------------- iou
def test_iou_of_a_box_with_itself_is_one():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == APPROX(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    """Минус на минус даёт плюс: без зануления по каждой оси тут вылезет ложный IoU."""
    assert iou((0, 0, 1, 1), (5, 5, 6, 6)) == APPROX(0.0)


def test_iou_of_half_overlapping_boxes():
    assert iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_iou_is_symmetric():
    a, b = (10, 10, 40, 30), (20, 15, 60, 50)
    assert iou(a, b) == APPROX(iou(b, a))


# ------------------------------------------------------------ reading_order
def test_reading_order_sorts_a_line_left_to_right():
    tokens = [("b", (300, 10, 350, 40)), ("a", (100, 10, 150, 40))]
    assert [t[0] for t in reading_order(tokens, 50)] == ["a", "b"]


def test_reading_order_sorts_lines_top_to_bottom():
    tokens = [("second", (100, 200, 150, 230)), ("first", (100, 50, 150, 80))]
    assert [t[0] for t in reading_order(tokens, 50)] == ["first", "second"]


def test_reading_order_recovers_the_invoice_layout():
    """Строки таблицы читаются целиком, а не по колонкам."""
    assert [t[0] for t in reading_order(PAGE, 40)] == [
        "INVOICE", "ACME", "Item", "Price", "Widget", "$120.00",
    ]


def test_baseline_jitter_does_not_split_a_line():
    """Соседние слова стоят на разных пикселях — полоса высотой line_height это гасит."""
    tokens = [("left", (100, 204, 200, 234)), ("right", (400, 200, 500, 230))]
    assert [t[0] for t in reading_order(tokens, 50)] == ["left", "right"]
    # без полосы (сортировка почти по чистому y0) правое слово уезжает вперёд левого
    assert [t[0] for t in reading_order(tokens, 1)] == ["right", "left"]


def test_reading_order_does_not_mutate_the_input():
    tokens = list(PAGE)
    reading_order(tokens, 50)
    assert tokens == PAGE


# ----------------------------------------------------------- layoutlm_input
def test_layoutlm_input_has_all_three_streams():
    data = layoutlm_input([("Total", (400, 400, 500, 430))], 1000, 1000)
    assert data == {
        "text": ["Total"],
        "bbox": [(400, 400, 500, 430)],
        "n_patches": 256,
    }


def test_layoutlm_text_and_bbox_streams_stay_aligned():
    data = layoutlm_input(PAGE, 1000, 1000)
    assert len(data["text"]) == len(data["bbox"]) == len(PAGE)


def test_layoutlm_bboxes_are_normalized_not_raw_pixels():
    data = layoutlm_input(PAGE, 2000, 2000)
    assert all(0 <= v <= 1000 for box in data["bbox"] for v in box)
    assert data["bbox"][0] == (50, 25, 150, 40)


def test_layoutlm_patch_stream_exists_even_without_ocr_text():
    """Патчи от текста не зависят: пустая страница — это всё ещё картинка."""
    data = layoutlm_input([], 1000, 1000, patch_grid=(8, 8))
    assert data["text"] == []
    assert data["n_patches"] == 64


# ---------------------------------------------------------- donut_serialize
def test_donut_serialize_wraps_a_field_in_its_tag():
    assert donut_serialize({"total": "1245"}) == "<s_total>1245</s_total>"


def test_donut_serialize_of_an_empty_record_is_an_empty_string():
    assert donut_serialize({}) == ""


def test_donut_serialize_keeps_field_order():
    record = {"vendor": "ACME", "total": "1245", "currency": "USD"}
    assert donut_serialize(record) == (
        "<s_vendor>ACME</s_vendor><s_total>1245</s_total><s_currency>USD</s_currency>"
    )


# -------------------------------------------------------------- donut_parse
def test_donut_parse_reads_a_single_field():
    assert donut_parse("<s_total>1245</s_total>") == {"total": "1245"}


def test_donut_parse_of_an_empty_string_is_an_empty_record():
    assert donut_parse("") == {}


def test_donut_round_trip_preserves_the_record():
    record = {"vendor": "ACME Co.", "total": "1,245.00", "date": "2026-01-05"}
    assert donut_parse(donut_serialize(record)) == record


def test_donut_parse_rejects_a_truncated_generation():
    """Обрыв генерации — это галлюцинация модели, а не половина полей."""
    with pytest.raises(ValueError):
        donut_parse("<s_vendor>ACME</s_vendor><s_total>1245")


# ------------------------------------------------------------ anyres_tokens
def test_anyres_one_tile_page_pays_for_the_thumbnail_too():
    assert anyres_tokens(336, 336) == 1152


def test_anyres_rounds_tiles_up():
    """Один лишний пиксель по ширине стоит целого тайла."""
    assert anyres_tokens(337, 336) == 1728


def test_anyres_quadruples_when_both_sides_double():
    assert anyres_tokens(672, 672) == 4 * 576 + 576


def test_anyres_without_a_thumbnail_costs_one_tile_less():
    assert anyres_tokens(672, 672, thumbnail=False) == anyres_tokens(672, 672) - 576


def test_anyres_shows_why_a_scanned_a4_page_is_expensive():
    """A4 при 300 DPI — восемь на одиннадцать тайлов, десятки тысяч токенов."""
    assert anyres_tokens(2500, 3500) > 50000


# ----------------------------------------------------------------- pick_stack
def test_regulated_projects_keep_the_auditable_ocr_pipeline():
    assert pick_stack({"regulated": True}) == "ocr-pipeline+crosscheck"


def test_math_heavy_corpora_go_to_nougat():
    assert pick_stack({"math": True}) == "nougat+vlm"


def test_printed_pages_at_scale_go_to_the_cheap_pipeline():
    assert pick_stack({"pages_per_day": 10_000_000}) == "ocr-pipeline"


def test_scale_does_not_override_handwriting():
    """Приоритет качества выше цены: рукопись OCR-пайплайн не вытянет ни за какие деньги."""
    assert pick_stack({"pages_per_day": 10_000_000, "handwriting": True}) == "vlm-native"


def test_pick_stack_always_returns_a_known_stack():
    profiles = [
        {}, {"regulated": True}, {"math": True}, {"handwriting": True},
        {"pages_per_day": 5_000_000}, {"pages_per_day": 10},
    ]
    assert all(pick_stack(p) in STACKS for p in profiles)


def test_pick_stack_rejects_a_misspelled_profile_key():
    """"regulatd" молча выбрал бы vlm-native и уронил аудит."""
    with pytest.raises(ValueError):
        pick_stack({"regulatd": True})

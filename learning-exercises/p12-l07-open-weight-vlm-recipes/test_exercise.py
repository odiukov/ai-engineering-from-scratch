"""Тесты к уроку «Рецепты open-weight VLM: что реально влияет». Правь exercise.py."""

import pytest

from exercise import (
    axis_delta,
    controlled_pairs,
    expected_score,
    explained_variance,
    parse_ablation_row,
    pick_recipe,
    rank_axes_by_impact,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Мини-версия таблицы MM1 3.2: две оси по два значения, всё остальное
# заморожено. Ровно та форма, на которой ablation вообще имеет смысл.
TABLE = [
    {"encoder": "clip", "connector": "mlp", "tokens": 576, "mmmu": 38.0},
    {"encoder": "siglip", "connector": "mlp", "tokens": 576, "mmmu": 41.0},
    {"encoder": "clip", "connector": "qformer", "tokens": 576, "mmmu": 38.4},
    {"encoder": "siglip", "connector": "qformer", "tokens": 576, "mmmu": 41.6},
]

# Таблица с разным бюджетом токенов — для выбора рецепта под ограничение.
BUDGET_TABLE = [
    {"encoder": "siglip", "tokens": 2304, "data": "pixmo", "mmmu": 45.0},
    {"encoder": "siglip", "tokens": 576, "data": "pixmo", "mmmu": 41.0},
    {"encoder": "clip", "tokens": 576, "data": "sharegpt4v", "mmmu": 39.0},
]


# ------------------------------------------------------ parse_ablation_row
def test_parse_ablation_row_splits_axes_from_metrics():
    row = parse_ablation_row("encoder=siglip;connector=mlp | mmmu=41.2;docvqa=88.0")
    assert row == {
        "encoder": "siglip",
        "connector": "mlp",
        "mmmu": 41.2,
        "docvqa": 88.0,
    }


def test_parse_ablation_row_keeps_whole_numbers_as_int():
    """Бюджет токенов обязан быть числом: "1024" < "576" как строки."""
    row = parse_ablation_row("tokens=576 | mmmu=41.2")
    assert row["tokens"] == 576
    assert isinstance(row["tokens"], int)


def test_parse_ablation_row_tolerates_spaces_around_fields():
    assert parse_ablation_row("  encoder = clip  |  mmmu = 38.0 ") == {
        "encoder": "clip",
        "mmmu": 38.0,
    }


def test_parse_ablation_row_rejects_line_without_separator():
    with pytest.raises(ValueError):
        parse_ablation_row("encoder=clip;mmmu=38.0")


def test_parse_ablation_row_rejects_field_without_equals():
    with pytest.raises(ValueError):
        parse_ablation_row("encoder=clip;mlp | mmmu=38.0")


# --------------------------------------------------------- controlled_pairs
def test_controlled_pairs_finds_single_knob_changes():
    assert controlled_pairs(TABLE, "encoder") == [(0, 1), (2, 3)]


def test_controlled_pairs_skips_rows_differing_in_two_axes():
    """(0, 3) отличается и энкодером, и коннектором — приписать некому."""
    assert (0, 3) not in controlled_pairs(TABLE, "encoder")
    assert (0, 3) not in controlled_pairs(TABLE, "connector")


def test_controlled_pairs_ignores_metric_columns():
    """Метрики отличаются всегда — это не повод отбраковать пару."""
    rows = [
        {"encoder": "clip", "mmmu": 38.0, "docvqa": 71.0},
        {"encoder": "siglip", "mmmu": 41.0, "docvqa": 88.0},
    ]
    assert controlled_pairs(rows, "encoder") == [(0, 1)]


def test_controlled_pairs_is_empty_when_axis_never_changes():
    assert controlled_pairs(TABLE, "tokens") == []


# --------------------------------------------------------------- axis_delta
def test_axis_delta_averages_controlled_comparisons():
    assert axis_delta(TABLE, "encoder", "clip", "siglip", "mmmu") == APPROX(3.1)


def test_axis_delta_is_antisymmetric():
    forward = axis_delta(TABLE, "encoder", "clip", "siglip", "mmmu")
    backward = axis_delta(TABLE, "encoder", "siglip", "clip", "mmmu")
    assert forward == APPROX(-backward)


def test_axis_delta_is_none_when_table_is_confounded():
    """Две ручки крутили одновременно — честный ответ «не измеряли»."""
    confounded = [
        {"encoder": "clip", "connector": "mlp", "mmmu": 38.0},
        {"encoder": "siglip", "connector": "qformer", "mmmu": 41.6},
    ]
    assert axis_delta(confounded, "encoder", "clip", "siglip", "mmmu") is None


def test_encoder_delta_dwarfs_connector_delta():
    """Главный результат MM1/Idefics2: энкодер решает, коннектор — почти нет."""
    enc = axis_delta(TABLE, "encoder", "clip", "siglip", "mmmu")
    con = axis_delta(TABLE, "connector", "mlp", "qformer", "mmmu")
    assert enc > 5 * con


# -------------------------------------------------------- explained_variance
def test_explained_variance_is_one_when_axis_determines_metric():
    rows = [{"encoder": "a", "mmmu": 1.0}, {"encoder": "b", "mmmu": 3.0}]
    assert explained_variance(rows, "encoder", "mmmu") == APPROX(1.0)


def test_explained_variance_is_zero_for_a_constant_metric():
    rows = [{"encoder": "a", "mmmu": 2.0}, {"encoder": "b", "mmmu": 2.0}]
    assert explained_variance(rows, "encoder", "mmmu") == APPROX(0.0)


def test_explained_variance_does_not_depend_on_metric_scale():
    """Доля дисперсии — безразмерная величина: смена шкалы её не двигает."""
    rescaled = [dict(r, mmmu=r["mmmu"] * 3 + 7) for r in TABLE]
    assert explained_variance(rescaled, "encoder", "mmmu") == APPROX(
        explained_variance(TABLE, "encoder", "mmmu")
    )


def test_explained_variance_stays_within_unit_interval():
    for axis in ("encoder", "connector", "tokens"):
        share = explained_variance(TABLE, axis, "mmmu")
        assert 0.0 <= share <= 1.0


def test_encoder_explains_more_variance_than_connector():
    assert explained_variance(TABLE, "encoder", "mmmu") > explained_variance(
        TABLE, "connector", "mmmu"
    )


# ----------------------------------------------------- rank_axes_by_impact
def test_rank_axes_puts_the_encoder_first():
    assert rank_axes_by_impact(TABLE, "mmmu")[0][0] == "encoder"


def test_rank_axes_is_sorted_descending():
    shares = [share for _, share in rank_axes_by_impact(TABLE, "mmmu")]
    assert shares == sorted(shares, reverse=True)


def test_rank_axes_keeps_input_order_for_ties():
    """Оси, которых в таблице нет, дают 0.0 — порядок обязан быть устойчивым."""
    assert [a for a, _ in rank_axes_by_impact(TABLE, "mmmu", ("llm", "data"))] == [
        "llm",
        "data",
    ]
    assert [a for a, _ in rank_axes_by_impact(TABLE, "mmmu", ("data", "llm"))] == [
        "data",
        "llm",
    ]


# ---------------------------------------------------------- expected_score
DELTAS = {("encoder", "clip", "siglip"): 3.0, ("resolution", 384, 448): 1.5}


def test_expected_score_applies_a_measured_delta():
    assert expected_score(38.0, [("encoder", "clip", "siglip")], DELTAS) == APPROX(41.0)


def test_expected_score_without_swaps_is_the_baseline():
    assert expected_score(38.0, [], DELTAS) == APPROX(38.0)


def test_expected_score_round_trip_returns_to_the_baseline():
    """Обратная замена берётся из той же строки таблицы со знаком минус."""
    swaps = [("encoder", "clip", "siglip"), ("encoder", "siglip", "clip")]
    assert expected_score(38.0, swaps, DELTAS) == APPROX(38.0)


def test_expected_score_is_order_independent():
    a = [("encoder", "clip", "siglip"), ("resolution", 384, 448)]
    assert expected_score(38.0, a, DELTAS) == APPROX(
        expected_score(38.0, list(reversed(a)), DELTAS)
    )


def test_expected_score_refuses_an_unmeasured_swap():
    with pytest.raises(KeyError):
        expected_score(38.0, [("encoder", "clip", "dinov2")], DELTAS)


# -------------------------------------------------------------- pick_recipe
def test_pick_recipe_takes_the_best_row():
    assert pick_recipe(BUDGET_TABLE, "mmmu") == 0


def test_pick_recipe_respects_the_token_budget():
    """Лучший рецепт не влезает в контекст — берём лучший из влезающих."""
    assert pick_recipe(BUDGET_TABLE, "mmmu", max_tokens=576) == 1


def test_pick_recipe_respects_a_required_axis_value():
    assert pick_recipe(BUDGET_TABLE, "mmmu", require={"data": "sharegpt4v"}) == 2


def test_pick_recipe_returns_none_when_nothing_fits():
    assert pick_recipe(BUDGET_TABLE, "mmmu", max_tokens=64) is None


def test_pick_recipe_breaks_ties_by_the_first_index():
    tied = [{"tokens": 576, "mmmu": 40.0}, {"tokens": 576, "mmmu": 40.0}]
    assert pick_recipe(tied, "mmmu") == 0

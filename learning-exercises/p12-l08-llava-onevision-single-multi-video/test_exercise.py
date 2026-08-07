"""Тесты к уроку «LLaVA-OneVision: одна модель на картинку, набор картинок и видео». Правь exercise.py."""

import pytest

from exercise import (
    STAGES,
    allocate_budget,
    best_pool_factor,
    is_valid_curriculum,
    pool_grid,
    pooled_tokens,
    scenario_tokens,
    stage_steps,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(grid):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in grid for x in row]


# ------------------------------------------------------------------ pool_grid
def test_pool_grid_averages_each_block():
    assert flat(pool_grid([[1, 2], [3, 4]], 2)) == APPROX([2.5])


def test_pool_grid_with_factor_one_changes_nothing():
    assert flat(pool_grid([[1, 2], [3, 4]], 1)) == APPROX([1.0, 2.0, 3.0, 4.0])


def test_pool_grid_pools_blocks_not_rows():
    """Ловушка: усреднять надо квадрат 2x2, а не пары соседей в строке."""
    grid = [
        [0, 0, 10, 10],
        [0, 0, 10, 10],
        [20, 20, 30, 30],
        [20, 20, 30, 30],
    ]
    assert flat(pool_grid(grid, 2)) == APPROX([0.0, 10.0, 20.0, 30.0])


def test_pool_grid_preserves_the_overall_mean():
    """Пулинг теряет детали, но не двигает среднее — блоки одного размера."""
    grid = [[1, 5, 9, 2], [4, 4, 0, 8], [7, 3, 6, 6], [2, 2, 1, 5]]
    before = sum(flat(grid)) / 16
    after = sum(flat(pool_grid(grid, 2))) / 4
    assert after == APPROX(before)


def test_pool_grid_shrinks_both_dimensions():
    pooled = pool_grid([[0] * 6 for _ in range(6)], 3)
    assert (len(pooled), len(pooled[0])) == (2, 2)


def test_pool_grid_rejects_a_non_divisible_side():
    with pytest.raises(ValueError):
        pool_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2)


# -------------------------------------------------------------- pooled_tokens
def test_pooled_tokens_without_pooling_is_the_whole_grid():
    assert pooled_tokens(24, 1) == 576


def test_pooled_tokens_falls_quadratically_with_the_factor():
    """Пулинг вдвое режет токены вчетверо — этим и оплачиваются лишние кадры."""
    assert pooled_tokens(24, 1) == 4 * pooled_tokens(24, 2)
    assert pooled_tokens(24, 2) == 4 * pooled_tokens(24, 4)


def test_pooled_tokens_agrees_with_pool_grid():
    grid = [[0] * 12 for _ in range(12)]
    assert pooled_tokens(12, 3) == len(flat(pool_grid(grid, 3)))


def test_pooled_tokens_rejects_a_non_divisible_side():
    with pytest.raises(ValueError):
        pooled_tokens(27, 2)


# ------------------------------------------------------------ scenario_tokens
def test_scenario_tokens_for_anyres_nine_with_thumbnail():
    assert scenario_tokens(9, 24, 2, thumbnail=True) == 1440


def test_scenario_tokens_thumbnail_costs_exactly_one_view():
    with_thumb = scenario_tokens(9, 24, 2, thumbnail=True)
    without = scenario_tokens(9, 24, 2)
    assert with_thumb - without == pooled_tokens(24, 2)


def test_scenario_tokens_for_thirty_two_video_frames():
    assert scenario_tokens(32, 24, 3) == 2048


def test_scenario_tokens_rejects_a_scenario_without_views():
    with pytest.raises(ValueError):
        scenario_tokens(0, 24, 2)


# ----------------------------------------------------------- best_pool_factor
def test_best_pool_factor_picks_the_weakest_pooling_that_fits():
    """Слабее пулинг — богаче представление, поэтому берём минимальный factor."""
    assert best_pool_factor(32, 24, 2600) == 3


def test_best_pool_factor_result_actually_fits_the_budget():
    factor = best_pool_factor(32, 24, 2600)
    assert scenario_tokens(32, 24, factor) <= 2600


def test_best_pool_factor_is_monotone_in_the_budget():
    """Больше бюджет — пулинг не может стать сильнее."""
    factors = [best_pool_factor(32, 24, b) for b in (600, 2600, 5000, 20000)]
    assert factors == sorted(factors, reverse=True)


def test_best_pool_factor_skips_factors_that_do_not_divide_the_grid():
    """Сетку 27x27 вдвое не ужать — годятся только 1 и 3."""
    assert best_pool_factor(1, 27, 100) == 3


def test_best_pool_factor_returns_none_when_nothing_fits():
    assert best_pool_factor(32, 24, 10) is None


# ------------------------------------------------------------ allocate_budget
SCENARIOS = {"single": (9, 24, True), "video": (32, 24, False)}


def test_allocate_budget_fits_every_scenario():
    plan = allocate_budget(SCENARIOS, 2600)
    assert all(slot["tokens"] <= 2600 for slot in plan.values())


def test_video_pools_harder_than_a_single_image_at_the_same_budget():
    """Суть OneVision: бюджет общий, а геометрия под него подстраивается."""
    plan = allocate_budget(SCENARIOS, 2600)
    assert plan["video"]["factor"] > plan["single"]["factor"]


def test_allocate_budget_reports_an_impossible_scenario_as_none():
    assert allocate_budget({"video": (32, 24, False)}, 10) == {"video": None}


# --------------------------------------------------------- is_valid_curriculum
def test_the_full_curriculum_is_valid():
    assert is_valid_curriculum(STAGES) is True


def test_skipping_a_middle_stage_is_allowed():
    assert is_valid_curriculum(("si", "tt")) is True


def test_video_stage_before_the_single_image_base_is_rejected():
    """Статья ablate'ит это явно: без перцептивной базы картинки проседают."""
    assert is_valid_curriculum(("ov", "si")) is False
    assert is_valid_curriculum(("ov", "tt")) is False


def test_repeating_a_stage_is_rejected():
    assert is_valid_curriculum(("si", "si", "ov")) is False


def test_an_empty_curriculum_is_rejected():
    assert is_valid_curriculum(()) is False


def test_an_unknown_stage_raises_instead_of_returning_false():
    with pytest.raises(ValueError):
        is_valid_curriculum(("si", "video"))


# ------------------------------------------------------------------ stage_steps
def test_stage_steps_splits_proportionally():
    assert stage_steps(100, {"si": 0.5, "ov": 0.3, "tt": 0.2}) == {
        "si": 50,
        "ov": 30,
        "tt": 20,
    }


def test_stage_steps_sum_is_exactly_the_total():
    """Округление вниз молча съело бы шаги обучения."""
    for total in (7, 10, 101, 1000):
        assert sum(stage_steps(total, {"si": 1, "ov": 1, "tt": 1}).values()) == total


def test_stage_steps_distributes_the_remainder_deterministically():
    assert stage_steps(10, {"si": 1, "ov": 1, "tt": 1}) == {"si": 3, "ov": 4, "tt": 3}


def test_stage_steps_depends_on_weight_ratios_not_their_scale():
    assert stage_steps(10, {"si": 1, "ov": 1, "tt": 1}) == stage_steps(
        10, {"si": 20, "ov": 20, "tt": 20}
    )


def test_stage_steps_rejects_weights_that_sum_to_zero():
    with pytest.raises(ValueError):
        stage_steps(10, {"si": 0, "ov": 0})

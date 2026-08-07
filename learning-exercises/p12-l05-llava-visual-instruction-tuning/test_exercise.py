"""Тесты к уроку «LLaVA и visual instruction tuning». Правь exercise.py."""

import pytest

from exercise import (
    IMAGE_TOKEN,
    PATCH_TOKEN,
    anyres_token_count,
    build_llava_prompt,
    context_usage,
    expand_image_placeholder,
    gelu,
    mlp_projector,
    pick_anyres_grid,
    projector_param_count,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, abs=1e-4)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def ramp(rows, cols):
    """Детерминированная матрица весов: никакого глобального random."""
    return [[((r * cols + c) % 7 - 3) * 0.25 for c in range(cols)] for r in range(rows)]


# --------------------------------------------------------------------- gelu
def test_gelu_of_zero_is_zero():
    assert gelu(0.0) == APPROX(0.0)


def test_gelu_dips_below_zero_where_relu_would_flatten():
    """Отличие от ReLU: на -1 GELU отдаёт -0.159, а не ноль."""
    assert gelu(-1.0) == ROUGH(-0.15880)
    assert gelu(-1.0) < 0


def test_gelu_passes_large_positive_values_almost_untouched():
    assert gelu(6.0) == ROUGH(6.0)


def test_gelu_satisfies_the_reflection_identity():
    """gelu(x) - gelu(-x) == x — быстрая проверка, что формула набрана верно."""
    for x in (0.3, 1.0, 2.5, 7.0):
        assert gelu(x) - gelu(-x) == ROUGH(x)


# ------------------------------------------------------------- mlp_projector
def test_mlp_projector_on_one_dimensional_toy():
    got = mlp_projector([[1.0]], [[1.0]], [0.0], [[2.0]], [0.0])
    assert flat(got) == ROUGH([1.68238])


def test_mlp_projector_keeps_every_patch_as_its_own_token():
    """Отличие от Q-Former: сжатия нет, 40 патчей дают 40 токенов LLM."""
    patches = ramp(40, 6)
    got = mlp_projector(patches, ramp(8, 6), [0.0] * 8, ramp(12, 8), [0.0] * 12)
    assert len(got) == 40
    assert all(len(t) == 12 for t in got)


def test_mlp_projector_shares_weights_across_positions():
    W1, b1, W2, b2 = ramp(4, 3), [0.1] * 4, ramp(5, 4), [0.0] * 5
    patches = [[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5], [1.0, 2.0, 3.0]]
    got = mlp_projector(patches, W1, b1, W2, b2)
    assert got[0] == APPROX(got[2])
    assert got[0] != APPROX(got[1])


def test_mlp_projector_is_not_linear():
    """Между слоями стоит GELU — иначе два linear схлопнулись бы в один."""
    W1, b1, W2, b2 = ramp(4, 2), [0.0] * 4, ramp(3, 4), [0.0] * 3
    a, b = [1.0, -2.0], [0.5, 3.0]
    got = mlp_projector([a, b, [x + y for x, y in zip(a, b)]], W1, b1, W2, b2)
    summed = [x + y for x, y in zip(got[0], got[1])]
    assert got[2] != ROUGH(summed)


def test_mlp_projector_rejects_wrong_patch_length():
    with pytest.raises(ValueError):
        mlp_projector([[1.0, 2.0, 3.0]], ramp(4, 2), [0.0] * 4, ramp(3, 4), [0.0] * 3)


# ------------------------------------------------------ projector_param_count
def test_llava_projector_is_about_21_million_params():
    assert projector_param_count(1024, 4096, 4096) == 20_979_712


def test_projector_is_a_rounding_error_next_to_the_llm():
    """21M против 7B — меньше трети процента, отсюда «обучается за часы»."""
    assert projector_param_count(1024, 4096, 4096) / 7_000_000_000 < 0.005


def test_projector_param_count_rejects_zero_dimension():
    with pytest.raises(ValueError):
        projector_param_count(1024, 0, 4096)


# ------------------------------------------------------- build_llava_prompt
def test_build_llava_prompt_exact_format():
    assert build_llava_prompt("A chat.", "Describe this image.") == (
        "A chat. USER: <image> Describe this image. ASSISTANT:"
    )


def test_image_placeholder_comes_before_the_question():
    """LLM причинная: токены вопроса обязаны видеть картинку, а не наоборот."""
    prompt = build_llava_prompt("Sys.", "What colour is the car?")
    assert prompt.index(IMAGE_TOKEN) < prompt.index("colour")


def test_prompt_ends_exactly_at_the_generation_point():
    assert build_llava_prompt("Sys.", "Hi.").endswith("ASSISTANT:")


def test_build_llava_prompt_rejects_a_blank_user_turn():
    with pytest.raises(ValueError):
        build_llava_prompt("Sys.", "   ")


# -------------------------------------------------- expand_image_placeholder
def test_expand_replaces_the_placeholder_with_patch_tokens():
    assert expand_image_placeholder("USER: <image> hi", 3) == [
        "USER:", PATCH_TOKEN, PATCH_TOKEN, PATCH_TOKEN, "hi",
    ]


def test_expand_grows_the_sequence_by_n_minus_one():
    prompt = build_llava_prompt("A chat.", "Describe this image.")
    plain = len(prompt.split())
    assert len(expand_image_placeholder(prompt, 576)) == plain - 1 + 576


def test_expanded_patch_tokens_stay_contiguous():
    """Визуальный блок — сплошной кусок, а не рассыпанные по промпту токены."""
    got = expand_image_placeholder("a <image> b", 4)
    positions = [i for i, w in enumerate(got) if w == PATCH_TOKEN]
    assert positions == list(range(positions[0], positions[0] + 4))


def test_expand_handles_a_multi_image_prompt():
    got = expand_image_placeholder("<image> and <image>", 2)
    assert got.count(PATCH_TOKEN) == 4


def test_expand_refuses_a_prompt_without_a_placeholder():
    """Иначе картинка тихо не доедет до модели, а ответ всё равно будет."""
    with pytest.raises(ValueError):
        expand_image_placeholder("USER: describe it ASSISTANT:", 576)


# --------------------------------------------------------- pick_anyres_grid
def test_a_tile_sized_image_needs_a_single_tile():
    assert pick_anyres_grid(336, 336) == (1, 1)


def test_a_double_resolution_square_uses_the_two_by_two_grid():
    assert pick_anyres_grid(672, 672) == (2, 2)


def test_a_tall_image_gets_rows_not_columns():
    assert pick_anyres_grid(1344, 672) == (2, 1)
    assert pick_anyres_grid(672, 1344) == (1, 2)


def test_grid_choice_is_transpose_symmetric():
    """Повернуть картинку на 90 градусов — значит поменять местами rows и cols."""
    for h, w in ((336, 1008), (1000, 400), (500, 500)):
        rows, cols = pick_anyres_grid(h, w)
        assert pick_anyres_grid(w, h) == (cols, rows)


def test_grid_choice_rejects_a_zero_side():
    with pytest.raises(ValueError):
        pick_anyres_grid(0, 336)


# ------------------------------------------------------- anyres_token_count
def test_anyres_on_a_672_square_costs_2880_tokens():
    """Число из урока: четыре плитки по 576 плюс превью."""
    assert anyres_token_count(672, 672) == 2880


def test_base_llava_without_thumbnail_is_576_tokens():
    assert anyres_token_count(336, 336, thumbnail=False) == 576


def test_the_thumbnail_costs_exactly_one_tile():
    with_thumb = anyres_token_count(672, 672)
    without = anyres_token_count(672, 672, thumbnail=False)
    assert with_thumb - without == 576


def test_a_bigger_image_never_costs_fewer_tokens():
    assert anyres_token_count(1344, 672) >= anyres_token_count(336, 336)


# ------------------------------------------------------------ context_usage
def test_context_usage_reports_used_free_and_share():
    got = context_usage(576, 100, 2048)
    assert got["used"] == 676
    assert got["free"] == 1372
    assert got["fits"] is True
    assert got["image_share"] == ROUGH(576 / 676)


def test_anyres_image_does_not_fit_a_2k_context():
    """2880 визуальных токенов в окне 2048 — это переполнение до первого слова."""
    got = context_usage(anyres_token_count(672, 672), 0, 2048)
    assert got["fits"] is False
    assert got["free"] < 0


def test_the_same_image_is_a_rounding_error_at_128k():
    got = context_usage(anyres_token_count(672, 672), 500, 131072)
    assert got["fits"] is True
    assert got["image_share"] < 0.9


def test_context_usage_rejects_a_nonpositive_window():
    with pytest.raises(ValueError):
        context_usage(576, 100, 0)

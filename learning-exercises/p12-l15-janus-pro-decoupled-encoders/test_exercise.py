"""Тесты к уроку «Janus-Pro: раздельные энкодеры». Правь exercise.py."""

import pytest

from exercise import (
    ENCODERS,
    GENERATE_WORDS,
    UNDERSTAND_WORDS,
    cosine_similarity,
    encoder_for,
    nearest_code,
    reconstruction_error,
    route,
    semantic_margin,
    vq_encode,
    vq_reconstruct,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in matrix for v in row]


# Четыре «картинки» двух классов. Первые две координаты несут класс,
# третья — внутриклассовую вариацию, ту самую мелкую деталь, ради которой
# и существует reconstruction-токенизатор.
VECTORS = [
    [1.0, 0.0, 0.5],
    [1.0, 0.0, -0.5],
    [0.0, 1.0, 0.5],
    [0.0, 1.0, -0.5],
]
LABELS = ["a", "a", "b", "b"]

# Codebook «под понимание»: классовые средние, деталь выброшена.
SEMANTIC_CODEBOOK = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
# Codebook «под реконструкцию»: по коду на каждую картинку, деталь сохранена.
RECON_CODEBOOK = [list(v) for v in VECTORS]


# ------------------------------------------------------- cosine_similarity
def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_cosine_ignores_vector_length():
    """SigLIP сравнивает направления: яркость картинки не должна менять смысл."""
    assert cosine_similarity([1.0, 2.0], [3.0, 6.0]) == APPROX(1.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == APPROX(-1.0)


def test_cosine_rejects_the_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


# ------------------------------------------------------------- nearest_code
def test_nearest_code_picks_the_closest_entry():
    assert nearest_code([0.9, 0.1], [[1.0, 0.0], [0.0, 1.0]]) == 0


def test_nearest_code_breaks_ties_towards_the_lower_index():
    """Иначе квантование одного и того же патча плавает от запуска к запуску."""
    assert nearest_code([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]]) == 0


def test_nearest_code_rejects_an_empty_codebook():
    with pytest.raises(ValueError):
        nearest_code([1.0], [])


def test_nearest_code_rejects_a_codebook_of_wrong_dimension():
    with pytest.raises(ValueError):
        nearest_code([1.0, 0.0], [[1.0, 0.0, 0.0]])


# ---------------------------------------------------------------- vq_encode
def test_vq_encode_worked_example():
    assert vq_encode([[0.9, 0.1], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]) == [0, 1]


def test_vq_encode_returns_one_token_per_patch():
    assert len(vq_encode(VECTORS, SEMANTIC_CODEBOOK)) == len(VECTORS)


def test_vq_encode_of_the_codebook_itself_is_the_identity():
    assert vq_encode(RECON_CODEBOOK, RECON_CODEBOOK) == [0, 1, 2, 3]


# ----------------------------------------------------------- vq_reconstruct
def test_vq_reconstruct_worked_example():
    got = vq_reconstruct([1, 0], [[1.0, 0.0], [0.0, 1.0]])
    assert flat(got) == flat([[0.0, 1.0], [1.0, 0.0]])


def test_vq_reconstruct_rejects_an_index_outside_the_codebook():
    with pytest.raises(ValueError):
        vq_reconstruct([2], [[1.0], [2.0]])


def test_vq_reconstruct_returns_copies_not_references():
    """Иначе правка декодированной картинки задним числом испортит codebook."""
    codebook = [[1.0, 0.0]]
    out = vq_reconstruct([0], codebook)
    out[0][0] = 99.0
    assert codebook[0][0] == APPROX(1.0)


# ------------------------------------------------------ reconstruction_error
def test_reconstruction_is_lossless_when_the_codebook_holds_every_vector():
    assert reconstruction_error(VECTORS, RECON_CODEBOOK) == APPROX(0.0)


def test_reconstruction_error_worked_example():
    assert reconstruction_error([[1.0, 1.0]], [[1.0, 0.0]]) == APPROX(0.5)


def test_reconstruction_error_does_not_depend_on_codebook_order():
    shuffled = [RECON_CODEBOOK[i] for i in (2, 0, 3, 1)]
    assert reconstruction_error(VECTORS, shuffled) == APPROX(
        reconstruction_error(VECTORS, RECON_CODEBOOK)
    )


def test_reconstruction_error_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        reconstruction_error([], RECON_CODEBOOK)


# ----------------------------------------------------------- semantic_margin
def test_semantic_margin_of_a_perfectly_grouped_representation_is_one():
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    assert semantic_margin(vectors, LABELS) == APPROX(1.0)


def test_semantic_margin_ignores_vector_length():
    """Метрика про углы: масштаб отдельных векторов её не двигает."""
    scaled = [[10.0 * x for x in v] for v in VECTORS]
    assert semantic_margin(scaled, LABELS) == APPROX(semantic_margin(VECTORS, LABELS))


def test_semantic_margin_goes_negative_when_the_labels_are_wrong():
    """Свои дальше чужих — представление противоречит разметке."""
    assert semantic_margin(VECTORS, ["a", "b", "b", "a"]) < 0


def test_semantic_margin_needs_more_than_one_class():
    with pytest.raises(ValueError):
        semantic_margin(VECTORS, ["a", "a", "a", "a"])


# ------------------------- главный тезис урока: один энкодер не тянет обе задачи
def test_semantic_codebook_wins_on_meaning_and_loses_on_pixels():
    semantic = vq_reconstruct(vq_encode(VECTORS, SEMANTIC_CODEBOOK), SEMANTIC_CODEBOOK)
    recon = vq_reconstruct(vq_encode(VECTORS, RECON_CODEBOOK), RECON_CODEBOOK)
    assert semantic_margin(semantic, LABELS) > semantic_margin(recon, LABELS)
    assert reconstruction_error(VECTORS, SEMANTIC_CODEBOOK) > reconstruction_error(
        VECTORS, RECON_CODEBOOK
    )


def test_the_tradeoff_is_a_real_loss_not_a_rounding_artifact():
    """Разрыв большой в обе стороны — компромисс одного энкодера виден числом."""
    semantic = vq_reconstruct(vq_encode(VECTORS, SEMANTIC_CODEBOOK), SEMANTIC_CODEBOOK)
    assert semantic_margin(semantic, LABELS) == APPROX(1.0)
    assert semantic_margin(VECTORS, LABELS) == APPROX(0.6)
    assert reconstruction_error(VECTORS, SEMANTIC_CODEBOOK) == APPROX(0.25 / 3)


# -------------------------------------------------------------------- route
def test_route_recognises_an_understanding_request():
    assert route("Describe what's in this image") == "understand"


def test_route_recognises_a_generation_request():
    assert route("Render a cyberpunk cityscape at night") == "generate"


def test_route_calls_a_mixed_request_ambiguous():
    """«Нарисуй и опиши» — счёт равный, гадать нельзя."""
    assert route("Sketch a cat and then describe its breed") == "ambiguous"


def test_route_calls_a_request_without_keywords_ambiguous():
    assert route("hello there") == "ambiguous"


def test_route_ignores_letter_case():
    assert route("GENERATE a picture") == route("generate a picture") == "generate"


def test_route_leans_on_the_module_keyword_lists():
    """Слова из списков и решают: по одному с каждой стороны — снова ничья."""
    assert route(UNDERSTAND_WORDS[0]) == "understand"
    assert route(GENERATE_WORDS[0]) == "generate"
    assert route(f"{UNDERSTAND_WORDS[0]} {GENERATE_WORDS[0]}") == "ambiguous"


# -------------------------------------------------------------- encoder_for
def test_encoder_for_understanding_is_the_semantic_tower():
    assert encoder_for("understand") == "siglip"


def test_understanding_and_generation_get_different_encoders():
    """В этом и состоит decoupling: у Chameleon оба ответа были бы одинаковы."""
    assert encoder_for("understand") != encoder_for("generate")
    assert set(ENCODERS) == {"understand", "generate", "ambiguous"}


def test_encoder_for_rejects_an_unknown_task():
    with pytest.raises(ValueError):
        encoder_for("segment")

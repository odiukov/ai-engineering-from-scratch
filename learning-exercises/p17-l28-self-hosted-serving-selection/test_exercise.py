"""Тесты к уроку «Выбор движка для self-hosted inference». Правь exercise.py."""

import pytest

from exercise import (
    WEIGHT_PROFILES,
    eligible_engines,
    maintenance_multiplier,
    normalize_weights,
    pick_engine,
    pipeline_plan,
    rank_engines,
    supports,
    weighted_score,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

NOW = "2026-08-07"
BEFORE_MAINTENANCE = "2025-12-01"

LESSON_PIPELINE = [
    ("dev", "Apple Silicon", "dev"),
    ("staging", "CPU", "staging"),
    ("prod", "NVIDIA Hopper", "prod_general"),
]


def winners(hardware, profile, now=NOW):
    """Кто победил на этом железе при этих приоритетах."""
    return pick_engine(hardware, WEIGHT_PROFILES[profile], now)["engine"]


# ----------------------------------------------------------------- supports
def test_the_nvidia_locked_engine_runs_on_hopper_but_not_on_amd():
    """Главное ограничение AMD из урока: TRT-LLM туда просто не ставится."""
    assert supports("TRT-LLM", "NVIDIA Hopper") is True
    assert supports("TRT-LLM", "AMD") is False


def test_llama_cpp_covers_cpu_and_apple_silicon():
    assert supports("llama.cpp", "CPU") is True
    assert supports("llama.cpp", "Apple Silicon") is True


def test_an_unknown_engine_is_an_error_not_a_silent_no():
    """Тихий False превратил бы опечатку в «этот движок сюда не ставится»."""
    with pytest.raises(ValueError):
        supports("vLMM", "NVIDIA Hopper")


def test_a_typo_in_the_hardware_name_is_an_error():
    with pytest.raises(ValueError):
        supports("vLLM", "NVIDIA Hoper")


# --------------------------------------------------------- eligible_engines
def test_cpu_leaves_only_the_llama_cpp_family():
    assert eligible_engines("CPU") == ["Ollama", "llama.cpp"]


def test_amd_drops_the_nvidia_locked_engine():
    assert eligible_engines("AMD") == ["SGLang", "vLLM"]


def test_hopper_admits_every_gpu_engine():
    assert eligible_engines("NVIDIA Hopper") == [
        "Ollama",
        "SGLang",
        "TGI",
        "TRT-LLM",
        "vLLM",
    ]


def test_eligibility_is_a_hard_filter_that_no_weight_can_override():
    """TRT-LLM лучший по throughput — и всё равно не появится в списке для AMD."""
    ranked = rank_engines("AMD", {"throughput": 1.0}, NOW)
    assert [engine for engine, _ in ranked] == ["vLLM", "SGLang"]


# -------------------------------------------------------- normalize_weights
def test_weights_are_scaled_to_sum_to_one():
    normalized = normalize_weights({"throughput": 3, "ecosystem": 1})
    assert sum(normalized.values()) == APPROX(1.0)


def test_normalization_keeps_the_proportions():
    assert normalize_weights({"throughput": 4, "ecosystem": 1}) == {
        "throughput": APPROX(0.8),
        "ecosystem": APPROX(0.2),
    }


def test_an_unknown_criterion_is_an_error():
    with pytest.raises(ValueError):
        normalize_weights({"vibes": 1.0})


def test_all_zero_weights_are_not_a_question_with_an_answer():
    """«Ни один критерий не важен» — это баг в конфиге, а не запрос."""
    with pytest.raises(ValueError):
        normalize_weights({"throughput": 0, "ecosystem": 0})


def test_a_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        normalize_weights({"throughput": -1.0, "ecosystem": 2.0})


# ---------------------------------------------------- maintenance_multiplier
def test_an_actively_developed_engine_is_not_penalized():
    assert maintenance_multiplier("vLLM", NOW) == APPROX(1.0)


def test_tgi_is_penalized_after_the_december_2025_announcement():
    assert maintenance_multiplier("TGI", NOW) < 1.0


def test_the_penalty_starts_on_the_announcement_day_itself():
    """11 декабря 2025 новость уже вышла — штраф действует с этого дня."""
    assert maintenance_multiplier("TGI", "2025-12-11") < 1.0
    assert maintenance_multiplier("TGI", "2025-12-10") == APPROX(1.0)


# ----------------------------------------------------------- weighted_score
def test_a_single_criterion_score_is_the_raw_table_value():
    assert weighted_score("vLLM", {"throughput": 1.0}, NOW) == APPROX(5.0)


def test_criteria_missing_from_the_weights_do_not_contribute():
    """«Кто лучше по throughput» — законный вопрос, отвечать надо только по нему."""
    only = weighted_score("Ollama", {"throughput": 1.0}, NOW)
    assert only == APPROX(1.0)


def test_the_maintenance_penalty_lands_on_the_final_score():
    """Сырая оценка экосистемы у TGI 5.0, на 2026 год остаётся 3.5."""
    assert weighted_score("TGI", {"ecosystem": 1.0}, NOW) == APPROX(3.5)
    assert weighted_score("TGI", {"ecosystem": 1.0}, BEFORE_MAINTENANCE) == APPROX(5.0)


def test_scaling_all_weights_by_the_same_factor_changes_nothing():
    small = {"throughput": 2, "ecosystem": 1}
    big = {"throughput": 200, "ecosystem": 100}
    assert weighted_score("vLLM", small, NOW) == APPROX(weighted_score("vLLM", big, NOW))


# ------------------------------------------------------------- rank_engines
def test_the_order_flips_when_the_priorities_flip():
    """Одна таблица, два набора весов — и «лучший движок» меняется местами."""
    setup = rank_engines("CPU", {"setup_simplicity": 1.0}, NOW)
    coverage = rank_engines("CPU", {"model_coverage": 1.0}, NOW)
    assert [e for e, _ in setup] == ["Ollama", "llama.cpp"]
    assert [e for e, _ in coverage] == ["llama.cpp", "Ollama"]


def test_one_hopper_box_produces_three_different_winners():
    """Железо одно, приоритеты разные — и «лучший стек» каждый раз другой."""
    assert winners("NVIDIA Hopper", "dev") == "Ollama"
    assert winners("NVIDIA Hopper", "prod_general") == "vLLM"
    assert winners("NVIDIA Hopper", "prod_agentic") == "SGLang"


def test_ranking_is_sorted_by_descending_score():
    scores = [s for _, s in rank_engines("NVIDIA Hopper", WEIGHT_PROFILES["dev"], NOW)]
    assert scores == sorted(scores, reverse=True)


def test_ties_are_broken_by_name_so_the_report_reproduces():
    """При профиле staging SGLang и TRT-LLM набирают поровну — порядок обязан быть стабилен."""
    ranked = dict(rank_engines("NVIDIA Hopper", WEIGHT_PROFILES["staging"], NOW))
    assert ranked["SGLang"] == APPROX(ranked["TRT-LLM"])
    order = [e for e, _ in rank_engines("NVIDIA Hopper", WEIGHT_PROFILES["staging"], NOW)]
    assert order.index("SGLang") < order.index("TRT-LLM")


def test_the_maintenance_date_alone_changes_who_wins():
    """Веса те же, железо то же — сдвинулась только дата, и победитель другой."""
    profile = WEIGHT_PROFILES["ecosystem_first"]
    before = rank_engines("NVIDIA Hopper", profile, BEFORE_MAINTENANCE)
    after = rank_engines("NVIDIA Hopper", profile, NOW)
    assert before[0][0] == "TGI"
    assert after[0][0] == "vLLM"


# -------------------------------------------------------------- pick_engine
def test_agentic_priorities_pick_sglang_on_the_very_same_amd_box():
    general = pick_engine("AMD", WEIGHT_PROFILES["prod_general"], NOW)
    agentic = pick_engine("AMD", WEIGHT_PROFILES["prod_agentic"], NOW)
    assert (general["engine"], agentic["engine"]) == ("vLLM", "SGLang")


def test_the_runner_up_is_the_second_line_of_the_ranking():
    ranked = rank_engines("NVIDIA Hopper", WEIGHT_PROFILES["prod_agentic"], NOW)
    choice = pick_engine("NVIDIA Hopper", WEIGHT_PROFILES["prod_agentic"], NOW)
    assert choice["runner_up"] == ranked[1][0]


def test_margin_is_the_gap_to_the_second_place():
    """Отрыв важнее оценки: маленький margin значит «выбор ни на чём не держится»."""
    ranked = rank_engines("AMD", WEIGHT_PROFILES["prod_general"], NOW)
    choice = pick_engine("AMD", WEIGHT_PROFILES["prod_general"], NOW)
    assert choice["margin"] == APPROX(ranked[0][1] - ranked[1][1])


def test_a_single_eligible_engine_leaves_no_runner_up():
    only_vllm = {"vLLM": ("AMD",)}
    choice = pick_engine("AMD", {"throughput": 1.0}, NOW, support=only_vllm)
    assert (choice["engine"], choice["runner_up"], choice["margin"]) == ("vLLM", None, None)


# ------------------------------------------------------------ pipeline_plan
def test_the_lesson_pipeline_comes_out_as_ollama_then_llama_cpp_then_vllm():
    plan = pipeline_plan(LESSON_PIPELINE, NOW)
    assert [step["engine"] for step in plan] == ["Ollama", "llama.cpp", "vLLM"]


def test_a_format_conversion_is_inserted_exactly_where_the_format_changes():
    """GGUF на ноутбуке и в staging, safetensors в проде — конвертация одна."""
    plan = pipeline_plan(LESSON_PIPELINE, NOW)
    assert [step["conversion"] for step in plan] == [None, None, "GGUF -> safetensors"]


def test_the_first_stage_never_carries_a_conversion():
    plan = pipeline_plan([("prod", "NVIDIA Hopper", "prod_general")], NOW)
    assert plan[0]["conversion"] is None and plan[0]["weight_format"] == "safetensors"


def test_an_unknown_weight_profile_is_an_error():
    """Тихий дефолт посоветовал бы движок под чужие приоритеты."""
    with pytest.raises(ValueError):
        pipeline_plan([("prod", "NVIDIA Hopper", "prod_cheap")], NOW)


def test_the_plan_moves_off_tgi_once_maintenance_mode_starts():
    stages = [("prod", "NVIDIA Hopper", "ecosystem_first")]
    assert pipeline_plan(stages, BEFORE_MAINTENANCE)[0]["engine"] == "TGI"
    assert pipeline_plan(stages, NOW)[0]["engine"] == "vLLM"

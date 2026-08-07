"""Тесты к уроку «Constitutional AI и самоулучшение». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    CONSTITUTION,
    critique,
    group_relative_advantage,
    grpo_step,
    reward_format,
    reward_math,
    revise,
    self_improvement_round,
    total_reward,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

BY_ID = {p["id"]: p for p in CONSTITUTION}

PADDED = "Paris is the capital. " + "It has museums and parks and bridges and cafes. " * 8
BURIED = (
    "Let me think about this carefully because 10 times 7 is 70 "
    "and 5 times 7 is 35, so the total is 105."
)


def fix_once(response, principle):
    """Полный цикл CAI: раскритиковал -> переписал."""
    return revise(response, critique(response, principle))


# --------------------------------------------------------------- critique
def test_critique_of_a_clean_answer_finds_nothing():
    assert critique("Paris.", BY_ID["no_refusal"])["problems"] == []


def test_critique_catches_an_unwarranted_refusal():
    result = critique("I cannot help with that. The capital is Paris.", BY_ID["no_refusal"])
    assert result["problems"] == ["unwarranted refusal"]


def test_critique_is_case_and_whitespace_insensitive():
    """«  I CANNOT ...» — тот же отказ, регистр и пробелы ничего не меняют."""
    result = critique("   I CANNOT do that. Paris.", BY_ID["no_refusal"])
    assert result["problems"] == ["unwarranted refusal"]


def test_critique_catches_hedging_even_with_punctuation():
    """«maybe,» с запятой — то же самое слово-хедж."""
    assert critique("The answer is maybe, 105.", BY_ID["no_hedging"])["problems"] == ["hedging"]


def test_each_principle_only_looks_at_its_own_rule():
    """Многословный ответ не нарушает no_refusal — критик проверяет один принцип."""
    assert critique(PADDED, BY_ID["no_refusal"])["problems"] == []


# ----------------------------------------------------------------- revise
def test_revise_leaves_a_clean_answer_untouched():
    assert revise("Paris.", {"principle": "no_refusal", "problems": []}) == "Paris."


def test_revise_drops_the_refusal_sentence():
    text = "I cannot help with that. The capital is Paris."
    assert revise(text, critique(text, BY_ID["no_refusal"])) == "The capital is Paris."


def test_revise_removes_the_hedge_word():
    text = "The answer is maybe 105."
    assert revise(text, critique(text, BY_ID["no_hedging"])) == "The answer is 105."


def test_revise_keeps_only_the_last_number():
    assert revise(BURIED, critique(BURIED, BY_ID["plain_number"])) == "105."


def test_revise_falls_back_to_the_original_when_there_is_nothing_to_cut():
    """Потерять ответ хуже, чем оставить его плохим: пустую строку не возвращаем."""
    text = "no digits anywhere in this rather long and wordy sentence indeed"
    forced = {"principle": "plain_number", "problems": ["answer buried in prose"]}
    assert revise(text, forced) == text


def test_the_critique_revise_loop_converges_for_every_principle():
    """Главный контракт CAI: после одной правки критик обязан замолчать.

    Не сойдётся — в проде цикл «критика -> правка» закрутится навсегда.
    """
    samples = {
        "no_refusal": "I cannot help with that. The capital is Paris.",
        "no_padding": PADDED,
        "no_hedging": "The answer is maybe 105.",
        "plain_number": BURIED,
    }
    for pid, text in samples.items():
        principle = BY_ID[pid]
        assert critique(text, principle)["problems"] != [], pid
        assert critique(fix_once(text, principle), principle)["problems"] == [], pid


# ------------------------------------------------------------ reward_math
def test_reward_math_accepts_the_correct_answer():
    assert reward_math("What is 15 * 7?", "The answer is 105.") == APPROX(1.0)


def test_reward_math_rejects_a_wrong_answer():
    assert reward_math("What is 15 * 7?", "I think it is 104.") == APPROX(0.0)


def test_reward_math_reads_the_last_number_not_the_first():
    """Модель рассуждает вслух — верный ответ стоит в конце, а не в начале."""
    assert reward_math("What is 15 * 7?", "10*7=70, 5*7=35, so 105") == APPROX(1.0)


def test_reward_math_handles_subtraction_with_a_negative_result():
    assert reward_math("What is 3 - 10?", "The answer is -7.") == APPROX(1.0)


def test_reward_math_returns_zero_on_a_prompt_without_arithmetic():
    """Верификатор обязан пережить любой вход, а не бросить исключение."""
    assert reward_math("Who wrote Hamlet?", "Shakespeare") == APPROX(0.0)


def test_reward_math_returns_zero_when_the_response_has_no_digits():
    assert reward_math("What is 3 + 4?", "seven") == APPROX(0.0)


# ---------------------------------------------------------- reward_format
def test_reward_format_accepts_the_answer_tag_and_rejects_a_bare_answer():
    assert reward_format("<answer>105</answer>") == APPROX(1.0)
    assert reward_format("105") == APPROX(0.0)


def test_reward_format_accepts_a_tag_spanning_several_lines():
    assert reward_format("<answer>\n105\n</answer>") == APPROX(1.0)


# ----------------------------------------------------------- total_reward
def test_total_reward_pays_a_small_bonus_for_the_format():
    assert total_reward("What is 3 + 4?", "<answer>7</answer>") == APPROX(1.1)
    assert total_reward("What is 3 + 4?", "7") == APPROX(1.0)


def test_correctness_outweighs_format_at_the_default_weight():
    """Верный ответ без тегов должен стоить дороже неверного в тегах."""
    plain_right = total_reward("What is 3 + 4?", "7")
    tagged_wrong = total_reward("What is 3 + 4?", "<answer>8</answer>")
    assert plain_right > tagged_wrong


def test_a_heavy_format_weight_makes_reward_hacking_possible():
    """format_weight = 1.0 — и «красиво оформленная ошибка» сравнялась с правдой."""
    plain_right = total_reward("What is 3 + 4?", "7", format_weight=1.0)
    tagged_wrong = total_reward("What is 3 + 4?", "<answer>8</answer>", format_weight=1.0)
    assert tagged_wrong == APPROX(plain_right)


# ------------------------------------------------ group_relative_advantage
def test_group_relative_advantage_of_a_two_element_group():
    assert group_relative_advantage([0.0, 1.0]) == pytest.approx([-1.0, 1.0])


def test_group_relative_advantage_has_unit_spread():
    adv = group_relative_advantage([0.0, 1.0, 1.0, 0.1])
    mean = sum(adv) / len(adv)
    std = math.sqrt(sum((a - mean) ** 2 for a in adv) / len(adv))
    # группа сама себе базовая линия: центр в нуле, разброс единичный
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert std == pytest.approx(1.0, abs=1e-6)


def test_a_collapsed_group_produces_no_gradient_signal():
    """Все награды одинаковы — сигнала нет, шаг надо пропустить, а не делить на ноль."""
    assert group_relative_advantage([1.0, 1.0, 1.0]) == pytest.approx([0.0, 0.0, 0.0])


def test_group_relative_advantage_ignores_the_reward_scale():
    """Сдвиг и растяжение наград ничего не меняют — важен только порядок внутри группы."""
    base = group_relative_advantage([0.0, 1.0, 0.5])
    shifted = group_relative_advantage([100.0, 101.0, 100.5])
    stretched = group_relative_advantage([0.0, 20.0, 10.0])
    assert base == pytest.approx(shifted)
    assert base == pytest.approx(stretched)


# -------------------------------------------------------------- grpo_step
def test_grpo_step_at_the_reference_point():
    out = grpo_step([0.0, 0.0], [0.0, 0.0], [1.0, -1.0])
    assert out["mean_ratio"] == APPROX(1.0)
    assert out["kl"] == APPROX(0.0)
    assert out["policy_loss"] == APPROX(0.0)
    assert out["clipped_fraction"] == APPROX(0.0)


def test_grpo_ratio_is_an_exponent_of_the_difference():
    """Ловушка: логарифмы вычитают, а не делят. Деление молча даст не то число."""
    assert grpo_step([math.log(2.0)], [0.0], [0.0])["mean_ratio"] == pytest.approx(2.0)


def test_grpo_clip_caps_a_too_large_step():
    """Политика ускакала (ratio ~ 2.7), но surrogate обрезан ровно на 1 + eps."""
    out = grpo_step([1.0], [0.0], [1.0], clip_eps=0.2)
    assert out["policy_loss"] == APPROX(-1.2)
    assert out["clipped_fraction"] == APPROX(1.0)


def test_grpo_kl_grows_as_the_policy_drifts_from_the_reference():
    near = grpo_step([0.1], [0.0], [1.0])["kl"]
    far = grpo_step([2.0], [0.0], [1.0])["kl"]
    assert far > near > 0


def test_bigger_beta_punishes_the_same_drift_harder():
    soft = grpo_step([1.0], [0.0], [1.0], beta=0.01)["total_loss"]
    hard = grpo_step([1.0], [0.0], [1.0], beta=1.0)["total_loss"]
    assert hard > soft


# --------------------------------------------- self_improvement_round
def test_a_zero_std_group_is_the_mode_collapse_signal():
    """Sampler выдаёт одно и то же — разброс ноль, преимущества нулевые."""
    out = self_improvement_round(["What is 3 + 4?"], lambda p, r: "7", random.Random(0), 4)
    entry = out["per_prompt"][0]
    assert entry["mean_reward"] == APPROX(1.0)
    assert entry["std_reward"] == APPROX(0.0)
    assert entry["advantages"] == pytest.approx([0.0] * 4)


def test_self_improvement_round_keeps_the_best_response():
    responses = iter(["6", "7", "5", "9"])
    out = self_improvement_round(
        ["What is 3 + 4?"], lambda p, r: next(responses), random.Random(0), 4
    )
    entry = out["per_prompt"][0]
    assert entry["best_response"] == "7"
    assert entry["best_reward"] == APPROX(1.0)
    assert entry["mean_reward"] == APPROX(0.25)


def test_self_improvement_round_is_reproducible_from_the_same_seed():
    """Всё случайное идёт через переданный rng — иначе отладить обучение нельзя."""
    sampler = lambda p, rng: rng.choice(["6", "7", "8", "9"])
    first = self_improvement_round(["What is 3 + 4?"], sampler, random.Random(7), 8)
    second = self_improvement_round(["What is 3 + 4?"], sampler, random.Random(7), 8)
    assert first["overall_mean"] == APPROX(second["overall_mean"])
    assert first["per_prompt"][0]["advantages"] == pytest.approx(
        second["per_prompt"][0]["advantages"]
    )


def test_self_improvement_round_averages_over_all_prompts():
    sampler = lambda p, r: "7"
    out = self_improvement_round(["What is 3 + 4?", "What is 5 * 5?"], sampler, random.Random(0), 2)
    assert len(out["per_prompt"]) == 2
    assert out["overall_mean"] == APPROX(0.5)

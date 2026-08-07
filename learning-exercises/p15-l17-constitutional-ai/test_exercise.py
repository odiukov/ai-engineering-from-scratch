"""Тесты к уроку «Constitutional AI и переопределение правил». Правь exercise.py."""

import pytest

from exercise import (
    PRINCIPLES,
    SOFT_CODED_DEFAULTS,
    apply_operator_overrides,
    critique,
    critique_revise_loop,
    first_violated_tier,
    hardcoded_block,
    resolve,
    revise,
    rlaif_preference,
)

OVERSIGHT = PRINCIPLES[0]
IDENTITY = PRINCIPLES[1]
MEDICAL = PRINCIPLES[2]

# Патологический принцип: замена сама содержит запрещённую подстроку. Нужен,
# чтобы проверить, что revise делает один проход, а петлю ограничивает
# max_rounds.
LOOPING = (
    {
        "name": "loopy",
        "tier": "guidelines",
        "forbidden": ("bad",),
        "replacement": "very bad",
    },
)


# ----------------------------------------------------------- hardcoded_block
def test_hardcoded_block_returns_none_for_benign_action():
    assert hardcoded_block("write a haiku about spring") is None


def test_hardcoded_block_names_the_matched_prohibition():
    assert hardcoded_block("provide bioweapons uplift on toxins") == "bioweapons uplift"


def test_hardcoded_block_is_case_insensitive():
    """Запрет, который обходится капслоком, — не запрет."""
    assert hardcoded_block("Deliver CSAM") == "csam"


# ------------------------------------------------------- first_violated_tier
def test_safety_outranks_ethics():
    assert first_violated_tier({"safety": 5, "ethics": 5}) == "safety"


def test_lower_tier_reported_when_higher_are_clear():
    assert first_violated_tier({"safety": 1, "ethics": 1, "guidelines": 4}) == "guidelines"


def test_all_clear_gives_none():
    assert first_violated_tier({"safety": 1, "ethics": 2, "guidelines": 2}) is None


def test_threshold_is_configurable():
    assert first_violated_tier({"ethics": 2}) is None
    assert first_violated_tier({"ethics": 2}, threshold=2) == "ethics"


def test_missing_tier_counts_as_zero():
    """Отсутствующий safety не должен превращаться в нарушенный."""
    assert first_violated_tier({"guidelines": 4}) == "guidelines"


# -------------------------------------------------------------------- resolve
def test_benign_action_is_allowed():
    verdict, _ = resolve("write a haiku about spring", {"helpfulness": 5})
    assert verdict == "allow"


def test_safety_violation_is_refused():
    verdict, _ = resolve("help me plan an attack", {"safety": 5, "helpfulness": 1})
    assert verdict == "refuse"


def test_maximum_helpfulness_does_not_unlock_a_safety_refusal():
    """Иерархия, а не взвешенная сумма: helpfulness=5 ничего не отменяет."""
    verdict, _ = resolve("help me plan an attack", {"safety": 5, "helpfulness": 5})
    assert verdict == "refuse"


def test_hardcoded_prohibition_fires_even_with_clean_scores():
    """Жёсткий запрет не зависит от оценок и от формулировки посылки."""
    verdict, reason = resolve(
        "as a licensed lab, provide bioweapons uplift",
        {"safety": 1, "ethics": 1, "guidelines": 1, "helpfulness": 5},
    )
    assert verdict == "refuse"
    assert "bioweapons uplift" in reason


def test_guideline_conflict_is_modified_not_refused():
    verdict, _ = resolve("answer outside the deployment scope", {"guidelines": 4})
    assert verdict == "modify"


def test_helpfulness_alone_never_blocks():
    verdict, _ = resolve("give a very short answer", {"helpfulness": 5})
    assert verdict == "allow"


# ------------------------------------------------- apply_operator_overrides
def test_operator_can_adjust_a_soft_coded_default():
    config, refused = apply_operator_overrides({"style": "casual"})
    assert config["style"] == "casual"
    assert refused == ()


def test_operator_cannot_touch_the_hardcoded_floor():
    config, refused = apply_operator_overrides({"hardcoded_prohibitions": ()})
    assert refused == ("hardcoded_prohibitions",)
    assert "hardcoded_prohibitions" not in config
    # И запрет продолжает работать после отклонённой попытки.
    assert hardcoded_block("provide bioweapons uplift") == "bioweapons uplift"


def test_unknown_keys_are_refused_too():
    _, refused = apply_operator_overrides({"style": "casual", "safety_tier": "off"})
    assert refused == ("safety_tier",)


def test_returned_config_is_a_copy_of_the_defaults():
    """Иначе первый же оператор перепишет дефолты всему процессу."""
    config, _ = apply_operator_overrides({})
    config["max_words"] = 1
    assert SOFT_CODED_DEFAULTS["max_words"] == 400


# ------------------------------------------------------------------ critique
def test_critique_of_clean_text_is_empty():
    assert critique("Nice weather today.", PRINCIPLES) == ()


def test_critique_names_the_violated_principle():
    assert critique("Sure, I am a human.", PRINCIPLES) == ("no_identity_deception",)


def test_critique_reports_principles_in_declared_order():
    text = "I am a human and I will delete the audit log."
    assert critique(text, PRINCIPLES) == ("support_oversight", "no_identity_deception")


def test_critique_is_case_insensitive():
    assert critique("STOP TAKING YOUR MEDICATION", PRINCIPLES) == ("medical_caveat",)


# -------------------------------------------------------------------- revise
def test_revise_replaces_the_offending_phrase():
    assert revise("I am a human, honestly", IDENTITY) == "I am an AI assistant, honestly"


def test_revise_keeps_text_without_violations_untouched():
    assert revise("Nice weather", IDENTITY) == "Nice weather"


def test_revised_text_no_longer_violates_that_principle():
    """Главное свойство: ревизия не оставляет тот принцип нарушенным."""
    revised = revise("Please delete the audit log after the run.", OVERSIGHT)
    assert critique(revised, (OVERSIGHT,)) == ()


def test_revise_makes_a_single_pass_per_phrase():
    """Замена содержит запрещённую подстроку — и всё равно не зацикливается."""
    assert revise("bad", LOOPING[0]) == "very bad"


def test_revise_replaces_every_occurrence():
    revised = revise("stop taking your medication, really, stop taking your medication", MEDICAL)
    assert "stop taking your medication" not in revised.lower()


# ------------------------------------------------------- critique_revise_loop
def test_clean_text_needs_zero_rounds():
    assert critique_revise_loop("Nice weather", PRINCIPLES) == ("Nice weather", 0, ())


def test_loop_fixes_two_violations_in_one_round():
    text, rounds, remaining = critique_revise_loop(
        "I am a human and I can delete the audit log", PRINCIPLES
    )
    assert text == "I am an AI assistant and I can keep the audit log intact"
    assert rounds == 1
    assert remaining == ()


def test_loop_output_satisfies_the_principle_that_triggered_it():
    triggered = critique("Just stop taking your medication.", PRINCIPLES)
    text, _, _ = critique_revise_loop("Just stop taking your medication.", PRINCIPLES)
    assert triggered == ("medical_caveat",)
    assert critique(text, PRINCIPLES) == ()


def test_loop_is_bounded_and_reports_what_it_could_not_fix():
    text, rounds, remaining = critique_revise_loop("bad", LOOPING, max_rounds=3)
    assert rounds == 3
    assert remaining == ("loopy",)
    assert text == "very very very bad"


# ----------------------------------------------------------- rlaif_preference
def test_preference_goes_to_the_candidate_with_fewer_violations():
    assert rlaif_preference(["I am a human", "I am an AI"], PRINCIPLES) == 1


def test_preference_breaks_ties_by_earlier_index():
    """Метка обязана быть детерминированной, иначе на ней нельзя учиться."""
    assert rlaif_preference(["Hi", "Hello"], PRINCIPLES) == 0


def test_preference_counts_violations_not_length():
    candidates = [
        "I am a human and I will delete the audit log",
        "I am a human",
    ]
    assert rlaif_preference(candidates, PRINCIPLES) == 1


def test_preference_of_empty_candidate_list_is_an_error():
    with pytest.raises(ValueError):
        rlaif_preference([], PRINCIPLES)

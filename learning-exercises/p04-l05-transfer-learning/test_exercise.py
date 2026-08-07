"""Тесты к уроку «Transfer learning и дообучение». Правь exercise.py."""

import pytest

from exercise import (
    freeze_backbone,
    freeze_bn_stats,
    param_groups,
    pick_recipe,
    progressive_unfreeze,
    sgd_step,
    stage_lrs,
    trainable_summary,
)

APPROX = lambda x: pytest.approx(x, abs=1e-12)

STAGES = [["conv1", "bn1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"], ["fc"]]


def model():
    """Маленький ResNet-подобный список параметров, как named_parameters()."""
    spec = [
        ("conv1.weight", "conv", 9408),
        ("bn1.weight", "bn", 64),
        ("bn1.bias", "bn", 64),
        ("layer1.0.conv1.weight", "conv", 36864),
        ("layer1.0.bn1.weight", "bn", 64),
        ("layer2.0.conv1.weight", "conv", 73728),
        ("layer3.0.conv1.weight", "conv", 294912),
        ("layer4.0.conv1.weight", "conv", 1179648),
        ("fc.weight", "linear", 5120),
        ("fc.bias", "linear", 10),
    ]
    return [
        {"name": n, "kind": k, "size": s, "value": 1.0, "trainable": True}
        for n, k, s in spec
    ]


def names(params, trainable=True):
    return [p["name"] for p in params if p["trainable"] is trainable]


# ------------------------------------------------------------- pick_recipe
def test_tiny_dataset_freezes_the_whole_backbone():
    assert pick_recipe(500) == "freeze_backbone"
    assert pick_recipe(500, "far") == "freeze_backbone"


def test_one_thousand_is_already_the_second_row_of_the_table():
    """Границы включающие: 999 — ещё заморозка, 1000 — уже нет."""
    assert pick_recipe(999) == "freeze_backbone"
    assert pick_recipe(1000) == "freeze_early"


def test_far_domain_unfreezes_more_at_the_same_dataset_size():
    """КТ-снимкам мало разморозить голову: статистика пикселей другая."""
    assert pick_recipe(5000, "close") == "freeze_early"
    assert pick_recipe(5000, "far") == "finetune_all"


def test_scratch_training_is_only_for_big_and_far():
    assert pick_recipe(500_000, "far") == "train_from_scratch"
    assert pick_recipe(500_000, "close") == "finetune_all"


def test_recipe_never_gets_more_conservative_as_data_grows():
    """Больше данных — не меньше свободы. Немонотонность здесь была бы багом."""
    rank = {
        "freeze_backbone": 0,
        "freeze_early": 1,
        "finetune_all": 2,
        "train_from_scratch": 3,
    }
    sizes = [10, 999, 1000, 9999, 10_000, 99_999, 100_000, 10**6]
    for domain in ("close", "far"):
        ranks = [rank[pick_recipe(n, domain)] for n in sizes]
        assert ranks == sorted(ranks)


# --------------------------------------------------------- freeze_backbone
def test_only_the_head_stays_trainable():
    assert names(freeze_backbone(model())) == ["fc.weight", "fc.bias"]


def test_freezing_does_not_mutate_the_original_model():
    """Линейный пробник и полный fine-tune считают на одной модели по очереди."""
    m = model()
    freeze_backbone(m)
    assert all(p["trainable"] for p in m)


def test_head_prefix_is_configurable():
    """У EfficientNet голова называется classifier, у ViT — heads.head."""
    m = [{**p, "name": p["name"].replace("fc.", "classifier.")} for p in model()]
    frozen = freeze_backbone(m, head_prefix="classifier")
    assert names(frozen) == ["classifier.weight", "classifier.bias"]


def test_frozen_backbone_leaves_under_one_percent_trainable():
    summary = trainable_summary(freeze_backbone(model()))
    total = summary["trainable"] + summary["frozen"]
    assert summary["trainable"] / total < 0.01


# --------------------------------------------------------- freeze_bn_stats
def test_bn_parameters_go_to_eval_and_stop_training():
    frozen = freeze_bn_stats(model())
    bn = [p for p in frozen if p["kind"] == "bn"]
    assert bn and all(p["trainable"] is False for p in bn)
    assert all(p["eval_mode"] is True for p in bn)


def test_conv_and_linear_parameters_are_left_alone():
    """model.train() включает всё, а эта функция откатывает ровно BN."""
    frozen = freeze_bn_stats(model())
    others = [p for p in frozen if p["kind"] != "bn"]
    assert all(p["trainable"] is True for p in others)
    assert all("eval_mode" not in p for p in others)


def test_freezing_bn_does_not_mutate_the_original_model():
    m = model()
    freeze_bn_stats(m)
    assert all(p["trainable"] for p in m)
    assert all("eval_mode" not in p for p in m)


def test_bn_stays_frozen_after_a_stage_is_unfrozen():
    """Порядок вызовов важен: разморозили стадию — снова прижимаем BN."""
    unfrozen = progressive_unfreeze(model(), ["layer1"], epoch=0)
    final = freeze_bn_stats(unfrozen)
    bn_in_layer1 = [p for p in final if p["name"] == "layer1.0.bn1.weight"][0]
    conv_in_layer1 = [p for p in final if p["name"] == "layer1.0.conv1.weight"][0]
    assert bn_in_layer1["trainable"] is False
    assert conv_in_layer1["trainable"] is True


# ------------------------------------------------------ trainable_summary
def test_total_weight_count_survives_any_freezing():
    full = trainable_summary(model())
    probe = trainable_summary(freeze_backbone(model()))
    assert full["trainable"] + full["frozen"] == probe["trainable"] + probe["frozen"]


def test_summary_counts_weights_not_tensors():
    """У головы два тензора, но пять тысяч весов — решает вторая цифра."""
    assert trainable_summary(freeze_backbone(model()))["trainable"] == 5130


def test_untouched_model_has_nothing_frozen():
    assert trainable_summary(model())["frozen"] == 0


# ------------------------------------------------------------- stage_lrs
def test_last_stage_gets_exactly_the_base_lr():
    lrs = stage_lrs(STAGES, base_lr=1e-3, decay=0.3)
    assert lrs[-1][1] == APPROX(1e-3)


def test_each_earlier_stage_is_decay_times_slower():
    lrs = [lr for _, lr in stage_lrs(STAGES, base_lr=1e-3, decay=0.3)]
    for early, late in zip(lrs, lrs[1:]):
        assert early == pytest.approx(late * 0.3, rel=1e-12)


def test_stem_runs_at_decay_to_the_power_of_stage_count_minus_one():
    lrs = stage_lrs(STAGES, base_lr=1e-3, decay=0.3)
    assert lrs[0][1] == APPROX(1e-3 * 0.3 ** 5)


def test_stage_name_joins_its_prefixes():
    assert stage_lrs(STAGES)[0][0] == "conv1_bn1"


# ----------------------------------------------------------- param_groups
def test_group_collects_only_parameters_with_a_matching_prefix():
    groups = param_groups(model(), STAGES)
    by_name = {g["name"]: g["params"] for g in groups}
    assert by_name["layer1"] == ["layer1.0.conv1.weight", "layer1.0.bn1.weight"]
    assert by_name["fc"] == ["fc.weight", "fc.bias"]


def test_frozen_parameters_never_enter_a_group():
    """Замороженный параметр в группе всё равно поедет от weight decay."""
    groups = param_groups(freeze_backbone(model()), STAGES)
    collected = [n for g in groups for n in g["params"]]
    assert collected == ["fc.weight", "fc.bias"]


def test_empty_groups_are_dropped():
    groups = param_groups(freeze_backbone(model()), STAGES)
    assert [g["name"] for g in groups] == ["fc"]


def test_every_trainable_parameter_lands_in_exactly_one_group():
    groups = param_groups(model(), STAGES)
    collected = [n for g in groups for n in g["params"]]
    assert sorted(collected) == sorted(names(model()))
    assert len(collected) == len(set(collected))


def test_group_lr_comes_from_the_stage_schedule():
    groups = param_groups(model(), STAGES, base_lr=1e-3, decay=0.3)
    assert {g["name"]: g["lr"] for g in groups} == {
        name: APPROX(lr) for name, lr in stage_lrs(STAGES, 1e-3, 0.3)
    }


# ---------------------------------------------------- progressive_unfreeze
def test_first_epoch_trains_the_head_and_the_last_stage_only():
    schedule = ["layer4", "layer3", "layer2", "layer1"]
    got = names(progressive_unfreeze(model(), schedule, epoch=0))
    assert got == ["layer4.0.conv1.weight", "fc.weight", "fc.bias"]


def test_head_is_trainable_at_every_epoch():
    """Голова обучается с нулевой эпохи, иначе её случайные градиенты снесут backbone."""
    schedule = ["layer4", "layer3"]
    for epoch in range(5):
        assert "fc.weight" in names(progressive_unfreeze(model(), schedule, epoch))


def test_trainable_set_only_grows_with_epochs():
    schedule = ["layer4", "layer3", "layer2", "layer1"]
    previous = set()
    for epoch in range(6):
        current = set(names(progressive_unfreeze(model(), schedule, epoch)))
        assert previous <= current
        previous = current


def test_schedule_stops_when_exhausted():
    """Расписание кончилось — stem так и остаётся замороженным."""
    schedule = ["layer4", "layer3"]
    got = names(progressive_unfreeze(model(), schedule, epoch=99))
    assert "conv1.weight" not in got
    assert "layer3.0.conv1.weight" in got


# --------------------------------------------------------------- sgd_step
def test_trainable_parameter_moves_against_the_gradient():
    params = [{"name": "fc.w", "kind": "linear", "size": 1, "value": 1.0, "trainable": True}]
    out = sgd_step(params, {"fc.w": 2.0}, {"fc.w": 0.1})
    assert out[0]["value"] == APPROX(0.8)


def test_frozen_backbone_does_not_move_even_with_a_gradient():
    """Вся суть feature extraction: градиент есть, а веса стоят."""
    params = freeze_backbone(model())
    grads = {p["name"]: 1.0 for p in params}
    lrs = {p["name"]: 0.5 for p in params}
    out = sgd_step(params, grads, lrs)
    for before, after in zip(params, out):
        if before["name"].startswith("fc"):
            assert after["value"] != before["value"]
        else:
            assert after["value"] == APPROX(before["value"])


def test_parameter_missing_from_the_optimizer_silently_stays_put():
    """Разморозили слой, а оптимизатор собран раньше — лосс падает, метрика стоит."""
    params = model()
    grads = {p["name"]: 1.0 for p in params}
    lrs = {"fc.weight": 0.1}  # оптимизатор знает только про голову
    out = {p["name"]: p["value"] for p in sgd_step(params, grads, lrs)}
    assert out["fc.weight"] == APPROX(0.9)
    assert out["layer4.0.conv1.weight"] == APPROX(1.0)


def test_zero_gradient_leaves_the_value_alone():
    params = model()
    lrs = {p["name"]: 0.1 for p in params}
    out = sgd_step(params, {p["name"]: 0.0 for p in params}, lrs)
    assert [p["value"] for p in out] == APPROX([1.0] * len(params))


def test_discriminative_lr_makes_the_stem_move_less_than_the_head():
    """Ранние слои двигаются на три порядка слабее — ради этого всё и затевалось."""
    params = model()
    groups = param_groups(params, STAGES, base_lr=1e-2, decay=0.3)
    lrs = {n: g["lr"] for g in groups for n in g["params"]}
    out = {p["name"]: p["value"] for p in sgd_step(params, {n: 1.0 for n in lrs}, lrs)}
    stem_shift = 1.0 - out["conv1.weight"]
    head_shift = 1.0 - out["fc.weight"]
    assert stem_shift == pytest.approx(head_shift * 0.3 ** 5, rel=1e-9)


def test_sgd_step_does_not_mutate_the_input():
    params = model()
    sgd_step(params, {"fc.weight": 5.0}, {"fc.weight": 1.0})
    assert params[-2]["value"] == 1.0

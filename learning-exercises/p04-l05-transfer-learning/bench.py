"""Входные данные для замера скорости."""

import random

random.seed(0)

_STAGES = [["conv1", "bn1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"], ["fc"]]

# ~2000 тензоров, как у ResNet-152: перебор префиксов на каждой стадии
# становится заметным, если делать его в лоб
_params = []
for _stage in range(1, 5):
    for _block in range(80):
        for _kind, _size in (("conv", 36864), ("bn", 64), ("bn", 64)):
            _params.append(
                {
                    "name": f"layer{_stage}.{_block}.{_kind}1.weight",
                    "kind": _kind,
                    "size": _size,
                    "value": random.random(),
                    "trainable": True,
                }
            )
_params.append(
    {"name": "conv1.weight", "kind": "conv", "size": 9408, "value": 0.1, "trainable": True}
)
_params.append(
    {"name": "fc.weight", "kind": "linear", "size": 5120, "value": 0.2, "trainable": True}
)

_grads = {p["name"]: random.uniform(-1.0, 1.0) for p in _params}
_lrs = {p["name"]: 1e-3 for p in _params}

BENCH = {
    "pick_recipe": (50_000, "far"),
    "freeze_backbone": (_params,),
    "freeze_bn_stats": (_params,),
    "trainable_summary": (_params,),
    "stage_lrs": (_STAGES, 1e-3, 0.3),
    "param_groups": (_params, _STAGES, 1e-3, 0.3),
    "progressive_unfreeze": (_params, ["layer4", "layer3", "layer2", "layer1"], 2),
    "sgd_step": (_params, _grads, _lrs),
}

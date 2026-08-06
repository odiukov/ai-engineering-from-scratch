"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N_ROWS = 400
_N_COLS = 8

# каждая двадцатая ячейка — пропуск, чтобы импьютеру было что делать
_raw = [
    [None if random.random() < 0.05 else random.gauss(0, 3) for _ in range(_N_COLS)]
    for _ in range(_N_ROWS)
]
_clean = [[0.0 if v is None else v for v in row] for row in _raw]

_impute_state = {"medians": [0.0] * _N_COLS}
_scale_state = {"means": [0.0] * _N_COLS, "stds": [1.0] * _N_COLS}
_fitted = [
    {"kind": "impute", "state": _impute_state},
    {"kind": "scale", "state": _scale_state},
]
_dumped = (
    '[{"kind": "impute", "state": {"medians": %s}}, '
    '{"kind": "scale", "state": {"means": %s, "stds": %s}}]'
    % ([0.0] * _N_COLS, [0.0] * _N_COLS, [1.0] * _N_COLS)
)

BENCH = {
    "column_medians": (_raw,),
    "fit_step": ("impute", _raw),
    "apply_step": ("scale", _scale_state, _clean),
    "fit_pipeline": (["impute", "scale"], _raw),
    "transform_pipeline": (_fitted, _raw),
    "fit_transform_split": (["impute", "scale"], _raw, 300),
    "dump_pipeline": (_fitted,),
    "load_pipeline": (_dumped,),
}

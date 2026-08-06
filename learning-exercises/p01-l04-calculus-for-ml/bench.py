"""Входные данные для замера скорости."""

_f1 = lambda t: t * t
_f2 = lambda p: sum(x * x for x in p)
_point = [1.0] * 20

BENCH = {
    "derivative": (_f1, 3.0),
    "gradient": (_f2, _point),
    "descent_step": (_point, _point, 0.1),
    "minimize": (_f2, [1.0, 1.0], 0.1, 50),
    "is_close_to_minimum": (_f2, [0.0] * 5),
}

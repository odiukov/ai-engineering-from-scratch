"""Входные данные для замера скорости."""

_VERBS = ("mine", "gather", "craft", "place", "smelt", "brew", "tame", "build")
_NOUNS = ("ore", "sticks", "pickaxe", "table", "ingot", "potion", "wolf", "shelter")

_library = {}
for _i in range(200):
    _name = f"skill_{_i:03d}"
    _library[_name] = {
        "name": _name,
        "description": (f"{_VERBS[_i % len(_VERBS)]} "
                        f"{_NOUNS[(_i // 3) % len(_NOUNS)]} "
                        f"from the nearby area number {_i % 17}"),
        "code": f"step_{_i}()",
        "tags": ("gather",) if _i % 2 == 0 else ("craft",),
        # цепочка зависимостей длиной 200: топологический обход не должен
        # разворачиваться в квадрат
        "depends_on": (f"skill_{_i - 1:03d}",) if _i else (),
        "version": 1,
        "history": (),
    }

_runtime = {name: (lambda env: "ok") for name in _library}

_wanted = tuple(f"{v} {n} from the nearby area" for v in _VERBS for n in _NOUNS)

BENCH = {
    "make_skill": ("mine_ore", "mine iron ore from rock", "mine(3)"),
    "register_skill": (_library, _library["skill_000"], lambda s: (True, "ok")),
    "search_skills": (_library, "craft a pickaxe from ore and sticks", 5),
    "dependency_order": (_library, "skill_199"),
    "compose_skill": (_library, "combo", "combo of many skills",
                      tuple(f"skill_{i:03d}" for i in range(0, 60, 3))),
    "execute_skill": (_library, "skill_199", _runtime),
    "propose_next_task": (_library, _wanted),
}

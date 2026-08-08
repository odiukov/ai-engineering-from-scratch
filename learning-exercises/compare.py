#!/usr/bin/env python3
"""
Сравнение твоего exercise.py с эталонным solution.py.

Меряет то, что вообще поддаётся измерению:
  * сложность кода   — строки, циклы, вложенность, ветвления
  * скорость         — реальный прогон обеих версий на одинаковых входах

Запуск:
    python3 learning-exercises/compare.py p01-l01-linear-algebra-intuition

Рядом гоняется ruff (если есть uvx) — он ловит неиспользуемые переменные,
мёртвый код и стилевые огрехи.

Метрики — не приговор. Твой вариант длиннее эталона на строку и читается
лучше — значит твой лучше. Но вдвое больше циклов или заметно медленнее —
повод разобраться.
"""

import ast
import importlib.util
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"


# --------------------------------------------------------------- метрики AST
class FnMetrics(ast.NodeVisitor):
    """Считает по телу одной функции: циклы, ветвления, глубину вложенности."""

    def __init__(self):
        self.loops = 0
        self.branches = 0
        self.depth = 0
        self._cur = 0

    def _nest(self, node):
        self._cur += 1
        self.depth = max(self.depth, self._cur)
        self.generic_visit(node)
        self._cur -= 1

    def visit_For(self, node):
        self.loops += 1
        self._nest(node)

    def visit_While(self, node):
        self.loops += 1
        self._nest(node)

    def visit_comprehension(self, node):
        self.loops += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.branches += 1
        self._nest(node)

    def visit_IfExp(self, node):
        self.branches += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.branches += len(node.values) - 1
        self.generic_visit(node)


def measure(path):
    """{имя функции: {метрики}} для всех функций верхнего уровня файла."""
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = node.body
        # docstring не считаем кодом
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        if not body:
            lines = 0
        else:
            lines = body[-1].end_lineno - body[0].lineno + 1
        m = FnMetrics()
        for stmt in body:
            m.visit(stmt)
        out[node.name] = {
            "строк": lines,
            "циклов": m.loops,
            "вложенность": m.depth,
            "ветвлений": m.branches,
        }
    return out


# ------------------------------------------------------------------ загрузка
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bench(fn, args, budget=0.04):
    """Медианное время одного вызова в микросекундах. None, если падает.

    Число повторов подбирается под бюджет времени: быстрые функции гоняются
    много раз, медленные — мало. Иначе замер тяжёлой функции вешает скрипт.
    """
    try:
        fn(*args)
    except Exception:
        return None
    t0 = time.perf_counter()
    fn(*args)
    single = time.perf_counter() - t0
    reps = max(1, min(5000, int(budget / single) if single > 0 else 5000))
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        samples.append((time.perf_counter() - t0) / reps * 1e6)
    return statistics.median(samples)


# ---------------------------------------------------------------------- вывод
def verdict(mine, ref, lower_is_better=True):
    if mine is None or ref is None:
        return DIM + "—" + RESET
    if ref == 0:
        return DIM + "—" + RESET
    ratio = mine / ref
    if not lower_is_better:
        ratio = 1 / ratio if ratio else 0
    if ratio <= 1.15:
        return GREEN + "ок" + RESET
    if ratio <= 2.0:
        return YELLOW + f"×{ratio:.1f}" + RESET
    return RED + f"×{ratio:.1f}" + RESET


def main():
    lesson = sys.argv[1] if len(sys.argv) > 1 else None
    root = Path(__file__).parent
    if lesson is None:
        candidates = sorted(root.glob("p*-l*"))
        if not candidates:
            sys.exit("нет ни одного pNN-lNN-* каталога")
        lesson_dir = candidates[-1]
    else:
        lesson_dir = root / lesson
    if not lesson_dir.is_dir():
        sys.exit(f"нет каталога {lesson_dir}")

    mine_path = lesson_dir / "exercise.py"
    ref_path = lesson_dir / "solution.py"
    if not ref_path.exists():
        sys.exit(f"нет эталона {ref_path}")

    # exercise.py не в git — у каждого свой. Первый запуск создаёт из шаблона.
    template = lesson_dir / "exercise.template.py"
    if not mine_path.exists() and template.exists():
        mine_path.write_text(template.read_text())
        print(DIM + "создал exercise.py из шаблона" + RESET)

    print(f"\n{BOLD}Сравнение с эталоном — {lesson_dir.name}{RESET}\n")

    mine_m, ref_m = measure(mine_path), measure(ref_path)

    sys.path.insert(0, str(lesson_dir))
    mine_mod = load(mine_path, "_mine")
    ref_mod = load(ref_path, "_ref")

    bench_spec = {}
    bench_file = lesson_dir / "bench.py"
    if bench_file.exists():
        bench_spec = load(bench_file, "_bench").BENCH

    hdr = f"{'функция':<22}{'строк':>13}{'циклов':>13}{'вложен.':>13}{'мкс':>20}"
    print(BOLD + hdr + RESET)
    print(DIM + "-" * len(hdr) + RESET)
    print(DIM + f"{'':<22}{'ты / эталон':>13}{'ты / эталон':>13}{'ты / эталон':>13}{'ты / эталон':>20}" + RESET)
    print()

    unwritten = []
    for name, ref_stats in ref_m.items():
        my_stats = mine_m.get(name)
        my_fn = getattr(mine_mod, name, None)
        ref_fn = getattr(ref_mod, name, None)

        args = bench_spec.get(name)
        t_mine = bench(my_fn, args) if (args and my_fn) else None
        t_ref = bench(ref_fn, args) if (args and ref_fn) else None
        if t_mine is None and args:
            unwritten.append(name)

        def cell(key, width=13):
            a = my_stats[key] if my_stats else "?"
            b = ref_stats[key]
            return f"{a} / {b}".rjust(width)

        tcell = (f"{t_mine:.2f} / {t_ref:.2f}" if t_mine and t_ref else "— / —").rjust(18)
        flag = verdict(t_mine, t_ref)

        print(f"{name:<22}{cell('строк')}{cell('циклов')}{cell('вложенность')}{tcell} {flag}")

    print()
    if unwritten:
        print(DIM + "ещё не написаны: " + ", ".join(unwritten) + RESET)
        print()

    # ------------------------------------------------------------------ ruff
    print(f"{BOLD}Линтер{RESET}")
    try:
        r = subprocess.run(
            ["uvx", "ruff", "check", str(mine_path),
             "--select", "E,F,W,B,C4,SIM,PERF,RUF",
             # RUF001-003 ругаются на кириллицу в комментариях — это не ошибка
             "--ignore", "RUF001,RUF002,RUF003,E501",
             "--no-cache", "--quiet"],
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        print(out if out else GREEN + "чисто, замечаний нет" + RESET)
    except FileNotFoundError:
        print(DIM + "uvx не найден — линтер пропущен" + RESET)
    except subprocess.TimeoutExpired:
        print(DIM + "линтер не уложился в таймаут" + RESET)

    print()
    print(DIM + "Метрики — повод подумать, а не приговор. Читаемость важнее" + RESET)
    print(DIM + "лишней строки. Спроси в чате — разберём твой вариант словами." + RESET)
    print()


if __name__ == "__main__":
    main()

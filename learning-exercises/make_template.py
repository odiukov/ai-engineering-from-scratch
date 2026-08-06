#!/usr/bin/env python3
"""
Делает exercise.template.py из solution.py: оставляет сигнатуру и docstring,
тело заменяет на raise NotImplementedError.

    python3 learning-exercises/make_template.py                # все уроки
    python3 learning-exercises/make_template.py p01-l02-...    # один

Смысл: docstring с примерами пишется один раз — в эталоне, и оттуда
попадает в заготовку. Расхождению между ними взяться неоткуда.
"""

import ast
import sys
from pathlib import Path

HEADER = '''"""
{title}

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh {slug}
Разбор:  /check-code {slug}
"""
'''


def build(sol_path: Path) -> str:
    src = sol_path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    slug = sol_path.parent.name

    # заголовок модуля из первой строки docstring эталона
    # Соглашение: первая непустая строка docstring эталона — чистый заголовок
    # урока, опционально с суффиксом " — эталон".
    mod_doc = ast.get_docstring(tree) or ""
    title = next((ln for ln in mod_doc.splitlines() if ln.strip()), slug)
    title = title.split(" — эталон")[0].strip()

    out = [HEADER.format(title=title, slug=slug)]

    # импорты верхнего уровня переносим как есть
    imports = [
        "\n".join(lines[n.lineno - 1 : n.end_lineno])
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    if imports:
        out.append("\n".join(imports) + "\n")

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        sig = "\n".join(lines[node.lineno - 1 : node.body[0].lineno - 1])
        doc = node.body[0]
        has_doc = isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
        body = "\n".join(lines[doc.lineno - 1 : doc.end_lineno]) if has_doc else ""
        chunk = sig
        if body:
            chunk += "\n" + body
        chunk += "\n    raise NotImplementedError\n"
        out.append("\n" + chunk)

    return "\n".join(out)


def main():
    root = Path(__file__).parent
    targets = (
        [root / sys.argv[1]] if len(sys.argv) > 1 else sorted(root.glob("p*-l*"))
    )
    made = 0
    for d in targets:
        sol = d / "solution.py"
        if not sol.exists():
            print(f"  пропуск {d.name}: нет solution.py")
            continue
        (d / "exercise.template.py").write_text(build(sol))
        made += 1
        print(f"  {d.name}")
    print(f"\nсобрано заготовок: {made}")


if __name__ == "__main__":
    main()

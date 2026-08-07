"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_patch = "--- a/calc.py\n+++ b/calc.py\n@@ -1,200 +1,200 @@\n" + "".join(
    f" context line {i}\n-    old_value_{i} = {i}\n+    new_value_{i} = {i * 2}\n"
    for i in range(400)
)

_issue = " ".join(f"new_value_{i} = {i * 2}" for i in range(0, 400, 3))

_tasks = [(f"t{i:04d}", _issue, _patch) for i in range(20)]

_outcomes = [(f"t{i:04d}", random.random() < 0.4) for i in range(5000)]

_contaminated = {f"t{i:04d}" for i in range(0, 5000, 7)}

_question = ("Visit the arXiv listing and find the chart in the pdf, then "
             "extract the audio caption and finally search the video mirror "
             "after checking the graph. ") * 40

BENCH = {
    "parse_patch": (_patch,),
    "resolve_rate": (_outcomes,),
    "solution_leakage": (_issue, _patch),
    "contaminated_ids": (_tasks,),
    "clean_resolve_rate": (_outcomes, _contaminated),
    "pass_at_k": (400, 37, 200),
    "gaia_level": (_question,),
}

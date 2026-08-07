"""
Бенчмарки: SWE-bench, GAIA, AgentBench — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь мы руками собираем то, что harness SWE-bench делает за кадром: парсим
unified diff, гоняем ворота FAIL_TO_PASS / PASS_TO_PASS, считаем resolve rate
и pass@k, ищем утечку решения в тексте issue (аудит SWE-bench+) и раскладываем
вопрос GAIA по уровням сложности. Ни сети, ни LLM — только структуры данных.
"""

import math

# Статусы прогона тестов, как их печатает pytest в harness SWE-bench.
PASSED = "PASSED"
FAILED = "FAILED"

# Порог, с которого задача считается загрязнённой (SWE-bench+ нашёл утечку
# решения в 32.67% успешных патчей — там порог был экспертный, у нас числовой).
LEAKAGE_THRESHOLD = 0.5

# Слова, по которым GAIA-подобный классификатор меряет глубину декомпозиции.
STEP_WORDS = ("then", "after", "finally", "next", "and")
MODALITY_WORDS = ("image", "video", "audio", "pdf", "chart", "graph")
TOOL_WORDS = ("search", "look up", "find", "visit", "extract")


def parse_patch(patch):
    """Разобрать unified diff в словарь {путь: {"added": [...], "removed": [...]}}.

    parse_patch("--- a/f.py\\n+++ b/f.py\\n@@ -1 +1 @@\\n-x = 1\\n+x = 2\\n")
        ->  {"f.py": {"added": ["x = 2"], "removed": ["x = 1"]}}
    parse_patch("")  ->  {}

    Строки без префикса (контекст) не попадают никуда. Порядок строк внутри
    файла сохраняется.

    Ловушка: заголовки "+++ b/file" и "--- a/file" сами начинаются с "+" и "-".
    Если проверять префиксы в неверном порядке, путь файла уедет в added.
    Префикс "b/" из "+++ b/f.py" надо срезать — это метка "после патча",
    а не часть пути.

    Зачем: evaluator SWE-bench применяет патч к репозиторию, а аудит
    SWE-bench+ сравнивает добавленные строки с текстом issue.
    """
    files = {}
    current = None
    for line in patch.splitlines():
        # Порядок проверок принципиален: "+++"/"---" длиннее "+"/"-",
        # и разбирать их надо первыми.
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            for prefix in ("a/", "b/"):
                if path.startswith(prefix):
                    path = path[len(prefix):]
            current = files.setdefault(path, {"added": [], "removed": []})
            continue
        if line.startswith("@@"):
            continue
        if current is None:
            # строки до первого заголовка файла — мусор преамбулы
            continue
        if line.startswith("+"):
            current["added"].append(line[1:])
        elif line.startswith("-"):
            current["removed"].append(line[1:])
    return files


def is_resolved(test_results, fail_to_pass, pass_to_pass):
    """Ворота SWE-bench: задача решена, если прошли ОБА списка тестов.

    is_resolved({"t1": PASSED, "t2": PASSED}, ["t1"], ["t2"])  ->  True
    is_resolved({"t1": FAILED, "t2": PASSED}, ["t1"], ["t2"])  ->  False
    is_resolved({"t1": PASSED}, ["t1"], ["t2"])                ->  False

    test_results — словарь {имя теста: PASSED или FAILED}. Тест, которого в
    словаре нет, считается упавшим: "не запустился" — это не "прошёл".

    Пустой fail_to_pass — это ValueError. Задача без единого падающего теста
    не проверяет ничего: её "решит" пустой патч.

    Зачем: PASS_TO_PASS — второе горло. Патч, который чинит баг и ломает
    соседний тест, засчитан не будет.
    """
    if not fail_to_pass:
        raise ValueError("fail_to_pass пуст: задача не проверяет ничего")
    for name in list(fail_to_pass) + list(pass_to_pass):
        if test_results.get(name) != PASSED:
            return False
    return True


def resolve_rate(outcomes):
    """Доля решённых задач. outcomes — список пар (task_id, resolved).

    resolve_rate([("t1", True), ("t2", False)])  ->  0.5
    resolve_rate([])                             ->  0.0

    Метрика не зависит от порядка задач в списке — это просто среднее.
    Повторяющийся task_id — ValueError: одна задача, посчитанная дважды,
    тихо перекашивает знаменатель.
    """
    seen = set()
    solved = 0
    for task_id, resolved in outcomes:
        if task_id in seen:
            raise ValueError(f"task_id встречается дважды: {task_id!r}")
        seen.add(task_id)
        if resolved:
            solved += 1
    if not seen:
        return 0.0
    return solved / len(seen)


def solution_leakage(issue_text, patch):
    """Доля добавленных патчем содержательных строк, дословно лежащих в issue.

    p = "+++ b/f.py\\n+    return value * 2\\n"
    solution_leakage("the fix is: return value * 2", p)  ->  1.0
    solution_leakage("something is broken", p)           ->  0.0

    Содержательной считаем строку, у которой после strip осталось >= 8
    символов и которая не начинается с "#": короткие "}" и "pass" совпадут
    с чем угодно и раздуют оценку.

    Сравнение идёт по нормализованному тексту: нижний регистр плюс схлопнутые
    пробелы. Иначе отступ в патче помешает найти ту же строку в issue.

    Если содержательных добавленных строк нет — 0.0, а не деление на ноль.

    Зачем: SWE-bench+ показал, что в 32.67% успешных патчей решение лежало
    прямо в тексте issue. Модель не чинила баг, а списывала.
    """
    added = []
    for changes in parse_patch(patch).values():
        added.extend(changes["added"])

    meaningful = [ln for ln in added
                  if len(ln.strip()) >= 8 and not ln.strip().startswith("#")]
    if not meaningful:
        return 0.0

    haystack = " ".join(issue_text.lower().split())
    leaked = sum(1 for ln in meaningful
                 if " ".join(ln.lower().split()) in haystack)
    return leaked / len(meaningful)


def contaminated_ids(tasks, threshold=LEAKAGE_THRESHOLD):
    """Множество task_id, у которых утечка решения >= threshold.

    tasks — список троек (task_id, issue_text, patch).

    p = "+++ b/f.py\\n+    return value * 2\\n"
    contaminated_ids([("t1", "just do return value * 2", p)])  ->  {"t1"}
    contaminated_ids([("t1", "it is broken", p)])              ->  set()

    Сравнение именно >=, а не >: при threshold=1.0 задача с полностью
    списанным патчем обязана попасть в загрязнённые.
    """
    return {task_id for task_id, issue, patch in tasks
            if solution_leakage(issue, patch) >= threshold}


def clean_resolve_rate(outcomes, contaminated):
    """Resolve rate по чистым задачам плюс отчёт, сколько выкинули.

    Возвращает словарь {"rate": ..., "evaluated": ..., "excluded": ...}.

    clean_resolve_rate([("t1", True), ("t2", True)], {"t2"})
        ->  {"rate": 1.0, "evaluated": 1, "excluded": 1}
    clean_resolve_rate([("t1", True)], {"t1"})
        ->  {"rate": 0.0, "evaluated": 0, "excluded": 1}

    Загрязнённая задача уходит и из числителя, и из ЗНАМЕНАТЕЛЯ. Если
    оставить её в знаменателе, метрика скажет "стало хуже", хотя стало
    честнее. Поэтому excluded возвращается наружу: 50% по 500 задачам и 50%
    по 12 задачам — разные числа доверия.

    В excluded считаются только те id, которые реально встретились в
    outcomes: посторонний id в contaminated ничего не исключает.
    """
    kept = [(tid, ok) for tid, ok in outcomes if tid not in contaminated]
    excluded = sum(1 for tid, _ in outcomes if tid in contaminated)
    return {
        "rate": resolve_rate(kept),
        "evaluated": len(kept),
        "excluded": excluded,
    }


def pass_at_k(n, c, k):
    """Несмещённая оценка pass@k: шанс, что хотя бы одна из k попыток решает.

    n — сколько сэмплов сгенерировали, c — сколько из них решили задачу.

    pass_at_k(10, 1, 1)   ->  0.1
    pass_at_k(10, 1, 10)  ->  1.0
    pass_at_k(10, 0, 5)   ->  0.0

    Формула: 1 - C(n-c, k) / C(n, k) — вероятность, что в случайной выборке
    из k попыток НЕ окажется ни одной удачной, вычтенная из единицы.

    Наивное "c/n >= 1" мимо: pass@k по определению не убывает с ростом k,
    и на этом свойстве держатся все таблицы pass@1 / pass@5 / pass@10.

    ValueError, если k > n, k < 1, c < 0 или c > n.
    """
    if k < 1 or k > n:
        raise ValueError(f"k вне диапазона 1..n: k={k}, n={n}")
    if c < 0 or c > n:
        raise ValueError(f"c вне диапазона 0..n: c={c}, n={n}")
    if n - c < k:
        # неудачных сэмплов меньше k — промахнуться всеми k попытками нельзя
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def gaia_level(question):
    """Уровень сложности GAIA (1, 2 или 3) по глубине декомпозиции вопроса.

    gaia_level("What is the capital of France?")  ->  1
    gaia_level("Search the paper and extract the first author.")  ->  2

    Считаем три слагаемых по тексту в нижнем регистре:
      steps      = 1 + сколько слов из STEP_WORDS встретилось (по словам!);
      modalities = сколько РАЗНЫХ слов из MODALITY_WORDS встретилось;
      tools      = сколько РАЗНЫХ фраз из TOOL_WORDS встретилось.
    score = steps + modalities + tools; <= 2 даёт 1, <= 5 даёт 2, иначе 3.

    Ловушка: STEP_WORDS ищутся как отдельные слова, а не как подстроки.
    Иначе "and" найдётся внутри "Andrew" и уровень поедет.

    Зачем: GAIA задумана как "легко человеку (92%), трудно ИИ (GPT-4 с
    плагинами 15%)", и трудность там ровно в длине цепочки инструментов.
    """
    low = question.lower()
    words = [w.strip(".,;:!?()[]\"'") for w in low.split()]
    steps = 1 + sum(1 for w in words if w in STEP_WORDS)
    modalities = sum(1 for m in MODALITY_WORDS if m in low)
    tools = sum(1 for t in TOOL_WORDS if t in low)
    score = steps + modalities + tools
    if score <= 2:
        return 1
    if score <= 5:
        return 2
    return 3

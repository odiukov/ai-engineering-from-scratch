"""
Рецепты open-weight VLM: что реально влияет — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Пять осей пространства дизайна из Idefics2 плюс визуальный бюджет токенов,
# который Prismatic назвал главным источником разброса.
AXES = ("encoder", "connector", "llm", "data", "resolution", "tokens")


def parse_ablation_row(line):
    """Разобрать строку ablation-таблицы в плоский словарь.

    Формат: настройки осей, "|", метрики. Поля внутри половины через ";".

    parse_ablation_row("encoder=siglip;tokens=576 | mmmu=41.2")
        ->  {"encoder": "siglip", "tokens": 576, "mmmu": 41.2}
    parse_ablation_row("encoder=clip | mmmu=38.0;docvqa=71.5")
        ->  {"encoder": "clip", "mmmu": 38.0, "docvqa": 71.5}

    Числа приводим к числам: "576" -> int, "41.2" -> float, "siglip"
    остаётся строкой. Без этого "tokens" будет сравниваться как текст, и
    "1024" окажется меньше "576".

    Строка без "|" или поле без "=" — ValueError: молча пропустить кривую
    строку хуже, чем упасть, потому что дальше по ней считается дельта.
    """
    if "|" not in line:
        raise ValueError(f"нет разделителя '|': {line!r}")
    left, right = line.split("|", 1)
    row = {}
    for half in (left, right):
        for field in half.split(";"):
            field = field.strip()
            if not field:
                continue
            if "=" not in field:
                raise ValueError(f"поле без '=': {field!r}")
            key, raw = field.split("=", 1)
            raw = raw.strip()
            # порядок попыток важен: int("41.2") падает, float("41.2") нет,
            # поэтому сначала int, иначе все целые станут float
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            row[key.strip()] = value
    if not row:
        raise ValueError(f"пустая строка: {line!r}")
    return row


def controlled_pairs(rows, axis):
    """Индексы пар строк, отличающихся РОВНО одной осью axis.

    Это определение слова «ablation»: одна ручка крутится, остальные
    заморожены. Сравниваем только ключи из AXES, метрики игнорируем.

    rows = [{"encoder": "clip", "connector": "mlp", "mmmu": 38.0},
            {"encoder": "siglip", "connector": "mlp", "mmmu": 41.0},
            {"encoder": "siglip", "connector": "qformer", "mmmu": 41.6}]
    controlled_pairs(rows, "encoder")    ->  [(0, 1)]
    controlled_pairs(rows, "connector")  ->  [(1, 2)]

    Пара (0, 2) не считается ни для одной оси: там сменились две ручки
    сразу, и приписать разницу одной из них нельзя.

    Пары возвращаются в порядке возрастания (i, j), i < j.
    """
    pairs = []
    others = [a for a in AXES if a != axis]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a.get(axis) == b.get(axis):
                continue
            # None здесь означает «ось в строке не указана»; две строки без
            # этой оси считаются совпадающими, и это то, что нужно
            if all(a.get(k) == b.get(k) for k in others):
                pairs.append((i, j))
    return pairs


def axis_delta(rows, axis, value_a, value_b, metric):
    """Средний прирост metric при замене axis=value_a на axis=value_b.

    Считается только по контролируемым парам. Так MM1 получает свои
    «+3 балла MMMU за SigLIP вместо CLIP».

    rows = [{"encoder": "clip", "connector": "mlp", "mmmu": 38.0},
            {"encoder": "siglip", "connector": "mlp", "mmmu": 41.0}]
    axis_delta(rows, "encoder", "clip", "siglip", "mmmu")  ->  3.0
    axis_delta(rows, "encoder", "siglip", "clip", "mmmu")  ->  -3.0

    Если контролируемых пар с такими значениями нет — вернуть None, а не
    ноль: «не измеряли» и «измерили, разницы нет» — разные ответы.
    """
    deltas = []
    for i, j in controlled_pairs(rows, axis):
        a, b = rows[i], rows[j]
        if metric not in a or metric not in b:
            continue
        if a.get(axis) == value_a and b.get(axis) == value_b:
            deltas.append(b[metric] - a[metric])
        elif a.get(axis) == value_b and b.get(axis) == value_a:
            deltas.append(a[metric] - b[metric])
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def explained_variance(rows, axis, metric):
    """Доля дисперсии metric, объяснённая осью axis (это eta-квадрат).

    Ровно то, что делает Prismatic: «бюджет токенов объясняет ~60%
    разброса, энкодер ~20%, коннектор ~5%».

    Считается как SS_between / SS_total: сумма квадратов отклонений
    групповых средних от общего среднего, делённая на полную сумму
    квадратов.

    explained_variance([{"e": "a", "m": 1.0}, {"e": "b", "m": 3.0}],
                       "e", "m")   ->  1.0   (ось определяет метрику)
    explained_variance([{"e": "a", "m": 2.0}, {"e": "b", "m": 2.0}],
                       "e", "m")   ->  0.0   (метрика не шевелится)

    Если разброса нет вовсе (SS_total == 0) — вернуть 0.0, а не делить на
    ноль. Строки без axis или без metric пропускаем.
    """
    usable = [r for r in rows if axis in r and metric in r]
    if len(usable) < 2:
        return 0.0
    values = [r[metric] for r in usable]
    grand = sum(values) / len(values)
    ss_total = sum((v - grand) ** 2 for v in values)
    if ss_total == 0:
        return 0.0
    groups = {}
    for r in usable:
        groups.setdefault(r[axis], []).append(r[metric])
    ss_between = 0.0
    for vals in groups.values():
        mean = sum(vals) / len(vals)
        ss_between += len(vals) * (mean - grand) ** 2
    return ss_between / ss_total


def rank_axes_by_impact(rows, metric, axes=AXES):
    """Оси по убыванию объяснённой дисперсии: что ablation'ить первым.

    Возвращает список пар (axis, share).

    rank_axes_by_impact(rows, "mmmu")[0][0]  ->  "encoder"

    При равных долях порядок сохраняется как в axes — сортировка
    устойчивая, поэтому «не хуже» не превращается в «случайно раньше».
    """
    scored = [(axis, explained_variance(rows, axis, metric)) for axis in axes]
    return sorted(scored, key=lambda pair: -pair[1])


def expected_score(baseline, swaps, delta_table):
    """Предсказать метрику после набора замен, складывая измеренные дельты.

    delta_table: {(axis, было, стало): дельта}
    swaps: [(axis, было, стало), ...]

    table = {("encoder", "clip", "siglip"): 3.0,
             ("resolution", 384, 448): 1.5}
    expected_score(38.0, [("encoder", "clip", "siglip")], table)  ->  41.0
    expected_score(38.0, [], table)                               ->  38.0

    Обратную замену искать не надо отдельным ключом: если ("a","x","y")
    в таблице есть, то ("a","y","x") — это та же дельта со знаком минус.
    Замены, которой нет ни в прямом, ни в обратном виде, — KeyError.

    Модель аддитивная и потому наивная: в реальности оси взаимодействуют
    (SigLIP на 2B и на 70B дают разный прирост). Это оценка, не истина.
    """
    score = baseline
    for axis, was, now in swaps:
        if (axis, was, now) in delta_table:
            score += delta_table[(axis, was, now)]
        elif (axis, now, was) in delta_table:
            score -= delta_table[(axis, now, was)]
        else:
            raise KeyError((axis, was, now))
    return score


def pick_recipe(rows, metric, max_tokens=None, require=None):
    """Индекс лучшей строки таблицы под ограничениями. Нет такой — None.

    max_tokens: потолок визуальных токенов на изображение (бюджет LLM).
    require: обязательные значения осей, например {"data": "pixmo"}.

    pick_recipe(rows, "mmmu")                      ->  индекс максимума
    pick_recipe(rows, "mmmu", max_tokens=576)      ->  максимум среди тех,
                                                       у кого tokens <= 576

    При равной метрике берём меньший индекс: результат должен быть
    воспроизводимым, а не зависеть от порядка обхода словаря.

    Строки без metric не рассматриваем. Строку без "tokens" при заданном
    max_tokens тоже отбрасываем: неизвестная цена — не бесплатная.
    """
    best = None
    for i, row in enumerate(rows):
        if metric not in row:
            continue
        if max_tokens is not None:
            if "tokens" not in row or row["tokens"] > max_tokens:
                continue
        if require and any(row.get(k) != v for k, v in require.items()):
            continue
        if best is None or row[metric] > rows[best][metric]:
            best = i
    return best

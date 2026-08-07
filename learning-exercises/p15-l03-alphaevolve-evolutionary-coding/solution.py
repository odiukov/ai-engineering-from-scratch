"""
AlphaEvolve: эволюционный поиск программ — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Разбираем архитектуру AlphaEvolve (Novikov et al., DeepMind, arXiv:2506.13131)
на одной игрушечной задаче — символьной регрессии:

    seed-программа -> «LLM» предлагает точечную правку -> оценщик считает
    ошибку -> архив MAP-elites оставляет лучшего в каждой ячейке -> повтор

Роль LLM здесь играет mutate() на стандартном random — нам важна не
изобретательность генератора, а то, что архитектуру держит ОЦЕНЩИК. Уберёшь
held-out — и петля начнёт оптимизировать метрику, а не задачу.

Программа — вложенный кортеж:
    ("num", 2.0)          константа
    ("x",)                аргумент
    ("add", a, b)         сумма
    ("mul", a, b)         произведение

Например 2*x + 1 это ("add", ("mul", ("num", 2.0), ("x",)), ("num", 1.0)).
"""

import math

# Дескриптор MAP-elites: клетка задаётся глубиной выражения и ведром по
# величине самой большой константы. Обе координаты подрезаны сверху, иначе
# сетка растёт бесконечно и перестаёт быть сеткой.
MAX_DEPTH = 6
MAX_CONST_BUCKET = 4

# Из чего «LLM» собирает новые листья.
CONSTANTS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def evaluate_expr(expr, x):
    """Посчитать значение программы в точке x.

    evaluate_expr(("num", 2.0), 5.0)             ->  2.0
    evaluate_expr(("x",), 5.0)                   ->  5.0
    evaluate_expr(("mul", ("x",), ("x",)), 3.0)  ->  9.0

    Неизвестный тег — ValueError, а не тихий None. Оценщик, который молча
    возвращает мусор на сломанной программе, ровно та дырка, через которую
    эволюция и утекает.
    """
    tag = expr[0]
    if tag == "num":
        return float(expr[1])
    if tag == "x":
        return float(x)
    if tag == "add":
        return evaluate_expr(expr[1], x) + evaluate_expr(expr[2], x)
    if tag == "mul":
        return evaluate_expr(expr[1], x) * evaluate_expr(expr[2], x)
    raise ValueError(f"неизвестный узел программы: {tag!r}")


def depth(expr):
    """Глубина дерева программы: лист равен 1.

    depth(("x",))                              ->  1
    depth(("add", ("x",), ("num", 1.0)))       ->  2
    depth(("mul", ("add", ("x",), ("x",)), ("x",)))  ->  3

    Одна из двух координат ячейки MAP-elites: она разводит «короткую и
    тупую» программу и «длинную и хитрую» по разным клеткам, чтобы вторая не
    вытеснила первую до того, как её успеют улучшить.
    """
    if expr[0] in ("num", "x"):
        return 1
    return 1 + max(depth(expr[1]), depth(expr[2]))


def mse(expr, xs, target):
    """Среднеквадратичная ошибка программы на точках xs против target(x).

    mse(("x",), [1.0, 2.0], lambda x: x)     ->  0.0
    mse(("num", 0.0), [1.0, 3.0], lambda x: x)  ->  5.0   ((1 + 9) / 2)

    Это и есть машинно-проверяемый оценщик из урока: детерминированный,
    быстрый и не обсуждаемый. Всё, что AlphaEvolve выиграл — 48 умножений
    для 4x4, ядро FlashAttention, эвристика Borg, — выиграно там, где такой
    оценщик существует.

    Ловушки:
      * пустой список точек — ValueError. Средняя ошибка по нулю точек это не
        ноль, а отсутствие измерения, и «идеальная программа» с mse=0 на
        пустой выборке — классический способ обмануть петлю;
      * длинная цепочка умножений переполняет float. OverflowError ловим и
        возвращаем inf: такая программа просто худшая, а не повод падать.
    """
    if not xs:
        raise ValueError("оценщику нужна хотя бы одна точка")
    total = 0.0
    for x in xs:
        try:
            error = evaluate_expr(expr, x) - target(x)
        except OverflowError:
            return math.inf
        total += error * error
    return total / len(xs)


def cell_key(expr):
    """Ячейка MAP-elites: (глубина, ведро по максимальной константе).

    cell_key(("x",))                ->  (1, 0)
    cell_key(("num", 5.0))          ->  (1, 2)
    cell_key(("add", ("x",), ("num", 3.0)))  ->  (2, 1)

    Обе координаты подрезаны: глубина не выше MAX_DEPTH, ведро не выше
    MAX_CONST_BUCKET. Без подрезки каждая новая программа получала бы личную
    клетку, архив превращался бы в список, и элитизм перестал бы работать.

    Смысл сетки: конкурируют между собой только похожие программы. Ровно так
    AlphaEvolve не даёт лидеру задавить всё разнообразие на десятом поколении.
    """
    def biggest_const(e):
        if e[0] == "num":
            return abs(float(e[1]))
        if e[0] == "x":
            return 0.0
        return max(biggest_const(e[1]), biggest_const(e[2]))

    return (
        min(depth(expr), MAX_DEPTH),
        min(int(biggest_const(expr) // 2), MAX_CONST_BUCKET),
    )


def mutate(rng, expr):
    """«LLM» предлагает точечную правку программы. Вернуть НОВОЕ выражение.

    Четыре равновероятных хода: заменить всё листом, обернуть в сумму с
    листом, обернуть в произведение с листом, подправить одну константу.

    mutate(random.Random(0), ("x",))  ->  какое-то валидное выражение

    Почему не «случайные байты»: обычный генетический алгоритм на исходнике
    почти всегда выдаёт синтаксическую ошибку. LLM даёт компилируемые правки в
    осмысленной окрестности родителя, и оценщик перестаёт тратить вызовы
    впустую. Наш mutate — та же идея на минималках: он не умеет породить
    невалидное дерево.

    rng — параметр. Ветку эволюции нужно уметь повторить.
    """
    def leaf():
        if rng.random() < 0.5:
            return ("x",)
        return ("num", float(rng.choice(CONSTANTS)))

    def perturb(e):
        if e[0] == "num":
            return ("num", e[1] + rng.choice((-1.0, -0.5, 0.5, 1.0)))
        if e[0] == "x":
            return e
        if rng.random() < 0.5:
            return (e[0], perturb(e[1]), e[2])
        return (e[0], e[1], perturb(e[2]))

    roll = rng.random()
    if roll < 0.25:
        return leaf()
    if roll < 0.50:
        return ("add", expr, leaf())
    if roll < 0.75:
        return ("mul", expr, leaf())
    return perturb(expr)


def archive_insert(archive, expr, score):
    """Вставить программу в архив MAP-elites. Вернуть НОВЫЙ архив.

    Архив — словарь {ячейка: (программа, оценка)}. Меньшая оценка лучше.
    Программа занимает свою ячейку, только если она СТРОГО лучше жильца.

    archive_insert({}, ("x",), 1.0)          ->  {(1, 0): (("x",), 1.0)}
    archive_insert(a, ("x",), 5.0)           ->  a без изменений, если в
                                                 ячейке уже лежит оценка 1.0

    Строгое сравнение обязательно: при равных оценках жилец остаётся. Иначе
    порядок вставки начинает влиять на содержимое архива, и один и тот же
    прогон с тем же seed даёт разные результаты в зависимости от того, в каком
    порядке подъехали одинаково хорошие кандидаты.

    Вход не мутируем: функция возвращает копию. Так тесты могут вставлять один
    и тот же набор в разном порядке и сравнивать итог.
    """
    key = cell_key(expr)
    updated = dict(archive)
    incumbent = updated.get(key)
    if incumbent is None or score < incumbent[1]:
        updated[key] = (expr, score)
    return updated


def best_of(archive):
    """Лучший житель архива: пара (программа, оценка) с минимальной оценкой.

    best_of({(1, 0): (("x",), 3.0), (2, 1): (("num", 1.0), 1.0)})
        ->  (("num", 1.0), 1.0)

    При равных оценках выигрывает меньшая ячейка — иначе «лучший» плавает от
    запуска к запуску, и историю прогона нельзя сравнить с прошлой.

    Это элитизм: пока архив не забывает своего чемпиона, качество по
    поколениям не может ухудшиться, каким бы неудачным ни оказался потомок.

    Пустой архив — ValueError.
    """
    if not archive:
        raise ValueError("архив пуст, чемпиона нет")
    # ключ сравнения (оценка, ячейка): ячейка разводит ничьи детерминированно
    key = min(archive, key=lambda cell: (archive[cell][1], cell))
    return archive[key]


def evolve(rng, seed_expr, generations, train_xs, holdout_xs, target,
           use_holdout=True):
    """Эволюционный цикл AlphaEvolve. Вернуть словарь с итогом прогона.

    Ключи результата:
      "expr"        — лучшая программа по сигналу поиска;
      "score"       — её оценка тем же сигналом;
      "train_mse"   — ошибка на обучающих точках;
      "holdout_mse" — ошибка на отложенных;
      "gap"         — holdout_mse минус train_mse;
      "history"     — оценка чемпиона после каждого поколения.

    Сигнал поиска: при use_holdout=True это среднее train и holdout, при
    use_holdout=False — только train.

    evolve(rng, ("x",), 200, [0.0, 1.0], [0.5], lambda x: x)["train_mse"]
        ->  0.0 (программа ("x",) уже идеальна, петля её не портит)

    Два свойства, которые надо получить и проверить:
      * history не возрастает. Архив не теряет чемпиона — значит, ни одно
        поколение не может ухудшить результат;
      * при use_holdout=False «score» совпадает с train_mse. Петля буквально
        не видит отложенных точек, и «gap» показывает, насколько она себе
        польстила. Это и есть reward hacking в самой мягкой форме: оптимизируем
        измеримое, а не нужное.

    Родитель выбирается из sorted(archive) — по отсортированным ключам, а не
    по порядку словаря. Иначе прогон зависит от истории вставок, и seed
    перестаёт что-либо гарантировать.
    """
    def signal(expr):
        train = mse(expr, train_xs, target)
        if not use_holdout:
            return train
        return 0.5 * (train + mse(expr, holdout_xs, target))

    archive = archive_insert({}, seed_expr, signal(seed_expr))
    history = []
    for _ in range(generations):
        parent_cell = rng.choice(sorted(archive))
        child = mutate(rng, archive[parent_cell][0])
        archive = archive_insert(archive, child, signal(child))
        history.append(best_of(archive)[1])

    expr, score = best_of(archive)
    train_mse = mse(expr, train_xs, target)
    holdout_mse = mse(expr, holdout_xs, target)
    return {
        "expr": expr,
        "score": score,
        "train_mse": train_mse,
        "holdout_mse": holdout_mse,
        "gap": holdout_mse - train_mse,
        "history": history,
    }

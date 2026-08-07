"""
LLaVA-OneVision: одна модель на картинку, набор картинок и видео — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Стадии обучения OneVision в том порядке, в котором их обязан пройти
# учебный план: single-image -> OneVision -> task transfer.
STAGES = ("si", "ov", "tt")

# Множители пулинга, которые имеет смысл перебирать. 1 — пулинга нет.
POOL_FACTORS = (1, 2, 3, 4, 6, 8)


def pool_grid(grid, factor):
    """Средний пулинг 2D-сетки патчей блоками factor x factor.

    OneVision ужимает 24x24 патчей до 12x12 (factor=2) или 8x8 (factor=3).
    Пулинг делается в СЕТКЕ патчей, а не в плоском списке токенов: иначе
    соседи по вертикали разъедутся и локальность потеряется.

    pool_grid([[1, 2], [3, 4]], 2)  ->  [[2.5]]
    pool_grid([[1, 2], [3, 4]], 1)  ->  [[1.0, 2.0], [3.0, 4.0]]

    Результат всегда из float — среднее целых чисел целым не обязано быть.

    Сторона сетки, не кратная factor, — ValueError. Настоящий OneVision
    интерполирует билинейно и умеет 27 -> 14, мы ограничиваемся кратными:
    так виден сам механизм, а не арифметика краёв.
    """
    if factor < 1:
        raise ValueError(f"factor должен быть >= 1, получено {factor}")
    if not grid or not grid[0]:
        raise ValueError("пустая сетка")
    rows, cols = len(grid), len(grid[0])
    if rows % factor or cols % factor:
        raise ValueError(f"сетка {rows}x{cols} не кратна factor={factor}")
    out = []
    for r in range(0, rows, factor):
        line = []
        for c in range(0, cols, factor):
            # блок целиком, а не построчно: среднее по прямоугольнику
            block = [grid[r + dr][c + dc] for dr in range(factor) for dc in range(factor)]
            line.append(sum(block) / len(block))
        out.append(line)
    return out


def pooled_tokens(patches_per_side, pool_factor):
    """Сколько визуальных токенов останется от одной квадратной сетки патчей.

    pooled_tokens(24, 1)  ->  576   (пулинга нет)
    pooled_tokens(24, 2)  ->  144
    pooled_tokens(24, 3)  ->  64

    Зависимость квадратичная: пулинг вдвое режет токены вчетверо. Отсюда и
    берётся бюджет — один шаг пулинга освобождает место под три лишних кадра.

    Некратная сторона или неположительные аргументы — ValueError.
    """
    if patches_per_side < 1 or pool_factor < 1:
        raise ValueError("patches_per_side и pool_factor должны быть >= 1")
    if patches_per_side % pool_factor:
        raise ValueError(
            f"сторона {patches_per_side} не кратна pool_factor={pool_factor}"
        )
    side = patches_per_side // pool_factor
    return side * side


def scenario_tokens(views, patches_per_side, pool_factor, thumbnail=False):
    """Полный бюджет токенов сценария: views одинаковых сеток плюс превью.

    views — это тайлы AnyRes, картинки multi-image или кадры видео: для LLM
    разницы нет, она видит один плоский поток токенов.

    scenario_tokens(9, 24, 2, thumbnail=True)  ->  1440   (AnyRes-9)
    scenario_tokens(6, 24, 1)                  ->  3456   (multi-image)
    scenario_tokens(32, 24, 3)                 ->  2048   (видео, 32 кадра)

    thumbnail добавляет ровно один вид — уменьшённую копию всей картинки,
    чтобы модель видела общий план, а не только тайлы.
    """
    if views < 1:
        raise ValueError(f"views должно быть >= 1, получено {views}")
    per_view = pooled_tokens(patches_per_side, pool_factor)
    return (views + (1 if thumbnail else 0)) * per_view


def best_pool_factor(
    views, patches_per_side, budget, thumbnail=False, factors=POOL_FACTORS
):
    """Самый слабый пулинг, при котором сценарий влезает в бюджет токенов.

    Слабый пулинг = больше токенов = богаче представление. Поэтому берём
    МИНИМАЛЬНЫЙ factor из тех, что влезают, а не первый попавшийся.

    best_pool_factor(32, 24, 2600)  ->  3     (при 2 вышло бы 4608 токенов)
    best_pool_factor(1, 24, 10000)  ->  1
    best_pool_factor(32, 24, 10)    ->  None  (не влезает никак)

    Множители, на которые сторона не делится, просто пропускаем: сетку
    27x27 нельзя ужать вдвое.
    """
    for factor in sorted(factors):
        if patches_per_side % factor:
            continue
        if scenario_tokens(views, patches_per_side, factor, thumbnail) <= budget:
            return factor
    return None


def allocate_budget(scenarios, budget):
    """Подобрать пулинг каждому сценарию под ОБЩИЙ бюджет на образец.

    scenarios: {имя: (views, patches_per_side, thumbnail)}
    Возвращает {имя: {"factor": f, "tokens": t}} либо {имя: None}, если
    сценарий не влезает даже с максимальным пулингом.

    allocate_budget({"video": (32, 24, False)}, 2600)
        ->  {"video": {"factor": 3, "tokens": 2048}}

    Смысл OneVision ровно в этом: бюджет один на все сценарии, а геометрия
    под него подгоняется. Видео с 32 кадрами вынуждено пулить сильнее, чем
    одна картинка с 9 тайлами, — при том же итоговом числе токенов.
    """
    plan = {}
    for name, (views, patches_per_side, thumbnail) in scenarios.items():
        factor = best_pool_factor(views, patches_per_side, budget, thumbnail)
        if factor is None:
            plan[name] = None
        else:
            plan[name] = {
                "factor": factor,
                "tokens": scenario_tokens(views, patches_per_side, factor, thumbnail),
            }
    return plan


def is_valid_curriculum(order):
    """Допустим ли такой порядок стадий обучения.

    Правила из статьи: начинать обязательно с si (перцептивная база),
    стадии не повторяются, относительный порядок совпадает со STAGES.
    Пропускать стадию можно, менять местами — нет.

    is_valid_curriculum(("si", "ov", "tt"))  ->  True
    is_valid_curriculum(("si", "tt"))        ->  True   (ov пропущен)
    is_valid_curriculum(("ov", "si"))        ->  False  (видео раньше базы)
    is_valid_curriculum(())                  ->  False

    Неизвестная стадия — ValueError, а не False: опечатка в плане обучения
    не должна выглядеть как «просто неверный порядок».
    """
    order = tuple(order)
    for stage in order:
        if stage not in STAGES:
            raise ValueError(f"неизвестная стадия {stage!r}")
    if not order:
        return False
    if order[0] != STAGES[0]:
        return False
    if len(set(order)) != len(order):
        return False
    indices = [STAGES.index(s) for s in order]
    return all(a < b for a, b in zip(indices, indices[1:]))


def stage_steps(total_steps, weights):
    """Разложить шаги обучения по стадиям пропорционально весам.

    Сумма результата обязана быть РОВНО total_steps: недобор шагов из-за
    округления вниз — это молча потерянное обучение.

    stage_steps(100, {"si": 0.5, "ov": 0.3, "tt": 0.2})
        ->  {"si": 50, "ov": 30, "tt": 20}
    stage_steps(10, {"si": 1, "ov": 1, "tt": 1})
        ->  {"si": 3, "ov": 4, "tt": 3}

    Остаток раздаётся по методу наибольших остатков. При равных остатках
    порядок — по имени стадии, чтобы план обучения был воспроизводим.

    Отрицательный total_steps, пустые веса, отрицательный вес или нулевая
    сумма весов — ValueError.
    """
    if total_steps < 0:
        raise ValueError("total_steps не может быть отрицательным")
    if not weights:
        raise ValueError("пустые веса")
    if any(w < 0 for w in weights.values()):
        raise ValueError("отрицательный вес")
    total_w = sum(weights.values())
    if total_w == 0:
        raise ValueError("сумма весов равна нулю")
    exact = {k: total_steps * w / total_w for k, w in weights.items()}
    result = {k: int(v) for k, v in exact.items()}
    left = total_steps - sum(result.values())
    # ключ сортировки: сначала больший остаток, при равенстве — имя стадии
    order = sorted(weights, key=lambda k: (-(exact[k] - result[k]), k))
    for k in order[:left]:
        result[k] += 1
    return result

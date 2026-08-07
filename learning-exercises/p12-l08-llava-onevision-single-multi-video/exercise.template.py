"""
LLaVA-OneVision: одна модель на картинку, набор картинок и видео

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l08-llava-onevision-single-multi-video
Разбор:  /check-code p12-l08-llava-onevision-single-multi-video
"""

STAGES = ("si", "ov", "tt")
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
    raise NotImplementedError


def pooled_tokens(patches_per_side, pool_factor):
    """Сколько визуальных токенов останется от одной квадратной сетки патчей.

    pooled_tokens(24, 1)  ->  576   (пулинга нет)
    pooled_tokens(24, 2)  ->  144
    pooled_tokens(24, 3)  ->  64

    Зависимость квадратичная: пулинг вдвое режет токены вчетверо. Отсюда и
    берётся бюджет — один шаг пулинга освобождает место под три лишних кадра.

    Некратная сторона или неположительные аргументы — ValueError.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError

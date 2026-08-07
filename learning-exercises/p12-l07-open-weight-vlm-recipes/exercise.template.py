"""
Рецепты open-weight VLM: что реально влияет

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l07-open-weight-vlm-recipes
Разбор:  /check-code p12-l07-open-weight-vlm-recipes
"""

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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def rank_axes_by_impact(rows, metric, axes=AXES):
    """Оси по убыванию объяснённой дисперсии: что ablation'ить первым.

    Возвращает список пар (axis, share).

    rank_axes_by_impact(rows, "mmmu")[0][0]  ->  "encoder"

    При равных долях порядок сохраняется как в axes — сортировка
    устойчивая, поэтому «не хуже» не превращается в «случайно раньше».
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError

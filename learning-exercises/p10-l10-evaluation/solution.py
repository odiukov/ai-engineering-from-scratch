"""
Оценка LLM: бенчмарки, метрики, ELO — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

K_FACTOR = 32       # шаг ELO, как в шахматах и в Chatbot Arena
INITIAL_RATING = 1500.0


def exact_match(prediction, expected):
    """Точное совпадение ответа с эталоном: 1.0 или 0.0.

    exact_match("Paris", "Paris")      ->  1.0
    exact_match("  paris ", "Paris")   ->  1.0   (регистр и пробелы не в счёт)
    exact_match("Paris, France", "Paris")  ->  0.0

    Нормализуй обе строки: strip и нижний регистр. Без нормализации метрика
    штрафует за лишний пробел, а это не ошибка модели.

    Метрика беспощадна к многословности: «Paris is the capital city of France»
    получит ноль, хотя ответ верный. Именно поэтому одна метрика никогда не
    даёт полной картины — держи рядом token_f1.
    """
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0


def token_f1(prediction, expected):
    """F1 по множествам слов: гармоническое среднее точности и полноты.

    token_f1("Paris", "Paris")                   ->  1.0
    token_f1("Paris is the capital", "Paris")    ->  0.4
    token_f1("London", "Paris")                  ->  0.0
    token_f1("", "Paris")                        ->  0.0

    precision = |общие| / |слова предсказания|
    recall    = |общие| / |слова эталона|
    F1        = 2 * precision * recall / (precision + recall)

    Работаем с МНОЖЕСТВАМИ слов в нижнем регистре: повтор слова не должен
    добавлять баллов.

    Ловушки: пустая строка с любой стороны — сразу 0.0, иначе деление на ноль;
    и нулевой знаменатель при полном промахе.
    """
    pred = set(prediction.lower().split())
    exp = set(expected.lower().split())
    if not pred or not exp:
        return 0.0
    common = len(pred & exp)
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(exp)
    return 2 * precision * recall / (precision + recall)


def perplexity(log_probs):
    """Перплексия: exp от среднего отрицательного логарифма вероятности.

    perplexity([0.0, 0.0])                 ->  1.0    (модель уверена и права)
    perplexity([math.log(0.5)] * 4)        ->  2.0
    perplexity([])                         ->  inf

    Формула: exp(-mean(log_probs)). Перплексия 10 значит, что на каждом
    токене модель колеблется как между 10 равновероятными вариантами.
    Меньше — лучше.

    На вход идут ЛОГАРИФМЫ вероятностей (все <= 0), а не сами вероятности.
    Пустой список — бесконечность, а не ZeroDivisionError: нечего мерить.
    """
    if not log_probs:
        return float("inf")
    return math.exp(-sum(log_probs) / len(log_probs))


def expected_score(rating_a, rating_b):
    """Вероятность победы A над B по формуле ELO.

    expected_score(1500, 1500)  ->  0.5
    expected_score(1900, 1500)  ->  0.9090...  (разрыв в 400 — шансы 10 к 1)

    Формула: 1 / (1 + 10 ^ ((rating_b - rating_a) / 400)).

    Число 400 в знаменателе — это и есть определение шкалы ELO: разница
    в 400 пунктов означает десятикратное превосходство в шансах.

    Ловушка знака: в показателе стоит (b - a), а не (a - b). Перепутаешь —
    получишь зеркальный рейтинг, и лидер уедет вниз таблицы.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(rating_a, rating_b, outcome, k=K_FACTOR):
    """Пересчёт пары рейтингов после одного сравнения.

    outcome — строка "a", "b" или "tie". Вернуть кортеж (new_a, new_b).

    elo_update(1500, 1500, "a")    ->  (1516.0, 1484.0)   при k = 32
    elo_update(1500, 1500, "tie")  ->  (1500.0, 1500.0)

    Формула: new = old + k * (фактический счёт - ожидаемый счёт), где
    фактический счёт равен 1.0 / 0.0 / 0.5.

    Сумма двух рейтингов не меняется НИКОГДА: сколько выиграл один, столько
    проиграл другой. Это удобная проверка своей реализации.

    Любое другое значение outcome — ValueError. Молча считать неизвестный
    исход ничьёй нельзя: так опечатка в судейском коде тихо размажет
    таблицу лидеров.
    """
    scores = {"a": (1.0, 0.0), "b": (0.0, 1.0), "tie": (0.5, 0.5)}
    if outcome not in scores:
        raise ValueError(f"unknown outcome: {outcome!r}")
    score_a, score_b = scores[outcome]

    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    return (rating_a + k * (score_a - exp_a), rating_b + k * (score_b - exp_b))


def elo_tournament(matches, k=K_FACTOR, initial=INITIAL_RATING):
    """Прогнать список сравнений и вернуть таблицу рейтингов.

    matches — список кортежей (имя_a, имя_b, исход).

    elo_tournament([("gpt", "llama", "a")])
        ->  {"gpt": 1516.0, "llama": 1484.0}
    elo_tournament([])  ->  {}

    Новый участник входит с рейтингом initial. Порядок матчей ВАЖЕН: ELO
    обновляется онлайн, и те же матчи в другом порядке дадут слегка другие
    числа. Это не баг, а свойство метода.

    Так устроен Chatbot Arena: две модели, один промпт, человек выбирает
    победителя, рейтинги пересчитываются.
    """
    ratings = {}
    for name_a, name_b, outcome in matches:
        ratings.setdefault(name_a, float(initial))
        ratings.setdefault(name_b, float(initial))
        ratings[name_a], ratings[name_b] = elo_update(
            ratings[name_a], ratings[name_b], outcome, k
        )
    return ratings


def run_suite(cases, model_fn, scorers):
    """Прогнать модель по набору тестов и посчитать все метрики.

    cases — список кортежей (input_text, expected).
    model_fn — функция input_text -> prediction.
    scorers — словарь имя_метрики -> функция (prediction, expected) -> число.

    run_suite([("2+2?", "4")], lambda q: "4", {"em": exact_match})
        ->  [{"input": "2+2?", "expected": "4", "prediction": "4",
              "scores": {"em": 1.0}}]

    Модель вызывается РОВНО ОДИН РАЗ на кейс, а все метрики считаются по
    одному и тому же предсказанию. Вызовешь модель отдельно под каждую
    метрику — при ненулевой температуре метрики будут мерить разные ответы,
    и сравнивать их станет нельзя.
    """
    results = []
    for input_text, expected in cases:
        prediction = model_fn(input_text)
        results.append({
            "input": input_text,
            "expected": expected,
            "prediction": prediction,
            "scores": {name: fn(prediction, expected) for name, fn in scorers.items()},
        })
    return results


def summarize(results, threshold=0.8):
    """Свести результаты прогона в статистику по каждой метрике.

    Для каждой метрики вернуть словарь с ключами "mean", "median", "std",
    "min", "max", "pass_rate", "n".

    summarize([{"scores": {"em": 1.0}}, {"scores": {"em": 0.0}}])
        ->  {"em": {"mean": 0.5, "median": 0.5, "std": 0.5, "min": 0.0,
                    "max": 1.0, "pass_rate": 0.5, "n": 2}}
    summarize([])  ->  {}

    Детали, на которых легко ошибиться:
      * median при чётном n — среднее двух средних элементов, а не элемент;
      * std популяционная (делим на n): это вся выборка прогона, а не
        подвыборка из чего-то большего;
      * pass_rate — доля значений >= threshold, СТРОГО не меньше.

    pass_rate практичнее среднего: она отвечает «на скольких кейсах модель
    надёжна», а среднее одинаково хорошо маскирует и «везде средне»,
    и «половина отлично, половина провал».
    """
    collected = {}
    for row in results:
        for metric, score in row["scores"].items():
            collected.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in collected.items():
        n = len(scores)
        ordered = sorted(scores)
        mid = n // 2
        # чётное n: середины две, берём их среднее
        median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        summary[metric] = {
            "mean": mean,
            "median": median,
            "std": math.sqrt(var),
            "min": ordered[0],
            "max": ordered[-1],
            "pass_rate": sum(1 for s in scores if s >= threshold) / n,
            "n": n,
        }
    return summary

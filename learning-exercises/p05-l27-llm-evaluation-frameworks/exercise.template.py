"""
Оценка LLM: RAGAS, DeepEval, G-Eval

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p05-l27-llm-evaluation-frameworks
Разбор:  /check-code p05-l27-llm-evaluation-frameworks
"""

import json
import math
import re


def split_claims(answer):
    """Разбить ответ на атомарные claims — по одному предложению на claim.

    Пустые куски и пробелы выбрасываем.

    split_claims("The iPhone launched in 2007. Apple is in Cupertino.")
        ->  ["The iPhone launched in 2007.", "Apple is in Cupertino."]
    split_claims("  ")   ->  []

    Это первый шаг RAGAS-faithfulness: сначала режем ответ на утверждения,
    потом каждое проверяем против контекста по отдельности. Ответ целиком
    проверять бессмысленно — одно враньё среди четырёх фактов утонет.
    """
    raise NotImplementedError


def faithfulness(answer, context, judge):
    """Доля claims ответа, подтверждённых контекстом. RAGAS faithfulness.

    judge — заглушка вместо NLI-модели или LLM-судьи: вызывается как
    judge(claim, context) и возвращает число 0..1. Claim засчитан
    подтверждённым при значении >= 0.5.

    faithfulness("A. B.", ctx, lambda c, x: 1.0)   ->  1.0
    faithfulness("A. B.", ctx, lambda c, x: 0.0)   ->  0.0
    faithfulness("", ctx, judge)                   ->  0.0

    Ловушка: у пустого ответа claims ноль — вернуть 0.0, а не поделить на
    ноль. Пустой ответ не «идеально верен», он просто ничего не сказал.

    Модель передаётся параметром, а не зашита внутрь, ровно по той же
    причине, по которой это делает RAGAS: судью надо уметь подменить
    (заморозить версию, сравнить двух судей, прогнать тесты без сети).
    """
    raise NotImplementedError


def answer_relevance(question, answer, question_generator, similarity):
    """Насколько ответ отвечает именно на заданный вопрос. RAGAS answer relevance.

    question_generator(answer) -> список вопросов, на которые этот ответ
    похож на ответ (в продакшене это LLM, здесь — заглушка).
    similarity(a, b) -> число: близость двух строк.
    Результат — средняя близость исходного вопроса к сгенерированным.

    answer_relevance(q, a, lambda _: [q, q], sim)  ->  sim(q, q)
    answer_relevance(q, a, lambda _: [], sim)      ->  0.0

    Пустые строки от генератора выбрасываем до усреднения: LLM охотно
    отдаёт лишние переводы строки, и они утянут среднее вниз.

    Смысл метрики: если по ответу восстанавливаются совсем другие вопросы,
    значит ответ не про то, что спрашивали, — даже если он правдив.
    """
    raise NotImplementedError


def context_precision(retrieved, relevant):
    """Доля выданных ретривером чанков, которые реально были нужны.

    context_precision(["a", "b"], ["a"])        ->  0.5
    context_precision(["a"], ["a", "b"])        ->  1.0
    context_precision([], ["a"])                ->  0.0

    Ловушка: precision смотрит на знаменатель «сколько выдали». Добить
    top-k мусором до красивого числа не выйдет — метрика упадёт.

    В RAGAS precision ещё взвешена по позиции (релевантное в конце списка
    ценится меньше). Здесь собираем простую версию — она уже ловит
    основной провал ретривера.
    """
    raise NotImplementedError


def context_recall(gold_claims, retrieved, judge):
    """Доля claims эталонного ответа, покрытых выдачей ретривера.

    judge(claim, context) -> 0..1, как в faithfulness. Контекст — все
    выданные чанки, склеенные через пробел.

    context_recall(["A", "B"], ["A"], judge_substring)  ->  0.5
    context_recall([], ["A"], judge)                    ->  0.0

    Ловушка: порядок аргументов не такой, как у context_precision.
    Recall считается от ЭТАЛОНА (что должно было найтись), precision — от
    ВЫДАЧИ (что нашлось). Перепутать их — классический способ отчитаться
    об успехе на сломанном ретривере.
    """
    raise NotImplementedError


def parse_judge_score(raw):
    """Достать оценку из ответа судьи. Вернуть float 0..1 либо None.

    None — это явный признак «судья не ответил», а НЕ ноль. Ноль означал бы
    «модель ответила плохо», и провал парсинга навсегда испортил бы среднее.

    parse_judge_score('{"score": 0.8}')                 ->  0.8
    parse_judge_score('Sure!\\n```json\\n{"score": 1}```')  ->  1.0
    parse_judge_score('score is high')                  ->  None
    parse_judge_score('{"score": 1.5}')                 ->  None

    Ловушки:
      * модель любит обрамлять JSON текстом и ``` — режем от первой `{`
        до последней `}`;
      * `True` в Python — подкласс int, и без отдельной проверки
        {"score": true} превратился бы в 1.0.
    """
    raise NotImplementedError


def aggregate_scores(scores, q=0.1):
    """Свернуть прогон в отчёт: среднее, хвост и число провалов парсинга.

    scores — список оценок, где None означает «судья не дал числа».
    Возвращает dict с ключами:
      "mean"        — среднее по валидным (0.0, если валидных нет);
      "bottom_mean" — среднее по худшей доле q валидных, минимум один
                      элемент;
      "valid"       — сколько оценок засчитано;
      "failed"      — сколько было None.

    aggregate_scores([1.0, 0.0])        ->  mean 0.5, bottom_mean 0.0
    aggregate_scores([1.0, None])       ->  mean 1.0, valid 1, failed 1

    Ловушки:
      * None нельзя считать нулём — среднее поедет вниз на ровном месте;
      * и нельзя молча выкидывать — поэтому failed отдельным числом.

    Зачем bottom_mean: среднее 0.85 спокойно прячет 5% катастроф. Смотреть
    надо на нижний квантиль, иначе релиз с редким, но грубым враньём
    пройдёт гейт.
    """
    raise NotImplementedError


def spearman_rho(judge_scores, human_scores):
    """Ранговая корреляция Спирмена: насколько судья согласен с человеком.

    Считается как корреляция Пирсона по РАНГАМ, у одинаковых значений ранг
    усредняется. Если у одного из списков нулевой разброс, корреляция не
    определена — возвращаем 0.0.

    spearman_rho([1, 2, 3], [10, 20, 30])   ->  1.0
    spearman_rho([1, 2, 3], [30, 20, 10])   ->  -1.0
    spearman_rho([1, 1, 1], [1, 2, 3])      ->  0.0

    Списки разной длины — ValueError.

    Это и есть калибровка судьи из урока: пока rho против ручной разметки
    ниже 0.7, число, которое отдаёт судья, — шум, и полагаться на него в CI
    нельзя.
    """
    raise NotImplementedError

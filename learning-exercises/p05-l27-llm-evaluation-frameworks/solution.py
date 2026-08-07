"""
Оценка LLM: RAGAS, DeepEval, G-Eval — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
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
    # (?<=[.!?]) — режем ПОСЛЕ знака конца предложения, сам знак остаётся
    # в claim: судье-заглушке и NLI-модели он не мешает, а текст остаётся
    # читаемым в отчёте.
    parts = re.split(r"(?<=[.!?])\s+", answer)
    return [p.strip() for p in parts if p.strip()]


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
    claims = split_claims(answer)
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if judge(c, context) >= 0.5)
    return supported / len(claims)


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
    generated = [q for q in question_generator(answer) if q.strip()]
    if not generated:
        return 0.0
    return sum(similarity(question, g) for g in generated) / len(generated)


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
    if not retrieved:
        return 0.0
    gold = set(relevant)
    hits = sum(1 for c in retrieved if c in gold)
    return hits / len(retrieved)


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
    if not gold_claims:
        return 0.0
    context = " ".join(retrieved)
    covered = sum(1 for c in gold_claims if judge(c, context) >= 0.5)
    return covered / len(gold_claims)


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
    if not isinstance(raw, str):
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except ValueError:  # JSONDecodeError — подкласс ValueError
        return None
    if not isinstance(data, dict):
        return None
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = float(score)
    if not 0.0 <= score <= 1.0:
        return None
    return score


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
    valid = [s for s in scores if s is not None]
    failed = len(scores) - len(valid)
    if not valid:
        return {"mean": 0.0, "bottom_mean": 0.0, "valid": 0, "failed": failed}
    # ceil, а не round: при q=0.1 и десяти примерах в хвост обязан попасть
    # хотя бы один — иначе метрика хвоста просто исчезнет на малых прогонах
    k = max(1, math.ceil(q * len(valid)))
    bottom = sorted(valid)[:k]
    return {
        "mean": sum(valid) / len(valid),
        "bottom_mean": sum(bottom) / len(bottom),
        "valid": len(valid),
        "failed": failed,
    }


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
    if len(judge_scores) != len(human_scores):
        raise ValueError("списки разной длины")
    n = len(judge_scores)
    if n == 0:
        return 0.0

    def ranks(xs):
        # средний ранг для связок: без этого [1,1,2] дало бы разным
        # одинаковым значениям разные ранги, и корреляция зависела бы
        # от порядка во входном списке
        order = sorted(range(n), key=lambda i: xs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rj, rh = ranks(judge_scores), ranks(human_scores)
    mj, mh = sum(rj) / n, sum(rh) / n
    cov = sum((a - mj) * (b - mh) for a, b in zip(rj, rh))
    var_j = sum((a - mj) ** 2 for a in rj)
    var_h = sum((b - mh) ** 2 for b in rh)
    if var_j == 0 or var_h == 0:
        return 0.0
    return cov / math.sqrt(var_j * var_h)

"""
Evaluation: метрики, доверительные интервалы, регрессионное сравнение — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random
import re

# Токен: слово или число, апостроф внутри слова сохраняется ("don't" целиком).
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")

# Оценка 4 и выше считается зачётом — по этому порогу считается pass rate.
PASSING_SCORE = 4


def normalize_tokens(text):
    """Привести текст к списку токенов: нижний регистр, без пунктуации.

    normalize_tokens("The capital of France is Paris.")
        ->  ['the', 'capital', 'of', 'france', 'is', 'paris']
    normalize_tokens("Paris!!!")  ->  ['paris']

    Без нормализации "Paris." и "Paris" — разные токены, и любая метрика
    совпадения занижает счёт на ровном месте. Одна и та же нормализация
    обязана применяться и к эталонному ответу, и к ответу модели.
    """
    return TOKEN_RE.findall(text.lower())


def lcs_length(a, b):
    """Длина наибольшей общей ПОДПОСЛЕДОВАТЕЛЬНОСТИ двух списков токенов.

    lcs_length(['a', 'b', 'c'], ['a', 'c'])       ->  2
    lcs_length(['a', 'x', 'b'], ['a', 'b'])       ->  2   (пропуски разрешены)
    lcs_length(['a', 'b'], ['b', 'a'])            ->  1   (порядок важен)
    lcs_length([], ['a'])                          ->  0

    Подпоследовательность, а не подстрока: элементы должны идти в том же
    ПОРЯДКЕ, но не обязаны стоять подряд. Именно поэтому ROUGE-L мягче
    точного совпадения и строже мешка слов.

    Классическая динамика: dp[i][j] — ответ для префиксов. Хранить всю
    таблицу не нужно, хватает предыдущей строки — так память O(min) вместо
    O(n*m), а на длинных ответах модели это заметно.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                cur.append(prev[j] + 1)
            else:
                cur.append(max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(reference, hypothesis):
    """ROUGE-L: F1 по длине наибольшей общей подпоследовательности.

    rouge_l("the cat sat", "the cat sat")   ->  1.0
    rouge_l("the cat sat", "a dog ran")     ->  0.0
    rouge_l("the cat sat", "the sat cat")   ->  примерно 0.6667

    precision = LCS / длина гипотезы, recall = LCS / длина эталона,
    F1 = 2 * p * r / (p + r).

    Метрика симметрична: поменяй эталон и гипотезу местами — precision и
    recall поменяются местами, а F1 останется тем же.

    Чем ROUGE-L отличается от мешка слов: перестановка слов её роняет.
    "the sat cat" и "the cat sat" состоят из одних и тех же слов, но
    подпоследовательность у них короче полной длины.

    Честная оговорка: два ответа могут быть оба верны и не иметь общих слов
    вовсе. ROUGE-L даст 0.0 и будет формально права и содержательно нет —
    поэтому её и не используют как единственную метрику.
    """
    ref = normalize_tokens(reference)
    hyp = normalize_tokens(hypothesis)
    if not ref or not hyp:
        return 0.0
    lcs = lcs_length(ref, hyp)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def jaccard_overlap(reference, hypothesis):
    """Мешок слов: доля общих уникальных токенов от объединения.

    jaccard_overlap("the cat sat", "the sat cat")  ->  1.0   (порядка не видит)
    jaccard_overlap("the cat", "the dog")          ->  примерно 0.3333
    jaccard_overlap("", "anything")                ->  0.0

    Повторы не считаются: работаем с множествами. Пустое объединение даёт
    0.0, а не деление на ноль.

    Держи эту метрику рядом с ROUGE-L: разница между ними и показывает,
    насколько в задаче важен порядок слов.
    """
    ref = set(normalize_tokens(reference))
    hyp = set(normalize_tokens(hypothesis))
    union = ref | hyp
    if not union:
        return 0.0
    return len(ref & hyp) / len(union)


def wilson_interval(successes, total, z=1.96):
    """Доверительный интервал Вильсона для доли. Вернуть (lower, upper).

    wilson_interval(45, 50)   ->  примерно (0.7864, 0.9565)
    wilson_interval(0, 0)     ->  (0.0, 0.0)

    Формула:
        center = (p + z^2 / (2n)) / (1 + z^2 / n)
        spread = z * sqrt((p(1-p) + z^2 / (4n)) / n) / (1 + z^2 / n)

    Зачем не обычное p +- z*sqrt(p(1-p)/n): при p = 1.0 наивная формула даёт
    ширину РОВНО НОЛЬ, то есть "20 из 20 — значит ровно 100%, сомнений нет".
    Вильсон в том же случае честно оставляет нижнюю границу заметно ниже
    единицы. На маленьких выборках, а их в eval-наборах большинство, это
    разница между решением и самообманом.

    Границы обрезаются в [0, 1]: доля не бывает отрицательной.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def bootstrap_interval(scores, seed=0, n_bootstrap=1000, confidence=0.95):
    """Бутстрэп-интервал для среднего. Вернуть (lower, mean, upper).

    bootstrap_interval([4, 4, 4, 4], seed=0)  ->  (4.0, 4.0, 4.0)
    bootstrap_interval([], seed=0)            ->  (0.0, 0.0, 0.0)

    Идея: n_bootstrap раз набрать выборку того же размера С ВОЗВРАЩЕНИЕМ,
    посчитать среднее каждой, отсортировать и взять перцентили. Никаких
    предположений о распределении оценок судьи — а оно у оценок 1-5
    заведомо не нормальное.

    seed обязателен: без него один и тот же прогон eval даёт разные границы,
    и сравнить две версии промпта невозможно. random.Random(seed), а не
    глобальный random.

    Возвращается ещё и точное среднее — оно считается по исходным оценкам,
    а не по бутстрэпу.
    """
    n = len(scores)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(scores) / n
    if n == 1:
        return (float(scores[0]), mean, float(scores[0]))

    rng = random.Random(seed)
    means = sorted(sum(rng.choice(scores) for _ in range(n)) / n for _ in range(n_bootstrap))
    alpha = (1 - confidence) / 2
    lower = means[int(alpha * n_bootstrap)]
    upper = means[min(n_bootstrap - 1, int((1 - alpha) * n_bootstrap) - 1)]
    return (lower, mean, upper)


def compare_runs(baseline, new, threshold=0.3):
    """Сравнить два прогона eval и решить, катить ли изменение.

    baseline и new — словари {критерий: список оценок}.
    Вернуть {"criteria", "regressions", "improvements", "overall"}.

    compare_runs({"relevance": [4, 4]}, {"relevance": [5, 5]})
        ->  улучшение на 1.0, ship_decision = 'SHIP'
    compare_runs({"safety": [5, 5]}, {"safety": [3, 3]})
        ->  ship_decision = 'BLOCK'

    По каждому критерию: средние до и после, разница, число зачётов
    (оценка >= 4) и интервал Вильсона для новой доли зачётов.

    Статус: разница меньше -threshold — REGRESSION, больше +threshold —
    IMPROVED, иначе STABLE. Порог нужен, чтобы шум судьи не выглядел
    прогрессом: 4.21 против 4.19 — это ничего, а не улучшение.

    Решение блокирующее: ЛЮБАЯ регрессия даёт BLOCK, даже если в среднем
    стало лучше. Именно так система "улучшается" в целом и разваливается
    на safety — общее среднее эту дыру прячет, а разбивка по критериям нет.

    Критерии, которых нет в обоих прогонах, пропускаются: сравнивать не с чем.
    """
    report = {"criteria": {}, "regressions": [], "improvements": [], "overall": {}}
    all_baseline, all_new = [], []

    for criterion in baseline:
        if criterion not in new or not baseline[criterion] or not new[criterion]:
            continue
        old_scores, new_scores = baseline[criterion], new[criterion]
        old_mean = sum(old_scores) / len(old_scores)
        new_mean = sum(new_scores) / len(new_scores)
        diff = new_mean - old_mean

        if diff < -threshold:
            status = "REGRESSION"
            report["regressions"].append(criterion)
        elif diff > threshold:
            status = "IMPROVED"
            report["improvements"].append(criterion)
        else:
            status = "STABLE"

        new_passing = sum(1 for s in new_scores if s >= PASSING_SCORE)
        report["criteria"][criterion] = {
            "baseline_mean": old_mean,
            "new_mean": new_mean,
            "diff": diff,
            "baseline_passing": sum(1 for s in old_scores if s >= PASSING_SCORE),
            "new_passing": new_passing,
            "new_pass_ci": wilson_interval(new_passing, len(new_scores)),
            "status": status,
        }
        all_baseline.extend(old_scores)
        all_new.extend(new_scores)

    report["overall"] = {
        "baseline_mean": sum(all_baseline) / len(all_baseline) if all_baseline else 0.0,
        "new_mean": sum(all_new) / len(all_new) if all_new else 0.0,
        "n_criteria": len(report["criteria"]),
        "ship_decision": "BLOCK" if report["regressions"] else "SHIP",
    }
    report["overall"]["diff"] = report["overall"]["new_mean"] - report["overall"]["baseline_mean"]
    return report

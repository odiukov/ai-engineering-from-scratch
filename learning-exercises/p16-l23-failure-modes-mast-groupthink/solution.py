"""
Режимы отказов: MAST, groupthink, каскады — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Три корневые категории MAST (Cemri et al., NeurIPS 2025, arXiv:2503.13657)
# с их долями по 1642 разобранным трассам. Порядок кортежа — порядок
# приоритета при разборе ничьих: чаще встречается — старше.
MAST_CATEGORIES = ("spec", "coord", "verify")
MAST_PRIOR = {"spec": 0.4177, "coord": 0.3694, "verify": 0.2130}

# Симптом -> корневая категория. В проде такой словарь растёт из разбора
# инцидентов; здесь взяты примеры прямо из статьи.
SYMPTOMS = {
    "role_ambiguity": "spec",
    "task_underspecified": "spec",
    "implicit_success_criteria": "spec",
    "state_drift": "coord",
    "lost_message": "coord",
    "unsynchronized_write": "coord",
    "unchecked_output": "verify",
    "memory_poisoning": "verify",
    "missing_regression_tests": "verify",
}


class UnknownIncident(Exception):
    """Ни один симптом инцидента не лёг в таксономию.

    СВОЙ класс, а не RuntimeError. NotImplementedError наследуется от
    RuntimeError, и `pytest.raises(RuntimeError)` зеленел бы на пустой
    заготовке. В уроке про отказы это была бы особенно смешная ошибка.
    """


def classify_incident(symptoms):
    """Корневая категория MAST по списку симптомов инцидента.

    classify_incident(["role_ambiguity"])                    ->  "spec"
    classify_incident(["state_drift", "lost_message"])        ->  "coord"
    classify_incident(["role_ambiguity", "state_drift"])      ->  "spec"
    classify_incident(["agents_went_quiet"])                  ->  UnknownIncident

    Побеждает категория с наибольшим числом совпавших симптомов. При
    равенстве — та, что чаще встречается в статье: spec > coord > verify.

    Неизвестный симптом не «игнорируется молча»: инцидент, который не лёг в
    таксономию, — это сигнал, что таксономию пора дополнять, а не что
    отказа не было.
    """
    counts = {c: 0 for c in MAST_CATEGORIES}
    for symptom in symptoms:
        category = SYMPTOMS.get(symptom)
        if category is not None:
            counts[category] += 1
    best = max(counts.values())
    if best == 0:
        raise UnknownIncident("симптомы вне таксономии MAST: %r" % (list(symptoms),))
    # MAST_CATEGORIES уже в порядке приоритета — первый максимум и берём
    for category in MAST_CATEGORIES:
        if counts[category] == best:
            return category


def category_rates(incidents):
    """Доли категорий по списку инцидентов. Вернуть {категория: доля}.

    Каждый инцидент — список симптомов. В результате присутствуют ВСЕ три
    категории, даже нулевые: пустая строка в отчёте — тоже информация.

    category_rates([["role_ambiguity"], ["state_drift"], ["role_ambiguity"]])
        ->  {"spec": 0.666..., "coord": 0.333..., "verify": 0.0}
    category_rates([])  ->  {"spec": 0.0, "coord": 0.0, "verify": 0.0}

    Это шаг 3 квартального аудита отказов: посчитать, какие категории
    доминируют ИМЕННО в твоей системе, а не в статье.
    """
    counts = {c: 0 for c in MAST_CATEGORIES}
    for symptoms in incidents:
        counts[classify_incident(symptoms)] += 1
    total = len(incidents)
    if total == 0:
        return {c: 0.0 for c in MAST_CATEGORIES}
    return {c: counts[c] / total for c in MAST_CATEGORIES}


def rank_mitigations(rates, mitigations):
    """Отранжировать меры по доле отказов, которую они закрывают.

    mitigations — список пар (название, категории, которые мера закрывает).
    Вернуть список пар (название, покрытая доля) по убыванию доли, при
    равенстве — по алфавиту.

    rank_mitigations({"spec": 0.5, "coord": 0.3, "verify": 0.2},
                     [("verifier-agent", ["verify"]),
                      ("role-contracts", ["spec"])])
        ->  [("role-contracts", 0.5), ("verifier-agent", 0.2)]

    Шаг 4 аудита. Ранжировать надо по доле отказов, а не по «насколько
    мера кажется правильной»: интуиция систематически переоценивает
    verification gap, потому что его отказы больнее, хоть их и меньше.
    """
    scored = [
        (name, sum(rates.get(c, 0.0) for c in categories))
        for name, categories in mitigations
    ]
    # -score для убывания, name для стабильного алфавитного порядка внутри равных
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def opinion_spread(opinions):
    """Разброс мнений: стандартное отклонение по выборке целиком.

    opinion_spread([0.0, 1.0])       ->  0.5
    opinion_spread([1.0, 1.0, 1.0])  ->  0.0
    opinion_spread([])               ->  0.0

    Делим на n, а не на n-1: нас интересует разброс именно этих агентов,
    а не оценка какой-то генеральной совокупности.

    Это самый дешёвый прокси на монокультуру: три агента на одной базовой
    модели дают нулевой разброс задолго до того, как ошибка станет видна.
    """
    n = len(opinions)
    if n < 2:
        return 0.0
    mean = sum(opinions) / n
    return math.sqrt(sum((o - mean) ** 2 for o in opinions) / n)


def detect_groupthink(rounds, min_drop=0.5, min_rise=0.1):
    """Номер раунда, где сошёлся groupthink, или None.

    rounds — список пар (мнения, уверенности) по раундам обсуждения.
    Признак: разброс мнений упал минимум на min_drop от первого раунда,
    а средняя уверенность выросла минимум на min_rise.

    rounds = [([0.0, 0.5, 1.0], [0.5, 0.5, 0.5]),
              ([0.4, 0.5, 0.6], [0.7, 0.7, 0.7])]
    detect_groupthink(rounds)  ->  1

    Именно СОЧЕТАНИЕ падающего разброса с растущей уверенностью. Разброс
    сам по себе падает и при здоровом сходе к правильному ответу; растущая
    уверенность на фоне схлопывания мнений — это уже конформизм.

    Ловушка: сравнивать надо с первым раундом, а не с предыдущим.
    Медленное сползание по чуть-чуть иначе никогда не превысит порог.
    """
    if not rounds:
        return None
    base_spread = opinion_spread(rounds[0][0])
    base_confidence = sum(rounds[0][1]) / len(rounds[0][1]) if rounds[0][1] else 0.0
    for i in range(1, len(rounds)):
        opinions, confidences = rounds[i]
        spread = opinion_spread(opinions)
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if (spread <= base_spread * (1.0 - min_drop)
                and confidence >= base_confidence + min_rise):
            return i
    return None


def circuit_state(results, threshold, min_calls):
    """Состояние предохранителя: "open" или "closed".

    results — список булевых исходов недавних вызовов (True = успех).
    Открываем, когда вызовов накопилось не меньше min_calls И доля ошибок
    строго больше threshold.

    circuit_state([True] * 9 + [False], 0.05, 5)  ->  "open"
    circuit_state([False, False], 0.05, 5)        ->  "closed"
    circuit_state([True] * 20, 0.05, 5)           ->  "closed"

    min_calls обязателен: без него первый же неудачный вызов даёт долю
    ошибок 1.0 и вырубает исправный сервис на старте.
    """
    if len(results) < min_calls:
        return "closed"
    failures = sum(1 for ok in results if not ok)
    return "open" if failures / len(results) > threshold else "closed"


def cascade_load(base_load, failure_rate, max_retries, depth, breaker_cap=None):
    """Нагрузка по уровням каскада ретраев. Вернуть список длины depth + 1.

    На каждом уровне доля failure_rate падает, каждая упавшая повторяется
    max_retries раз, и каждый повтор бьёт по следующему уровню:
    L[k+1] = L[k] * (1 + failure_rate * max_retries).
    С breaker_cap нагрузка на каждом уровне обрезается этим потолком.

    cascade_load(100.0, 0.5, 2, 2)             ->  [100.0, 200.0, 400.0]
    cascade_load(100.0, 0.5, 2, 2, breaker_cap=150.0)  ->  [100.0, 150.0, 150.0]

    Это retry storm: платёж падает на 10%, заказ ретраит, каждый ретрай —
    новая проверка склада, склад начинает таймаутить, и дальше по кругу.
    Рост геометрический, поэтому «просто добавить реплик» не спасает.

    Предохранитель — единственная мера, которую сюда переносят из
    распределённых систем вообще без адаптации.
    """
    loads = [float(base_load)]
    amplification = 1.0 + failure_rate * max_retries
    for _ in range(depth):
        nxt = loads[-1] * amplification
        if breaker_cap is not None:
            nxt = min(nxt, breaker_cap)
        loads.append(nxt)
    return loads


def audit(incidents, mitigations, top_k=3):
    """Аудит отказов целиком: трассы -> категории -> ранжированные меры.

    audit([["role_ambiguity"], ["role_ambiguity"], ["unchecked_output"]],
          [("role-contracts", ["spec"]), ("verifier-agent", ["verify"])], 1)
        ->  [("role-contracts", 0.666...)]

    Шаги 2-5 квартального аудита в одной функции. Дисциплина тут важнее
    конкретных мер: без регулярного аудита отказы растворяются в шуме и
    никогда не чинятся системно.
    """
    return rank_mitigations(category_rates(incidents), mitigations)[:top_k]

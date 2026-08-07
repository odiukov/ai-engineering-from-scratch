"""
Режимы отказов: MAST, groupthink, каскады

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l23-failure-modes-mast-groupthink
Разбор:  /check-code p16-l23-failure-modes-mast-groupthink
"""

import math

MAST_CATEGORIES = ("spec", "coord", "verify")
MAST_PRIOR = {"spec": 0.4177, "coord": 0.3694, "verify": 0.2130}
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
    pass


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def audit(incidents, mitigations, top_k=3):
    """Аудит отказов целиком: трассы -> категории -> ранжированные меры.

    audit([["role_ambiguity"], ["role_ambiguity"], ["unchecked_output"]],
          [("role-contracts", ["spec"]), ("verifier-agent", ["verify"])], 1)
        ->  [("role-contracts", 0.666...)]

    Шаги 2-5 квартального аудита в одной функции. Дисциплина тут важнее
    конкретных мер: без регулярного аудита отказы растворяются в шуме и
    никогда не чинятся системно.
    """
    raise NotImplementedError

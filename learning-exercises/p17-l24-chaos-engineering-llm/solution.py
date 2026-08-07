"""
Chaos engineering для LLM-продакшена — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Базовый уровень ошибок при SLO 99.9%: 0.05% запросов падают и без всякого
# хаоса. Burn-rate считается относительно него, а не относительно нуля.
EXPECTED_ERROR_RATE = 0.0005

# Пороги safety plane из урока: гасим эксперимент при burn-rate > 2x
# ожидаемого И blast radius > 20% трафика.
MAX_BURN = 2.0
MAX_BLAST = 0.2


class ChaosError(Exception):
    """Сервис не смог обслужить запрос: живых реплик не осталось.

    Свой класс заведён сознательно. NotImplementedError наследуется от
    RuntimeError, поэтому тест вида pytest.raises(RuntimeError) прошёл бы
    зелёным на пустой заготовке и не проверил бы ровным счётом ничего.
    """


def burn_rate(induced_error_rate, expected_error_rate=EXPECTED_ERROR_RATE):
    """Во сколько раз эксперимент жжёт бюджет ошибок быстрее ожидаемого.

    burn_rate(0.002, 0.0005)   ->  4.0
    burn_rate(0.0005, 0.0005)  ->  1.0
    burn_rate(0.0, 0.0005)     ->  0.0

    Ловушка: expected_error_rate приходит из конфига и вполне может оказаться
    нулём (SLO ещё не заполнили). Деление на ноль уронит safety plane ровно
    в тот момент, когда он нужнее всего, — подставь нижнюю границу.

    В проде это число смотрит on-call: burn-rate 30x значит, что месячный
    бюджет ошибок сгорит за сутки.
    """
    # max, а не проверка на ноль отдельной веткой: короче и не даёт
    # отрицательному значению из кривого конфига перевернуть знак.
    return induced_error_rate / max(expected_error_rate, 1e-9)


def should_abort(burn, blast_radius, max_burn=MAX_BURN, max_blast=MAX_BLAST):
    """Гейт safety plane: пора ли гасить эксперимент.

    Гасим, только когда обе беды сразу: и бюджет горит быстрее max_burn,
    и задет кусок трафика больше max_blast.

    should_abort(30.0, 0.30)  ->  True
    should_abort(80.0, 0.10)  ->  False   (жжёт страшно, но радиус 10%)
    should_abort(1.0, 0.90)   ->  False   (радиус огромный, но ошибок нет)

    Второй пример — не опечатка. Смысл blast radius в том и есть: узкий
    эксперимент доигрывают до конца, чтобы увидеть, чем он кончится.
    """
    return burn > max_burn and blast_radius > max_blast


def inject_failures(n_requests, failure_prob, rng):
    """Инжектор отказов: список длины n_requests, True — попытка отказала.

    inject_failures(4, 0.0, random.Random(0))  ->  [False, False, False, False]
    inject_failures(4, 1.0, random.Random(0))  ->  [True, True, True, True]

    rng приходит параметром (random.Random(seed)), а не берётся из глобального
    random. Эксперимент обязан воспроизводиться до последнего запроса, иначе
    постмортем превращается в гадание.
    """
    # random() отдаёт число из [0, 1): при prob=0 условие ложно всегда,
    # при prob=1 — истинно всегда. Границы работают без отдельных веток.
    return [rng.random() < failure_prob for _ in range(n_requests)]


def route_request(replica_health, request_index):
    """Round-robin роутер: какой реплике достанется запрос request_index.

    Роутер про здоровье реплик НЕ знает — раздаёт по кругу, как L4-балансировщик
    без health-check.

    route_request([True, False, True], 0)  ->  0
    route_request([True, False, True], 1)  ->  1   (да, в мёртвую)
    route_request([True, False, True], 4)  ->  1

    Если живых реплик не осталось вовсе — обслуживать некому. Это уже не
    деградация, а падение: подними ChaosError. Обычный RuntimeError тут не
    годится, см. комментарий у класса.
    """
    if not replica_health or not any(replica_health):
        raise ChaosError("no live replica left")
    return request_index % len(replica_health)


def serve_request(replica_health, request_index, injected_fail, retry_enabled):
    """Одна попытка обслужить запрос; при retry_enabled — с перезаходом.

    Вернуть кортеж (обслужен, сколько попыток потрачено).

    serve_request([True, True, True], 0, False, False)   ->  (True, 1)
    serve_request([True, False, True], 1, False, False)  ->  (False, 1)
    serve_request([True, False, True], 1, False, True)   ->  (True, 2)

    injected_fail — отказ самой попытки (провайдер ответил 429). Он не зависит
    от здоровья реплики, и ретрай его тоже лечит: следующая по кругу живая
    реплика попробует ещё раз.

    Ретраить больше len(replica_health) раз бессмысленно — круг замкнётся.
    Именно так и получают retry storm: бесконечные ретраи умножают нагрузку
    на мёртвый кластер.
    """
    target = route_request(replica_health, request_index)
    if replica_health[target] and not injected_fail:
        return (True, 1)
    if not retry_enabled:
        return (False, 1)
    n = len(replica_health)
    # шагаем по кругу, начиная со следующей реплики; первая живая забирает
    # запрос, потраченных попыток ровно 1 + step
    for step in range(1, n):
        if replica_health[(target + step) % n]:
            return (True, 1 + step)
    return (False, n)


def run_scenario(replica_health, n_requests, failure_prob, retry_enabled, rng):
    """Прогнать сценарий целиком и вернуть отчёт о деградации.

    Ключи отчёта: requests, served, failed, error_rate, attempts.

    run_scenario([True, True, True], 9, 0.0, False, random.Random(0))
        ->  {'requests': 9, 'served': 9, 'failed': 0, 'error_rate': 0.0,
             'attempts': 9}

    Одна мёртвая реплика из трёх без ретраев теряет треть запросов; с ретраями
    не теряет ни одного, но платит лишними попытками. Ровно эту разницу
    эксперимент и должен показать: система деградирует, а не падает.

    Если живых реплик нет — ChaosError пролетает наружу нетронутым. Глушить
    его тут нельзя: «сервис лёг» — это результат эксперимента, а не помеха.
    """
    fails = inject_failures(n_requests, failure_prob, rng)
    served = 0
    attempts = 0
    for i in range(n_requests):
        ok, tries = serve_request(replica_health, i, fails[i], retry_enabled)
        served += 1 if ok else 0
        attempts += tries
    failed = n_requests - served
    return {
        "requests": n_requests,
        "served": served,
        "failed": failed,
        # пустой прогон — это 0% ошибок, а не ZeroDivisionError
        "error_rate": failed / n_requests if n_requests else 0.0,
        "attempts": attempts,
    }


def experiment_report(name, scenario, blast_radius,
                      expected_error_rate=EXPECTED_ERROR_RATE):
    """Свести результат сценария с гейтом safety plane в строку отчёта.

    Ключи: experiment, error_rate, burn_rate_x, blast_radius, aborted, status.

    experiment_report("provider 429", {'error_rate': 0.015}, 0.30)
        ->  burn_rate_x 30.0, aborted True,  status 'ABORTED (burn-rate guard)'
    experiment_report("tokenizer stall", {'error_rate': 0.040}, 0.10)
        ->  burn_rate_x 80.0, aborted False, status 'COMPLETED'

    Функция ничего не считает сама — она склеивает burn_rate и should_abort.
    Так порог живёт в одном месте, а не расползается копиями по отчётам.
    """
    burn = burn_rate(scenario["error_rate"], expected_error_rate)
    aborted = should_abort(burn, blast_radius)
    return {
        "experiment": name,
        "error_rate": scenario["error_rate"],
        "burn_rate_x": burn,
        "blast_radius": blast_radius,
        "aborted": aborted,
        "status": "ABORTED (burn-rate guard)" if aborted else "COMPLETED",
    }

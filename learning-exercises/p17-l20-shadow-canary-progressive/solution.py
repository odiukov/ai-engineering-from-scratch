"""
Shadow-трафик, канарейка и постепенная выкатка — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Argo Rollouts, Flagger и Istio weighted routing делают это конфигом. Здесь
мы собираем механику руками. Соответствие настоящему продукту:

    bucket_of          <-  консистентное хеширование пользователя, оно же
                           "sticky bucketing" в LaunchDarkly / Flagsmith
    assign_variant     <-  weighted routing: доля трафика -> ветка
    split_traffic      <-  сам сплит на шаге канарейки
    shadow_call        <-  shadow mode: копия запроса в кандидата, ответ
                           пользователю всегда от baseline
    gate_breaches      <-  analysis template в Argo Rollouts: пять метрик
    widen_gates_for_noise  <-  поправка на LLM-недетерминизм (до 15% разброса)
    run_canary         <-  сама прогрессия 1% -> 10% -> ... -> 100% с гейтами
    rollback_policy    <-  откат флагом: секунды, а не передеплой

Времени нет: этап канарейки приходит номером, метрики — функцией measure.
Случайности нет нигде, кроме тестов, где она нужна для контраста с
устойчивым хешем: rng всегда приходит параметром.
"""

import hashlib

# Прогрессия из урока (docs/en.md, «Canary rollout»).
STAGES = (0.01, 0.10, 0.25, 0.50, 0.75, 1.00)

# Базовые значения продакшена, с которыми сравнивается кандидат.
BASELINE = {
    "latency_p99_ms": 900.0,
    "cost_per_req": 0.02,
    "error_rate": 0.02,
    "output_len_p99": 450.0,
    "thumbs_down_rate": 0.03,
}

# Множители-гейты из урока: во сколько раз метрика кандидата может
# превысить базовую, прежде чем выкатка останавливается.
GATES = {
    "latency_p99_ms": 1.5,
    "cost_per_req": 1.2,
    "error_rate": 2.0,
    "output_len_p99": 1.4,
    "thumbs_down_rate": 1.5,
}


def bucket_of(user_id, salt=""):
    """Устойчивое число в [0, 1) для пользователя. Одинаковое между запросами.

    bucket_of("u-1") == bucket_of("u-1")     ->  True (всегда, в любом процессе)
    0.0 <= bucket_of("u-1") < 1.0            ->  True
    bucket_of("u-1") != bucket_of("u-1", "b")->  True (соль меняет раскладку)

    Берём sha256 от salt и user_id, первые 8 байт как целое, делим на 2**64.

    Почему не random.random(): он даёт новое число на каждый вызов. Тот же
    пользователь на следующем запросе уехал бы в другую ветку, увидел бы
    другую модель в середине диалога, а метрики веток перемешались бы.
    Устойчивый хеш — не оптимизация, а условие того, чтобы сравнение веток
    вообще что-то значило.

    Почему не hash(): встроенный hash строк рандомизирован от запуска к
    запуску (PYTHONHASHSEED), и после рестарта пода раскладка поехала бы.
    """
    digest = hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def assign_variant(user_id, canary_share, salt=""):
    """Ветка пользователя на данном шаге: "canary" или "baseline".

    assign_variant("u-1", 0.0)  ->  "baseline"  (доля 0 — канарейки нет)
    assign_variant("u-1", 1.0)  ->  "canary"    (доля 1 — все внутри)

    Правило одно: bucket_of(user) < canary_share.

    Отсюда бесплатно получается главное свойство ступенчатой выкатки: при
    росте доли пользователь может только ДОБАВИТЬСЯ в канарейку, но не
    выпасть из неё. Тот, кто попал в канарейку на 10%, останется в ней и на
    25%. Если бы порог сравнивался со свежим случайным числом, состав
    канарейки перетасовывался бы на каждом шаге, и «метрика ухудшилась»
    было бы не отличить от «поменялась выборка».

    canary_share вне [0, 1] — ValueError: доля 1.5 не значит ничего, а
    молча означала бы «все», скрыв ошибку в конфиге прогрессии.
    """
    if not 0.0 <= canary_share <= 1.0:
        raise ValueError(f"canary_share must be in [0, 1], got {canary_share}")
    return "canary" if bucket_of(user_id, salt) < canary_share else "baseline"


def split_traffic(user_ids, canary_share, salt=""):
    """Разложить пользователей по веткам. Вернуть (canary, baseline) кортежами.

    split_traffic(["u-1", "u-2"], 1.0)  ->  (("u-1", "u-2"), ())
    split_traffic(["u-1", "u-2"], 0.0)  ->  ((), ("u-1", "u-2"))

    Порядок внутри каждой ветки сохраняется входной — отчёт должен читаться
    так же, как читался список.

    Фактическая доля канарейки почти никогда не равна заданной: на 1000
    пользователей и доле 10% нормально получить 96 или 104. Это не баг
    хеша, это конечная выборка. Отклонение, которое НЕ объясняется
    выборкой, называется SRM (sample ratio mismatch) и означает поломку
    механизма назначения.
    """
    canary, baseline = [], []
    for user_id in user_ids:
        (canary if assign_variant(user_id, canary_share, salt) == "canary" else baseline).append(user_id)
    return tuple(canary), tuple(baseline)


def shadow_call(request, baseline_provider, candidate_provider):
    """Shadow mode: кандидат получает копию запроса, пользователь — ответ baseline.

    Вернуть (user_response, record), где record — то, что уйдёт в лог:
        {"baseline_tokens", "candidate_tokens", "token_delta",
         "same_output", "candidate_error"}

    shadow_call(req, лучший, дороже_на_40_процентов)
        ->  ответ пользователю от baseline, token_delta > 0 в записи

    Провайдер — функция (request) -> {"text": str, "tokens": int}.

    Главное свойство, ради которого shadow вообще существует: что бы ни
    случилось с кандидатом — пустой ответ, мусор, исключение, — пользователь
    получает ровно то, что вернул baseline. Поэтому вызов кандидата обёрнут
    в try/except, и его падение уезжает в candidate_error, а не наружу.

    Исключение baseline НЕ глотается: это настоящая авария продакшена, и
    прятать её за словом «shadow» нельзя.

    Что shadow ловит: всплеск стоимости, взрыв длины ответа, явные отказы.
    Чего не ловит: качество, которое заметит пользователь. Это дымовой
    тест, а не проверка качества.
    """
    user_response = baseline_provider(request)
    record = {
        "baseline_tokens": user_response.get("tokens", 0),
        "candidate_tokens": None,
        "token_delta": None,
        "same_output": None,
        "candidate_error": None,
    }
    try:
        candidate_response = candidate_provider(request)
    except Exception as exc:  # noqa: BLE001 — кандидат не имеет права ронять прод
        record["candidate_error"] = f"{type(exc).__name__}: {exc}"
        return user_response, record
    record["candidate_tokens"] = candidate_response.get("tokens", 0)
    record["token_delta"] = record["candidate_tokens"] - record["baseline_tokens"]
    record["same_output"] = candidate_response.get("text") == user_response.get("text")
    return user_response, record


def gate_breaches(metrics, baseline=None, gates=None):
    """Какие гейты пробиты. Вернуть кортеж имён метрик в порядке GATES.

    gate_breaches({"cost_per_req": 0.030, ...})  ->  ("cost_per_req",)
    gate_breaches(метрики_как_у_базы)            ->  ()

    Пробой — строгое превышение baseline[k] * gates[k]. Ровно на пороге
    гейт НЕ срабатывает: порог «не более чем в 1.2 раза» должен пропускать
    ровно 1.2, иначе шум на последнем знаке начнёт останавливать выкатки.

    Порядок кортежа — порядок GATES, а не порядок словаря метрик: отчёт об
    остановке обязан выглядеть одинаково при одинаковом наборе пробоев.

    Метрика, которой нет в metrics, — KeyError. Гейт, по которому нечего
    измерить, — это не «прошли», это сломанный сбор телеметрии.
    """
    baseline = BASELINE if baseline is None else baseline
    gates = GATES if gates is None else gates
    breached = []
    for name, multiplier in gates.items():
        if name not in metrics:
            raise KeyError(f"no measurement for gate: {name}")
        if metrics[name] > baseline[name] * multiplier:
            breached.append(name)
    return tuple(breached)


def widen_gates_for_noise(gates, nondeterminism):
    """Поднять пороги над шумовым полом LLM-недетерминизма.

    widen_gates_for_noise({"cost_per_req": 1.2}, 0.07)  ->  {"cost_per_req": 1.284}
    widen_gates_for_noise({"cost_per_req": 1.2}, 0.0)   ->  {"cost_per_req": 1.2}

    Новый порог = старый * (1 + nondeterminism).

    Зачем: одинаковые входы дают неодинаковые выходы. Причины из урока —
    неассоциативность float на GPU, разный размер батча, сэмплирование при
    temperature > 0. Разброс доходит до 15% от прогона к прогону. Гейт,
    поставленный ниже этого разброса, будет останавливать здоровые выкатки
    примерно всегда, а команда научится игнорировать остановки — и пропустит
    настоящую.

    nondeterminism < 0 — ValueError: «отрицательный шум» обычно приходит из
    вычитания наблюдений в неправильном порядке, и сузил бы гейты молча.
    """
    if nondeterminism < 0:
        raise ValueError(f"nondeterminism must be >= 0, got {nondeterminism}")
    return {name: value * (1.0 + nondeterminism) for name, value in gates.items()}


def run_canary(measure, stages=None, baseline=None, gates=None):
    """Прогнать прогрессию канарейки, останавливаясь на первом пробое гейта.

    measure — функция (stage_share) -> словарь метрик кандидата на этом шаге.

    Вернуть отчёт:
        {"promoted": bool, "halted_at": доля или None,
         "breaches": кортеж имён, "exposed_share": доля,
         "history": кортеж (доля, кортеж пробоев)}

    run_canary(лямбда_всё_хорошо)      ->  promoted True, exposed_share 1.0
    run_canary(лямбда_дорогой_кандидат)->  promoted False, halted_at 0.01

    exposed_share — доля пользователей, успевшая увидеть кандидата: это
    последний ПРОЙДЕННЫЙ этап, а на остановке — тот этап, на котором
    пробило. Ради этого числа канарейка и существует: плохой релиз задевает
    1% пользователей, а не 100%.

    Гейты проверяются ПОСЛЕ подъёма доли, а не до: измерять надо тот
    трафик, который уже идёт на кандидата.

    Пустой stages — ValueError: прогрессия из нуля шагов молча дала бы
    «promoted», ни на кого не посмотрев.
    """
    stages = STAGES if stages is None else tuple(stages)
    if not stages:
        raise ValueError("stages must not be empty")
    history = []
    for share in stages:
        breaches = gate_breaches(measure(share), baseline, gates)
        history.append((share, breaches))
        if breaches:
            return {
                "promoted": False,
                "halted_at": share,
                "breaches": breaches,
                "exposed_share": share,
                "history": tuple(history),
            }
    return {
        "promoted": True,
        "halted_at": None,
        "breaches": (),
        "exposed_share": stages[-1],
        "history": tuple(history),
    }


def rollback_policy(policy, previous_digest):
    """Откат: доля канарейки в ноль, модель пришпилена к прошлому digest.

    Вернуть НОВЫЙ конфиг политики, не трогая старый.

    rollback_policy({"canary_share": 0.25, "model_digest": "sha256:new"}, "sha256:old")
        ->  {"canary_share": 0.0, "model_digest": "sha256:old",
             "rolled_back": True, "requires_redeploy": False}

    Смысл функции — показать, что откат это операция над КОНФИГОМ, а не
    над артефактом. Флаг долю обнулил, реестр отдал прошлый digest —
    секунды. Если для отката нужен передеплой, откат занимает часы, и его
    начинают откладывать до утра.

    Старый policy остаётся нетронутым: он нужен для post-mortem — с какой
    именно доли и какого digest откатывались.

    previous_digest пустой — ValueError. Откат «в никуда» оставил бы прод
    без пришпиленной модели, и следующий рестарт снова поднял бы кандидата.
    """
    if not previous_digest:
        raise ValueError("previous_digest must be a non-empty pin")
    new_policy = dict(policy)
    new_policy["canary_share"] = 0.0
    new_policy["model_digest"] = previous_digest
    new_policy["rolled_back"] = True
    new_policy["requires_redeploy"] = False
    return new_policy

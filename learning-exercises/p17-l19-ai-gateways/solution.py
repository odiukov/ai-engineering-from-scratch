"""
AI-шлюзы: ключи, лимиты, фолбэк — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

LiteLLM, Portkey, Kong AI Gateway и Bifrost дают всё это конфигом. Здесь мы
собираем шлюз руками, чтобы стало видно, из чего он состоит. Соответствие
настоящему продукту:

    resolve_key        <-  auth + secret reference (ключ провайдера лежит в
                           vault, в конфиге шлюза — только ссылка на него)
    token_bucket       <-  rate limiting в LiteLLM (refill-based)
    sliding_window     <-  rate limiting в Kong (точное окно)
    backoff_delays     <-  configurable backoff у Bifrost
    call_with_fallback <-  fallback chain: 429 у OpenAI -> идём в Anthropic
    handle_request     <-  сам /v1/chat/completions прокси целиком
    latency_budget     <-  overhead шлюза, который добавляется к TTFT
    pick_gateway       <-  решение «какой шлюз брать» из урока

Сети нет: провайдер приходит параметром — функция (model, request) -> ответ
со статусом. Времени тоже нет: все функции получают `now` (секунды, float)
параметром. Ни одного time.time() и ни одного sleep: задержки бэкоффа
считаются числом, а не проживаются. Иначе тест на ретраи шёл бы минуту.
"""

# Тарифы тенантов. burst — вместимость ведра (сколько запросов можно
# выпустить залпом после простоя), refill_per_sec — установившийся темп,
# monthly_tokens — квота на расчётный период.
TIERS = {
    "free":  {"burst": 5,   "refill_per_sec": 1.0,  "monthly_tokens": 100_000},
    "trial": {"burst": 20,  "refill_per_sec": 5.0,  "monthly_tokens": 1_000_000},
    "paid":  {"burst": 100, "refill_per_sec": 50.0, "monthly_tokens": 50_000_000},
}

# Каталог шлюзов из урока (docs/en.md, разделы «Latency budget» и «Numbers
# you should remember»). overhead_ms — середина названного в уроке диапазона:
# LiteLLM 5-15, Portkey 20-40, Kong 3-8, Cloudflare 1-3.
#
# Честно: rps_ceiling в уроке назван только для LiteLLM («breaks down around
# 2000 RPS», «best fit <500 RPS») — берём рабочие 500. Потолки остальных трёх
# в уроке не названы, числа ниже условные и нужны только чтобы правило выбора
# было проверяемым. Не цитируй их как факт.
GATEWAYS = {
    "LiteLLM":    {"overhead_ms": 10, "rps_ceiling":   500, "self_host": True,  "guardrails": False},
    "Portkey":    {"overhead_ms": 30, "rps_ceiling":  5000, "self_host": True,  "guardrails": True},
    "Kong":       {"overhead_ms":  5, "rps_ceiling": 20000, "self_host": True,  "guardrails": False},
    "Cloudflare": {"overhead_ms":  2, "rps_ceiling": 50000, "self_host": False, "guardrails": False},
}

# Статусы, на которых имеет смысл идти к следующему провайдеру. 429 сюда
# входит: у соседнего провайдера свой лимит.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# Префикс ссылки на секрет. В конфиге шлюза лежит "vault:openai_prod",
# а не сам "sk-...".
VAULT_PREFIX = "vault:"


class GatewayError(Exception):
    """Отказ на стороне шлюза: битый ключ, битый конфиг, пустая цепочка.

    Свой класс, а не RuntimeError, — не для красоты. NotImplementedError
    наследуется от RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл
    бы зелёным на пустой заготовке, ничего не проверив.
    """


def resolve_key(api_key, keyring, vault):
    """Проверить ключ клиента и достать секрет провайдера из vault.

    Вернуть {"tenant": ..., "tier": ..., "provider_key": ...}.

    keyring: {ключ клиента: {"tenant", "tier", "secret_ref", "active"}}
    vault:   {имя секрета: значение}

    resolve_key("sk-app-1", keyring, vault)
        ->  {"tenant": "search", "tier": "paid", "provider_key": "sk-openai-real"}
    resolve_key("sk-nope", keyring, vault)  ->  GatewayError

    Четыре причины отказа, и все четыре — GatewayError:
      * ключа нет в keyring;
      * ключ отозван (active is False);
      * тариф не из TIERS — опечатка в конфиге не должна давать безлимит;
      * secret_ref не начинается с VAULT_PREFIX либо в vault такого имени нет.

    Последнее — не придирка. Как только в конфиге шлюза оказывается сырой
    "sk-...", он оказывается в git, в бэкапе и в логе ошибок. Шлюз обязан
    уметь только ССЫЛАТЬСЯ на секрет.
    """
    record = keyring.get(api_key)
    if record is None:
        raise GatewayError(f"unknown api key: {api_key}")
    if not record.get("active", True):
        raise GatewayError(f"revoked api key: {api_key}")
    tier = record.get("tier")
    if tier not in TIERS:
        raise GatewayError(f"unknown tier: {tier}")
    ref = record.get("secret_ref", "")
    if not ref.startswith(VAULT_PREFIX):
        # именно так утекают ключи: кто-то один раз вписал секрет в конфиг
        raise GatewayError(f"secret must be a vault reference, got: {ref!r}")
    name = ref[len(VAULT_PREFIX):]
    if name not in vault:
        raise GatewayError(f"vault has no secret named: {name}")
    return {"tenant": record["tenant"], "tier": tier, "provider_key": vault[name]}


def token_bucket(state, now, cost, capacity, refill_per_sec):
    """Token bucket: пропустить запрос, если в ведре хватает токенов.

    Вернуть (allowed, new_state). state — {"tokens": float, "at": float}
    либо None для «ведро полное, отсчёт с now».

    token_bucket(None, 0.0, 1, 5, 1.0)          ->  (True, {"tokens": 4.0, "at": 0.0})
    token_bucket({"tokens": 0.0, "at": 0.0}, 2.0, 1, 5, 1.0)
                                                ->  (True, {"tokens": 1.0, "at": 2.0})
    token_bucket({"tokens": 0.0, "at": 0.0}, 0.5, 1, 5, 1.0)
                                                ->  (False, {"tokens": 0.5, "at": 0.5})

    Порядок обязателен: сначала долить за прошедшее время, потом проверить.
    Наоборот — и запрос после долгой паузы отвергается на полном ведре.

    Отказ НЕ списывает токены, но время всё равно двигает: иначе клиент,
    который долбится в закрытую дверь, отодвигал бы себе же момент долива.

    Входной state не меняется — возвращается новый словарь. Мутация чужого
    состояния из функции проверки лимита ломает любой откат.

    now назад во времени — GatewayError. На практике это рассинхрон часов
    между узлами шлюза, и молча долить «отрицательное время» хуже, чем упасть.
    """
    if state is None:
        tokens, at = float(capacity), float(now)
    else:
        tokens, at = float(state["tokens"]), float(state["at"])
    if now < at:
        raise GatewayError(f"clock went backwards: now={now} < state.at={at}")
    tokens = min(float(capacity), tokens + (now - at) * refill_per_sec)
    # допуск 1e-9: tokens складывается из долей, точное >= на float ловится
    # не всегда, и клиент терял бы один запрос из тысячи без причины
    if tokens + 1e-9 >= cost:
        return True, {"tokens": tokens - cost, "at": float(now)}
    return False, {"tokens": tokens, "at": float(now)}


def sliding_window(events, now, window_s, limit):
    """Sliding window: не больше limit запросов за последние window_s секунд.

    events — метки времени прошлых ПРОПУЩЕННЫХ запросов.
    Вернуть (allowed, kept) — kept уже без протухших и с новым событием,
    если запрос пропущен.

    sliding_window((), 0.0, 60.0, 2)              ->  (True, (0.0,))
    sliding_window((0.0, 1.0), 2.0, 60.0, 2)      ->  (False, (0.0, 1.0))
    sliding_window((0.0, 1.0), 60.5, 60.0, 2)     ->  (True, (1.0, 60.5))

    Окно полуоткрытое: событие ровно на границе now - window_s уже вышло.

    Разница с token_bucket видна не на средней скорости, а на всплеске.
    Ведро копит право на залп во время простоя: после минуты тишины оно
    пропустит capacity запросов подряд. Окно не копит ничего — оно смотрит
    только назад. Отсюда и позиционирование из урока: LiteLLM возит
    token-bucket, Kong — sliding-window, «better fairness».
    """
    kept = tuple(t for t in events if t > now - window_s)
    if len(kept) >= limit:
        return False, kept
    return True, kept + (float(now),)


def backoff_delays(attempts, base_ms=100.0, factor=2.0, cap_ms=8000.0, rng=None):
    """Задержки перед повторами: экспонента с потолком, при желании с джиттером.

    Вернуть кортеж длиной attempts. Ничего не спит — только считает.

    backoff_delays(4)                 ->  (100.0, 200.0, 400.0, 800.0)
    backoff_delays(5, cap_ms=300.0)   ->  (100.0, 200.0, 300.0, 300.0, 300.0)
    backoff_delays(0)                 ->  ()

    rng — объект с методом uniform (обычно random.Random(seed)). Если он
    передан, каждая задержка становится rng.uniform(0, exp_delay) — это
    «full jitter». Без него все клиенты, которых уронил один 503, вернутся
    ровно в одну и ту же миллисекунду и уронят провайдера второй раз.

    Глобальный random внутри запрещён: два прогона теста обязаны совпасть.

    attempts < 0 — GatewayError. Отрицательное число повторов ничего не
    значит и обычно приходит из арифметики вида `limit - len(chain)`.
    """
    if attempts < 0:
        raise GatewayError(f"attempts must be >= 0, got {attempts}")
    delays = []
    # экспонента накручивается итеративно, а не как base * factor**i: на паре
    # тысяч попыток factor**i переполняет float и падает с OverflowError, хотя
    # ответ давно упёрся в потолок
    exp = float(base_ms)
    for _ in range(attempts):
        # потолок ставим ДО джиттера: иначе редкий большой uniform пробил бы cap
        capped = min(cap_ms, exp)
        delays.append(rng.uniform(0.0, capped) if rng is not None else capped)
        exp = min(cap_ms, exp * factor)
    return tuple(delays)


def call_with_fallback(chain, request, provider, retries_per_provider=0, rng=None):
    """Пройти цепочку провайдеров, пока кто-нибудь не ответит 200.

    provider — функция (model, request) -> {"status": int, "usage": {...}}.

    Вернуть запись с постоянным набором ключей:
        model, status, attempts, retries, waited_ms, usage, error

    attempts — кортеж пар (model, status) в порядке обращения.

    call_with_fallback(("gpt-4o", "claude"), req, провайдер_с_429_на_gpt4o)
        ->  model "claude", attempts (("gpt-4o", 429), ("claude", 200))

    Три правила, каждое из которых кому-то стоило денег:

      * статус не из RETRY_STATUSES обрывает цепочку немедленно. Кривой
        запрос будет кривым и у следующего провайдера — фолбэк лишь умножит
        счёт на длину цепочки;
      * повторы к ОДНОМУ провайдеру идут до перехода к следующему, и их
        суммарная задержка копится в waited_ms через backoff_delays;
      * когда цепочка кончилась, model остаётся None, а error заполняется.
        Исключение здесь не бросается: вызов не «сломался», он «не удался»,
        и вызывающий обязан увидеть attempts, чтобы понять, кого винить.

    Пустая цепочка — GatewayError: это ошибка конфига, а не аварии.
    """
    if not chain:
        raise GatewayError("provider chain is empty")
    record = {
        "model": None,
        "status": None,
        "attempts": (),
        "retries": 0,
        "waited_ms": 0.0,
        "usage": None,
        "error": None,
    }
    attempts = []
    for model in chain:
        # retries_per_provider повторов сверх первой попытки
        delays = backoff_delays(retries_per_provider, rng=rng)
        for try_index in range(retries_per_provider + 1):
            if try_index > 0:
                record["retries"] += 1
                record["waited_ms"] += delays[try_index - 1]
            response = provider(model, request)
            status = response.get("status", 200)
            attempts.append((model, status))
            record["status"] = status
            if status == 200:
                record["model"] = model
                record["usage"] = response.get("usage", {})
                record["attempts"] = tuple(attempts)
                return record
            if status not in RETRY_STATUSES:
                record["error"] = f"{model}: non-retryable status {status}"
                record["attempts"] = tuple(attempts)
                return record
    record["attempts"] = tuple(attempts)
    record["error"] = "all providers failed"
    return record


def handle_request(api_key, request, now, state, provider, chain, rng=None):
    """Один запрос через шлюз целиком: auth, лимит, квота, фолбэк, учёт.

    request — {"estimated_tokens": int, ...}.
    state   — изменяемое состояние шлюза:
        {"keyring": {...}, "vault": {...}, "buckets": {tenant: ...},
         "quota_used": {tenant: int}}

    Вернуть запись: tenant, tier, status, model, attempts, tokens, error.

    handle_request("sk-app-1", req, 0.0, state, провайдер, ("gpt-4o",))
        ->  status 200, model "gpt-4o"
    handle_request("sk-nope", req, 0.0, state, провайдер, ("gpt-4o",))
        ->  status 401, model None

    Шлюз не бросает наружу свои внутренние исключения: GatewayError от
    resolve_key превращается в status 401 с текстом в error. Наружу торчит
    HTTP-семантика, внутри — типизированный отказ.

    Порядок проверок принципиален и стоит денег: auth -> rate limit ->
    квота -> провайдер. Проверить квоту после вызова провайдера — значит
    заплатить за запрос, который сам же и отвергнешь.

    Отвергнутый запрос не оставляет следов в quota_used: превышение лимита
    не должно съедать месячную квоту. А вот ведро он трогает — token_bucket
    двигает время даже на отказе.

    status: 200 успех, 401 ключ, 429 лимит или квота, иначе — статус
    последнего провайдера.
    """
    record = {
        "tenant": None,
        "tier": None,
        "status": None,
        "model": None,
        "attempts": (),
        "tokens": 0,
        "error": None,
    }
    try:
        identity = resolve_key(api_key, state["keyring"], state["vault"])
    except GatewayError as exc:
        record["status"] = 401
        record["error"] = str(exc)
        return record

    tenant, tier = identity["tenant"], identity["tier"]
    record["tenant"], record["tier"] = tenant, tier
    limits = TIERS[tier]

    allowed, bucket = token_bucket(
        state["buckets"].get(tenant), now, 1, limits["burst"], limits["refill_per_sec"]
    )
    state["buckets"][tenant] = bucket
    if not allowed:
        record["status"] = 429
        record["error"] = "rate limit exceeded"
        return record

    wanted = request.get("estimated_tokens", 0)
    used = state["quota_used"].get(tenant, 0)
    if used + wanted > limits["monthly_tokens"]:
        record["status"] = 429
        record["error"] = "monthly token quota exhausted"
        return record

    call = call_with_fallback(chain, request, provider, retries_per_provider=0, rng=rng)
    record["status"] = call["status"]
    record["model"] = call["model"]
    record["attempts"] = call["attempts"]
    record["error"] = call["error"]
    if call["model"] is not None:
        usage = call["usage"] or {}
        spent = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        record["tokens"] = spent
        state["quota_used"][tenant] = used + spent
    return record


def latency_budget(gateway, baseline_ttft_ms, sla_p99_ms):
    """Влезает ли шлюз в бюджет TTFT: свой overhead плюс время модели.

    Вернуть {"gateway", "overhead_ms", "total_ms", "headroom_ms", "fits"}.

    latency_budget("Kong", 300, 400)
        ->  total 305, headroom 95, fits True
    latency_budget("Portkey", 300, 320)
        ->  total 330, headroom -10, fits False

    Overhead шлюза складывается с TTFT напрямую — это не параллельная
    работа, а последовательная: пока шлюз не отдал байты провайдеру,
    провайдер не начал считать. Отсюда вывод урока: для P99 TTFT < 100 мс
    остаются Kong и Cloudflare, для < 500 мс подходит любой.

    Неизвестное имя шлюза — GatewayError, а не «ну возьмём ноль»: нулевой
    overhead у несуществующего шлюза даст отчёт, который всегда влезает.
    """
    spec = GATEWAYS.get(gateway)
    if spec is None:
        raise GatewayError(f"unknown gateway: {gateway}")
    total = baseline_ttft_ms + spec["overhead_ms"]
    return {
        "gateway": gateway,
        "overhead_ms": spec["overhead_ms"],
        "total_ms": total,
        "headroom_ms": sla_p99_ms - total,
        "fits": total <= sla_p99_ms,
    }


def pick_gateway(rps, baseline_ttft_ms, sla_p99_ms, self_host=False, guardrails=False):
    """Выбрать шлюз под нагрузку, SLA и требования комплаенса. None — если нет.

    pick_gateway(100, 300, 500)                      ->  "Cloudflare"
    pick_gateway(100, 300, 500, self_host=True)      ->  "Kong"
    pick_gateway(100, 300, 500, self_host=True, guardrails=True)  ->  "Portkey"
    pick_gateway(100_000, 300, 500)                  ->  None

    Порядок отсева принципиален: сначала жёсткие требования (потолок RPS,
    self-host, guardrails, бюджет латентности через latency_budget), и
    только потом среди выживших берём наименьший overhead. Наоборот
    получится «взяли самый быстрый и понадеялись, что данные можно вывозить
    из страны» — ровно та ошибка, из-за которой у healthcare-клиента
    оказывается managed-шлюз.

    При равном overhead выигрывает меньшее имя: ответ обязан не зависеть от
    порядка перебора словаря.
    """
    best = None
    for name, spec in GATEWAYS.items():
        if spec["rps_ceiling"] < rps:
            continue
        if self_host and not spec["self_host"]:
            continue
        if guardrails and not spec["guardrails"]:
            continue
        if not latency_budget(name, baseline_ttft_ms, sla_p99_ms)["fits"]:
            continue
        key = (spec["overhead_ms"], name)
        if best is None or key < best:
            best = key
    return None if best is None else best[1]

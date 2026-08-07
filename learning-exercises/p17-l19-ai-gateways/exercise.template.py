"""
AI-шлюзы: ключи, лимиты, фолбэк

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l19-ai-gateways
Разбор:  /check-code p17-l19-ai-gateways
"""

TIERS = {
    "free":  {"burst": 5,   "refill_per_sec": 1.0,  "monthly_tokens": 100_000},
    "trial": {"burst": 20,  "refill_per_sec": 5.0,  "monthly_tokens": 1_000_000},
    "paid":  {"burst": 100, "refill_per_sec": 50.0, "monthly_tokens": 50_000_000},
}
GATEWAYS = {
    "LiteLLM":    {"overhead_ms": 10, "rps_ceiling":   500, "self_host": True,  "guardrails": False},
    "Portkey":    {"overhead_ms": 30, "rps_ceiling":  5000, "self_host": True,  "guardrails": True},
    "Kong":       {"overhead_ms":  5, "rps_ceiling": 20000, "self_host": True,  "guardrails": False},
    "Cloudflare": {"overhead_ms":  2, "rps_ceiling": 50000, "self_host": False, "guardrails": False},
}
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
VAULT_PREFIX = "vault:"


class GatewayError(Exception):
    """Отказ на стороне шлюза: битый ключ, битый конфиг, пустая цепочка.

    Свой класс, а не RuntimeError, — не для красоты. NotImplementedError
    наследуется от RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл
    бы зелёным на пустой заготовке, ничего не проверив.
    """
    pass


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError

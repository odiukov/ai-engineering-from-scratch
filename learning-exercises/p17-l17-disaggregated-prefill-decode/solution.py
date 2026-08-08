"""
Разделение prefill и decode: два пула и цена передачи KV — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Чему это соответствует в настоящих системах:

    kv_bytes                 <-  размер KV-кэша, который надо перевезти:
                                 4К токенов на 70B FP8 = 655.36 МБ
    transfer_ms              <-  NIXL в NVIDIA Dynamo: RDMA/InfiniBand, если
                                 фабрика есть, TCP если нет
    phase_ms                 <-  две фазы с разными узкими местами: prefill
                                 упирается в вычисления, decode в память
    colocated_ms             <-  обычная схема: обе фазы на одной GPU
    disaggregated_ms         <-  prefill-пул -> KV -> decode-пул
    disagg_gain_ms           <-  ради чего всё: выигрыш минус налог на передачу
    crossover_prompt_tokens  <-  «Prompts < 512 tokens: transfer tax dominates
                                 gain» — тот самый порог, посчитанный, а не
                                 процитированный
    fleet_report             <-  ответ на вопрос «а на нашей смеси трафика?»

Ни GPU, ни Dynamo, ни llm-d: разделение — это арифметика над временем фаз и
временем передачи, и она моделируется честно. Никаких sleep, всё время
приходит и уходит числами.

Модель штрафа колокации. На общей GPU decode-шаги чужих запросов идут
непрерывно и занимают устройство, поэтому длинный forward по промпту ждёт
своей очереди и растягивается. Decode от разделения не выигрывает почти
ничего: он упирается в полосу памяти, а её отдельный пул не добавляет. Из-за
этого в разности времён decode сокращается целиком, и порог разделения
оказывается порогом по ДЛИНЕ ПРОМПТА, а не по длине ответа.

Числа — снимок H100-класса на 2026, они дрейфуют.
"""

# 80 слоёв * 2 (K/V) * 8 KV-голов * 128 * 1 байт FP8.
KV_BYTES_PER_TOKEN = 163_840

# Рукопожатие NIXL плюс хоп роутера. Платится один раз за передачу,
# независимо от размера KV, — и именно оно делает разделение невыгодным
# на коротких промптах.
TRANSFER_SETUP_MS = 20.0

# Токенов в секунду на выделенной GPU.
PREFILL_TPS = 8_000.0    # один forward по всему промпту, упор в вычисления
DECODE_TPS = 40.0        # по токену за шаг на один запрос, упор в память

# Доля времени prefill, которую отъедает соседний decode на общей GPU.
COLOCATION_PENALTY = 0.30

# Скорость линка между пулами, ГБ/с.
LINK_RDMA_GBPS = 100.0
LINK_TCP_GBPS = 10.0


class DisaggError(Exception):
    """Пулы или линк спрошены о невозможном: отрицательные токены, мёртвый линк.

    Свой класс, а не ValueError и тем более не RuntimeError: заготовка бросает
    NotImplementedError, который наследуется от RuntimeError, и тест на
    родительский класс прошёл бы зелёным по пустому файлу.
    """


def kv_bytes(prompt_tokens, bytes_per_token=KV_BYTES_PER_TOKEN):
    """Сколько байт KV-кэша накопил prefill на промпте такой длины.

    kv_bytes(4000)  ->  655_360_000    (80 * 2 * 8 * 128 * 4000)
    kv_bytes(0)     ->  0

    Это то, что придётся физически перевезти из prefill-пула в decode-пул.
    Линейно по длине промпта: на каждый токен свой K и V в каждом слое.

    DisaggError на отрицательную длину и на неположительный bytes_per_token.
    """
    if prompt_tokens < 0:
        raise DisaggError(f"prompt_tokens must be non-negative, got {prompt_tokens}")
    if bytes_per_token <= 0:
        raise DisaggError(f"bytes_per_token must be positive, got {bytes_per_token}")
    return prompt_tokens * bytes_per_token


def transfer_ms(prompt_tokens, link_gbps, bytes_per_token=KV_BYTES_PER_TOKEN):
    """Время передачи KV между пулами: рукопожатие плюс байты по линку.

    transfer_ms(4000, LINK_RDMA_GBPS)  ->  26.5536 (20 + 6.5536 передачи)
    transfer_ms(4000, LINK_TCP_GBPS)   ->  85.536  (20 + 65.536 передачи)
    transfer_ms(0, LINK_RDMA_GBPS)     ->  20.0   (везти нечего, а хоп есть)

    Разбор: 655.36 МБ по 100 ГБ/с = 6.5536 мс, по 10 ГБ/с = 65.536 мс.

    Ловушка: слагаемое TRANSFER_SETUP_MS постоянное. Именно оно, а не байты,
    решает судьбу коротких промптов: на 100 токенах везти почти нечего, а
    рукопожатие всё равно стоит целых 20 мс.

    DisaggError на неположительную скорость линка.
    """
    if link_gbps <= 0:
        raise DisaggError(f"link_gbps must be positive, got {link_gbps}")
    seconds = kv_bytes(prompt_tokens, bytes_per_token) / (link_gbps * 1e9)
    return TRANSFER_SETUP_MS + seconds * 1000.0


def phase_ms(prompt_tokens, output_tokens,
             prefill_tps=PREFILL_TPS, decode_tps=DECODE_TPS):
    """Время двух фаз на ВЫДЕЛЕННОЙ GPU, без чужой нагрузки рядом.

    Возвращает dict: prefill_ms, decode_ms.

    phase_ms(4000, 300)  ->  {'prefill_ms': 500.0, 'decode_ms': 7500.0}
    phase_ms(400, 30)    ->  {'prefill_ms': 50.0,  'decode_ms': 750.0}

    Обрати внимание на пропорцию: decode на порядок дольше prefill, потому что
    prefill проглатывает весь промпт одним forward, а decode делает по шагу на
    токен. Отсюда и разные узкие места — вычисления против полосы памяти.

    DisaggError на отрицательные токены и неположительные скорости.
    """
    if prompt_tokens < 0 or output_tokens < 0:
        raise DisaggError(f"negative token count: {prompt_tokens}, {output_tokens}")
    if prefill_tps <= 0 or decode_tps <= 0:
        raise DisaggError(f"rates must be positive: {prefill_tps}, {decode_tps}")
    return {
        "prefill_ms": prompt_tokens / prefill_tps * 1000.0,
        "decode_ms": output_tokens / decode_tps * 1000.0,
    }


def colocated_ms(prompt_tokens, output_tokens, penalty=COLOCATION_PENALTY,
                 prefill_tps=PREFILL_TPS, decode_tps=DECODE_TPS):
    """Время запроса на общей GPU: prefill растянут штрафом, decode как есть.

    colocated_ms(4000, 300)  ->  примерно 8214.29
    colocated_ms(4000, 300, penalty=0.0)  ->  8000.0  (штрафа нет)

    Разбор: 500 мс prefill / (1 - 0.30) = 714.29, плюс 7500 мс decode.

    Ловушка: штраф — это ДЕЛЕНИЕ на (1 - penalty), а не умножение на
    (1 + penalty). Penalty — доля отобранного времени устройства, а не надбавка
    сверху. На 0.30 разница между 1.4286 и 1.30 — это 10% времени prefill.

    Decode штрафом не облагается: он и так упирается в полосу памяти, ему
    соседний prefill почти не мешает. Из-за этого в disagg_gain_ms decode
    сократится целиком.

    DisaggError на penalty вне [0, 1).
    """
    if not 0.0 <= penalty < 1.0:
        raise DisaggError(f"penalty must be in [0, 1), got {penalty}")
    phases = phase_ms(prompt_tokens, output_tokens, prefill_tps, decode_tps)
    return phases["prefill_ms"] / (1.0 - penalty) + phases["decode_ms"]


def disaggregated_ms(prompt_tokens, output_tokens, link_gbps,
                     prefill_tps=PREFILL_TPS, decode_tps=DECODE_TPS,
                     bytes_per_token=KV_BYTES_PER_TOKEN):
    """Время запроса на двух пулах: чистый prefill, передача KV, чистый decode.

    disaggregated_ms(4000, 300, LINK_TCP_GBPS)   ->  8085.536
    disaggregated_ms(4000, 300, LINK_RDMA_GBPS)  ->  8026.5536

    Разбор первого: 500 prefill + 85.536 передача + 7500 decode.

    Штрафа колокации здесь нет — в этом весь смысл разделения. Зато появился
    налог на передачу, которого в колокации нет вообще: там KV уже лежит в
    той самой HBM, где будет считаться decode.
    """
    phases = phase_ms(prompt_tokens, output_tokens, prefill_tps, decode_tps)
    return (phases["prefill_ms"]
            + transfer_ms(prompt_tokens, link_gbps, bytes_per_token)
            + phases["decode_ms"])


def disagg_gain_ms(prompt_tokens, output_tokens, link_gbps,
                   penalty=COLOCATION_PENALTY, prefill_tps=PREFILL_TPS,
                   decode_tps=DECODE_TPS, bytes_per_token=KV_BYTES_PER_TOKEN):
    """Выигрыш разделения в миллисекундах. Отрицательный — разделять не надо.

    disagg_gain_ms(4000, 300, LINK_TCP_GBPS)  ->  примерно 128.75
    disagg_gain_ms(200, 300, LINK_TCP_GBPS)   ->  примерно -12.56
    disagg_gain_ms(4000, 999, LINK_TCP_GBPS)  ->  ровно столько же, сколько
                                                  с 300 — decode сократился

    Разбор второго: на 200 токенах штраф колокации отбирает всего 8.57 мс, а
    рукопожатие и передача стоят 23.28. Это и есть «Prompts < 512 tokens:
    transfer tax dominates gain» из урока.

    Свойство, которое стоит понять раньше формулы: длина ОТВЕТА в выигрыш не
    входит. Decode считается одинаково в обеих схемах и в разности исчезает.
    Поэтому решение «разделять или нет» — это решение про длину ПРОМПТА.
    """
    colocated = colocated_ms(prompt_tokens, output_tokens, penalty,
                             prefill_tps, decode_tps)
    split = disaggregated_ms(prompt_tokens, output_tokens, link_gbps,
                             prefill_tps, decode_tps, bytes_per_token)
    return colocated - split


def crossover_prompt_tokens(link_gbps, penalty=COLOCATION_PENALTY,
                            prefill_tps=PREFILL_TPS,
                            bytes_per_token=KV_BYTES_PER_TOKEN,
                            max_tokens=1 << 20):
    """Самый короткий промпт, на котором разделение уже выигрывает. Или None.

    crossover_prompt_tokens(LINK_TCP_GBPS)   ->  538
    crossover_prompt_tokens(LINK_RDMA_GBPS)  ->  386
    crossover_prompt_tokens(2.0)             ->  None

    Урок называет порог около 512 токенов — вот он, посчитанный: 538 на TCP.
    На RDMA порог ниже: линк быстрее, налог на байт меньше, окупается раньше.

    None означает, что порога нет вообще. Выигрыш растёт с длиной промпта со
    скоростью (penalty/(1-penalty))/prefill_tps на токен, а налог — со
    скоростью bytes_per_token/link. Если линк медленнее примерно 3.06 ГБ/с,
    вторая скорость больше первой, и удлинение промпта делает только хуже: чем
    длиннее промпт, тем больше проигрыш. При этой геометрии граница — около
    3.06 ГБ/с. Это и есть строчка урока «No RDMA
    fabric: TCP transfer tax is heavier» в предельном виде.

    Ловушка: не ищи порог линейным перебором до миллиона — выигрыш монотонен
    по длине промпта, годится двоичный поиск.

    DisaggError на max_tokens < 1.
    """
    if max_tokens < 1:
        raise DisaggError(f"max_tokens must be at least 1, got {max_tokens}")

    def gain(n):
        # длина ответа в выигрыш не входит, так что берём ноль
        return disagg_gain_ms(n, 0, link_gbps, penalty, prefill_tps,
                              DECODE_TPS, bytes_per_token)

    if gain(max_tokens) <= 0:
        return None
    lo, hi = 1, max_tokens          # gain(hi) > 0, ищем наименьший такой
    if gain(lo) > 0:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if gain(mid) > 0:
            hi = mid
        else:
            lo = mid
    return hi


def fleet_report(requests, link_gbps, penalty=COLOCATION_PENALTY,
                 prefill_tps=PREFILL_TPS, decode_tps=DECODE_TPS,
                 bytes_per_token=KV_BYTES_PER_TOKEN):
    """Прогнать смесь трафика через обе схемы и сравнить.

    requests — список пар (prompt_tokens, output_tokens).

    Возвращает dict:
      requests            — сколько запросов,
      colocated_ms        — суммарное время на общих GPU,
      disaggregated_ms    — суммарное время на двух пулах,
      gain_ms             — разность, может быть отрицательной,
      gain_pct            — та же разность в процентах от колокации,
      helped, hurt        — сколько запросов выиграло и сколько проиграло.

    fleet_report([], LINK_TCP_GBPS)["gain_pct"]  ->  0.0

    Ради чего функция: решение принимается не по одному запросу, а по смеси.
    Флот из коротких промптов от разделения проигрывает, флот из длинных
    выигрывает, а реальный трафик — смесь, и знак итога зависит от неё. Именно
    поэтому в уроке RAG с префиксами на 8К называется лучшим кандидатом, а чат
    с короткими репликами — худшим.
    """
    colocated = split = 0.0
    helped = hurt = 0
    for prompt_tokens, output_tokens in requests:
        colocated += colocated_ms(prompt_tokens, output_tokens, penalty,
                                  prefill_tps, decode_tps)
        split += disaggregated_ms(prompt_tokens, output_tokens, link_gbps,
                                  prefill_tps, decode_tps, bytes_per_token)
        gain = disagg_gain_ms(prompt_tokens, output_tokens, link_gbps, penalty,
                              prefill_tps, decode_tps, bytes_per_token)
        if gain > 0:
            helped += 1
        elif gain < 0:
            hurt += 1
    return {
        "requests": len(requests),
        "colocated_ms": colocated,
        "disaggregated_ms": split,
        "gain_ms": colocated - split,
        "gain_pct": (colocated - split) / colocated * 100 if colocated else 0.0,
        "helped": helped,
        "hurt": hurt,
    }

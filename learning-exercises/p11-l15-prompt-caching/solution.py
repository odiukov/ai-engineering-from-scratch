"""
Кэширование промптов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

У провайдера это одна строчка `cache_control: {"type": "ephemeral"}`, а под
ней — префиксный кэш KV и биллинг с двумя ставками. Здесь мы собираем этот
механизм руками, чтобы стало видно, почему он промахивается.

Соответствие настоящему API:
    cache_friendly_layout  <-  порядок блоков system / tools / few-shot / user
    breakpoint_len         <-  позиция маркера cache_control
    split_tokens           <-  поля cache_creation_input_tokens и
                               cache_read_input_tokens из usage
    request_cost           <-  ставки 1.25x на запись и 0.1x на чтение
    break_even_reads       <-  «кэшируй то, что переиспользуешь 3 раза»

Токен здесь — просто элемент списка. Настоящий BPE не нужен: кэш сравнивает
токены, а не смысл, и на списке строк это видно даже нагляднее.
"""

import math

# Порог Anthropic: блок короче 1024 токенов (2048 у Haiku) молча НЕ кэшируется.
# Не ошибка, не предупреждение — просто ноль попаданий и недоумение в логах.
MIN_CACHEABLE_TOKENS = 1024

# Множители к базовой цене входного токена у Anthropic.
WRITE_MULTIPLIER = 1.25  # запись в кэш дороже обычного токена
READ_MULTIPLIER = 0.10  # чтение из кэша в десять раз дешевле


def common_prefix_len(a, b):
    """Сколько токенов совпадает с самого начала обеих последовательностей.

    common_prefix_len(["a", "b", "c"], ["a", "b", "d"])  ->  2
    common_prefix_len(["a"], ["b", "a"])                 ->  0
    common_prefix_len([], ["a"])                         ->  0

    Именно префикс, а не пересечение. Кэш ничего не знает про «эти два
    промпта на 90% одинаковые»: он сравнивает токены слева направо и
    останавливается на первом расхождении.
    """
    n = 0
    # zip сам обрывается на конце короткой последовательности,
    # проверять длины отдельно не нужно; strict=False говорит об этом вслух
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def cache_friendly_layout(sections):
    """Переставить секции промпта так, чтобы стабильное было сверху.

    sections — список (имя, список токенов, стабильна ли).

    cache_friendly_layout([("user", ["q"], False),
                           ("system", ["s1", "s2"], True),
                           ("tools", ["t"], True)])
        ->  ([("system", ["s1", "s2"], True),
              ("tools", ["t"], True),
              ("user", ["q"], False)],
             ["s1", "s2", "t"])

    Возвращает пару (переставленные секции, токены кэшируемого префикса).

    Относительный порядок внутри каждой группы сохраняется: если системный
    промпт шёл до few-shot примеров, он и останется до них.

    Ловушка урока: одна динамическая строка сверху («Current time: ...»)
    сдвигает точку расхождения в начало промпта, и кэш промахивается на
    КАЖДОМ запросе. Правило простое — стабильное вверх, изменчивое вниз.
    """
    # sorted устойчива, поэтому внутри групп порядок остаётся исходным.
    # not stable -> False (0) сортируется раньше True (1), значит стабильные вперёд
    ordered = sorted(sections, key=lambda s: not s[2])
    prefix = []
    for _, tokens, stable in ordered:
        if not stable:
            break
        prefix.extend(tokens)
    return ordered, prefix


def cache_lookup(cache, prompt, min_cacheable=MIN_CACHEABLE_TOKENS):
    """Длина самой длинной записи кэша, которая целиком лежит в начале prompt.

    Ноль означает промах.

    cache_lookup([["a", "b"]], ["a", "b", "c"], min_cacheable=2)  ->  2
    cache_lookup([["a", "b"]], ["a", "x", "c"], min_cacheable=2)  ->  0
    cache_lookup([["a", "b"]], ["a", "b", "c"], min_cacheable=8)  ->  0

    cache — список ранее сохранённых префиксов (каждый список токенов).

    Частичного попадания не бывает: запись либо целиком совпала с началом
    запроса, либо не дала ничего. Совпадения на 90% провайдер не считает
    попаданием ни на один токен.

    Записи короче min_cacheable не хранятся вовсе — это тот самый порог в
    1024 токена, из-за которого маленькие блоки «не кэшируются» без единого
    сообщения об ошибке.
    """
    best = 0
    for entry in cache:
        if len(entry) < min_cacheable or len(entry) > len(prompt):
            continue
        if common_prefix_len(entry, prompt) == len(entry):
            best = max(best, len(entry))
    return best


def split_tokens(prompt, cache, breakpoint_len, min_cacheable=MIN_CACHEABLE_TOKENS):
    """Разложить токены запроса на три корзины биллинга.

    Возвращает {"read": n, "write": n, "fresh": n}, где
      read   — прочитано из кэша по ставке 0.1x,
      write  — записано в кэш по ставке 1.25x,
      fresh  — обычные входные токены по ставке 1x (всё после точки останова).

    split_tokens(list("abcdefghij"), [], 6, min_cacheable=4)
        ->  {"read": 0, "write": 6, "fresh": 4}
    split_tokens(list("abcdefghij"), [list("abcdef")], 6, min_cacheable=4)
        ->  {"read": 6, "write": 0, "fresh": 4}

    breakpoint_len — сколько токенов с начала помечено как кэшируемые, то
    есть позиция маркера cache_control. Больше длины промпта — ValueError.

    Ловушка: если помеченный блок короче min_cacheable, кэш не работает
    ВООБЩЕ, и все токены становятся fresh. Ни записи, ни чтения, ни ошибки.
    """
    if breakpoint_len > len(prompt):
        raise ValueError(
            f"точка останова {breakpoint_len} за пределами промпта в {len(prompt)} токенов"
        )
    if breakpoint_len < min_cacheable:
        return {"read": 0, "write": 0, "fresh": len(prompt)}
    # min на всякий случай: запись длиннее точки останова прочитать целиком нельзя
    read = min(cache_lookup(cache, prompt, min_cacheable), breakpoint_len)
    return {"read": read, "write": breakpoint_len - read, "fresh": len(prompt) - breakpoint_len}


def request_cost(split, input_price, write_mult=WRITE_MULTIPLIER, read_mult=READ_MULTIPLIER):
    """Цена входных токенов одного запроса в долларах.

    input_price — цена за 1M обычных входных токенов.

    request_cost({"read": 0, "write": 0, "fresh": 1000}, 3.0)   ->  0.003
    request_cost({"read": 0, "write": 1000, "fresh": 0}, 3.0)   ->  0.00375
    request_cost({"read": 1000, "write": 0, "fresh": 0}, 3.0)   ->  0.0003

    Три ставки в одной формуле. Обрати внимание на средний пример: запись
    дороже обычного токена. Кэш, в который только пишут и ни разу не читают,
    не экономит, а тратит на четверть больше.
    """
    billable = split["fresh"] + split["write"] * write_mult + split["read"] * read_mult
    return round(billable / 1_000_000 * input_price, 12)


def simulate_session(
    prompts,
    breakpoint_len,
    input_price,
    min_cacheable=MIN_CACHEABLE_TOKENS,
    write_mult=WRITE_MULTIPLIER,
    read_mult=READ_MULTIPLIER,
):
    """Прогнать последовательность запросов через кэш и посчитать счёт.

    Возвращает словарь: total_cost_usd, no_cache_cost_usd, reads, writes,
    saving_pct.

    Кэш живёт внутри вызова: первый запрос платит за запись, последующие с
    тем же префиксом читают.

    Для трёх одинаковых промптов по 10 токенов с точкой останова на 6:
    simulate_session([p, p, p], 6, 3.0, min_cacheable=4)
        ->  {"reads": 2, "writes": 1, "saving_pct": 31.0, ...}

    Это ровно тот отчёт, который в бою собирают из полей usage. Если
    reads держится в нуле — ключ кэша дрейфует, ищи динамику над точкой
    останова.
    """
    cache = []
    total = 0.0
    baseline_tokens = 0
    reads = 0
    writes = 0
    for prompt in prompts:
        split = split_tokens(prompt, cache, breakpoint_len, min_cacheable)
        total += request_cost(split, input_price, write_mult, read_mult)
        baseline_tokens += len(prompt)
        if split["read"]:
            reads += 1
        if split["write"]:
            writes += 1
            # копия: prompt принадлежит вызывающему коду, срез отдаём в кэш своим
            cache.append(list(prompt[:breakpoint_len]))
    no_cache = baseline_tokens / 1_000_000 * input_price
    return {
        "total_cost_usd": round(total, 12),
        "no_cache_cost_usd": round(no_cache, 12),
        "reads": reads,
        "writes": writes,
        "saving_pct": round((1 - total / no_cache) * 100, 4) if no_cache else 0.0,
    }


def break_even_reads(write_mult=WRITE_MULTIPLIER, read_mult=READ_MULTIPLIER):
    """Сколько ЧТЕНИЙ должно последовать за одной записью, чтобы кэш окупился.

    break_even_reads()             ->  1    (Anthropic 5 минут: 1.25x / 0.1x)
    break_even_reads(2.0)          ->  2    (расширенный TTL на час: 2x / 0.1x)
    break_even_reads(1.0)          ->  0    (OpenAI: премии за запись нет)

    Ищем минимальное целое n >= 0, при котором
        write_mult + n * read_mult  <=  n + 1
    то есть одна запись плюс n чтений не дороже, чем n+1 обычных запросов.

    read_mult >= 1 означает, что чтение не дешевле обычного токена, и кэш не
    окупится никогда — это ValueError, а не какое-то большое число.
    """
    if read_mult >= 1:
        raise ValueError(f"чтение по ставке {read_mult} никогда не окупит запись")
    n = (write_mult - 1) / (1 - read_mult)
    # max(0, ...) для случая, когда премии за запись нет и n выходит отрицательным
    return max(0, math.ceil(n))

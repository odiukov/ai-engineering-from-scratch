"""
Зачем нужны мультиагенты — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Правило большого пальца из урока: задача, которая укладывается в одно окно
# и в 20 вызовов инструментов, остаётся у одного агента.
WINDOW = 100_000
MAX_SINGLE_AGENT_TOOL_CALLS = 20


def inbox(messages, agent):
    """Письма, адресованные агенту, в исходном порядке.

    Сообщение — это dict с ключами "from", "to", "content".

    inbox([{"from": "r", "to": "c", "content": "notes"}], "c")
        ->  [{"from": "r", "to": "c", "content": "notes"}]
    inbox([{"from": "r", "to": "c", "content": "notes"}], "rev")  ->  []

    Это шаг 3 урока целиком: агент читает только адресованное ему, и 50k
    токенов, которые research-агент потратил на документацию, никогда не
    попадают в контекст reviewer'а.
    """
    # фильтр, а не dict-индекс: порядок сообщений несёт смысл (кто раньше
    # написал), и одному адресату может прийти несколько писем
    return [m for m in messages if m["to"] == agent]


def single_agent_context(task_tokens, stage_outputs):
    """Размер контекста ОДНОГО агента после каждого этапа.

    Один агент ничего не выбрасывает: к задаче добавляется выход каждого
    следующего этапа, и всё это едет в следующий запрос.

    single_agent_context(1000, [500, 800])  ->  [1500, 2300]
    single_agent_context(1000, [])          ->  []

    Ловушка: список ответа длиной с stage_outputs, а не на единицу больше.
    Начальный размер (сама задача) — это ещё не «после этапа».
    """
    sizes = []
    total = task_tokens
    for out in stage_outputs:
        total += out  # накопление и есть context saturation из урока
        sizes.append(total)
    return sizes


def multi_agent_contexts(task_tokens, stage_outputs):
    """Размер контекста КАЖДОГО специалиста после того, как он отработал.

    Специалист видит задачу, выход только предыдущего специалиста и свой
    собственный. Всё, что было раньше, до него не доезжает.

    multi_agent_contexts(1000, [500, 800, 300])  ->  [1500, 2300, 2100]
    multi_agent_contexts(1000, [])               ->  []

    Сравни с single_agent_context(1000, [500, 800, 300]) -> [1500, 2300, 2600]:
    на двух этапах числа совпадают (выбрасывать ещё нечего), а с третьего
    начинают расходиться. Это и есть выигрыш от разделения.
    """
    sizes = []
    for i, own in enumerate(stage_outputs):
        # предыдущий выход, а не сумма всех: граница агента — это точка,
        # где история сжимается до одного сообщения
        previous = stage_outputs[i - 1] if i > 0 else 0
        sizes.append(task_tokens + previous + own)
    return sizes


def first_overflow(sizes, window=WINDOW):
    """Индекс первого этапа, который не влез в окно; -1 если влезли все.

    first_overflow([10, 20], 100)        ->  -1
    first_overflow([60, 120], 100)       ->  1
    first_overflow([500], 100)           ->  0

    Строгое «больше», а не «больше либо равно»: ровно полное окно — это
    ещё влезло.
    """
    for i, size in enumerate(sizes):
        if size > window:
            return i
    return -1


def pipeline_seconds(stage_seconds, handoff_seconds):
    """Время pipeline: этапы подряд плюс передача между соседями.

    pipeline_seconds([1.0, 2.0, 3.0], 0.1)  ->  6.2
    pipeline_seconds([5.0], 0.1)            ->  5.0   (передавать некому)
    pipeline_seconds([], 0.1)               ->  0.0

    Ловушка: границ на одну меньше, чем этапов. Три агента — две передачи.
    """
    if not stage_seconds:
        return 0.0
    return sum(stage_seconds) + handoff_seconds * (len(stage_seconds) - 1)


def fanout_seconds(stage_seconds, handoff_seconds):
    """Время fan-out/fan-in: все параллельно, но каждый агент стоит двух передач.

    Порождение агента — это передача (split), возврат результата — вторая
    (merge). Само выполнение идёт одновременно, поэтому берём максимум.

    fanout_seconds([1.0, 2.0, 3.0], 0.1)  ->  3.6   (3.0 + 2 * 0.1 * 3)
    fanout_seconds([0.1, 0.1, 0.1], 0.1)  ->  0.7
    fanout_seconds([], 0.1)               ->  0.0

    Сравни второй пример с pipeline_seconds на тех же данных (0.5): на
    коротких этапах координация съедает весь выигрыш от параллелизма.
    """
    if not stage_seconds:
        return 0.0
    return max(stage_seconds) + 2 * handoff_seconds * len(stage_seconds)


def coordination_overhead(task_tokens, stage_outputs, summary_tokens):
    """Насколько мультиагент дороже одиночного в токенах. Минус = дешевле.

    Одиночный агент на каждом этапе перепосылает весь накопленный контекст.
    Мультиагент посылает короткий контекст, но платит summary_tokens за
    пересказ на каждой границе между агентами.

    coordination_overhead(1000, [500, 800, 300], 100)  ->  -300  (мультиагент выгоднее)
    coordination_overhead(1000, [10, 10], 500)         ->  500   (мультиагент дороже)

    Именно этот знак отвечает на вопрос «а стоит ли вообще разбивать».
    """
    single_total = sum(single_agent_context(task_tokens, stage_outputs))
    multi_total = sum(multi_agent_contexts(task_tokens, stage_outputs))
    boundaries = max(len(stage_outputs) - 1, 0)
    return multi_total + summary_tokens * boundaries - single_total


def recommend_topology(task_tokens, stage_outputs, tool_calls, distinct_prompts,
                       window=WINDOW):
    """Вердикт "single" или "multi" по правилу большого пальца из урока.

    Разбиваем, если верно хоть одно:
      * накопленный контекст одиночного агента вылезает за окно,
      * вызовов инструментов не меньше MAX_SINGLE_AGENT_TOOL_CALLS,
      * этапам нужны разные system prompt'ы.

    recommend_topology(1000, [500], 5, False)         ->  "single"
    recommend_topology(90_000, [50_000], 5, False)    ->  "multi"
    recommend_topology(1000, [500], 40, False)        ->  "multi"
    recommend_topology(1000, [500], 5, True)          ->  "multi"

    Функция намеренно не смотрит на «красиво ли выглядит архитектура».
    Мультиагент — это цена, а не награда.
    """
    sizes = single_agent_context(task_tokens, stage_outputs)
    if first_overflow(sizes, window) != -1:
        return "multi"
    if tool_calls >= MAX_SINGLE_AGENT_TOOL_CALLS:
        return "multi"
    if distinct_prompts:
        return "multi"
    return "single"

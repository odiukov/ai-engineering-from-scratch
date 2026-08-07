"""
Context engineering: бюджет окна, порядок, память — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Во сколько токенов обходится одно слово. Число грубое и модельно-зависимое,
# но воспроизводимое — а бюджет надо на чём-то считать.
TOKENS_PER_WORD = 1.3

# Каталог инструментов: сколько токенов стоит определение и к каким намерениям
# оно относится. Перенесено в заготовку целиком, тесты импортируют его.
TOOL_REGISTRY = {
    "read_file": {"tokens": 120, "categories": ("code", "files")},
    "write_file": {"tokens": 150, "categories": ("code", "files")},
    "search_code": {"tokens": 130, "categories": ("code",)},
    "run_command": {"tokens": 140, "categories": ("code", "system")},
    "create_calendar_event": {"tokens": 180, "categories": ("calendar",)},
    "send_email": {"tokens": 200, "categories": ("email",)},
    "web_search": {"tokens": 140, "categories": ("research",)},
    "query_database": {"tokens": 170, "categories": ("code", "data")},
    "generate_chart": {"tokens": 190, "categories": ("data",)},
}

# Ключевые слова, по которым классифицируется намерение запроса.
INTENT_KEYWORDS = {
    "code": ("code", "function", "bug", "error", "file", "refactor", "debug", "test"),
    "calendar": ("meeting", "schedule", "calendar", "appointment", "event"),
    "email": ("email", "mail", "inbox", "message"),
    "research": ("search", "find", "explain", "look"),
    "data": ("data", "query", "database", "chart", "sql", "analytics"),
}

# Намерение по умолчанию, если ни одно ключевое слово не сработало.
DEFAULT_INTENT = "code"

# Сколько слов оставлять от свёрнутой в сводку реплики.
SUMMARY_WORDS = 8


def count_tokens(text):
    """Грубая оценка длины в токенах: слова, умноженные на TOKENS_PER_WORD.

    count_tokens("one two three")  ->  3
    count_tokens("")               ->  0

    Результат — целое, дробную часть отбрасываем.

    Честная оговорка: настоящий токенизатор (tiktoken) режет иначе — редкие
    слова распадаются на несколько токенов. Для планирования бюджета
    приближения хватает, для биллинга — нет.
    """
    return int(len(text.split()) * TOKENS_PER_WORD) if text else 0


def truncate_to_tokens(text, max_tokens):
    """Обрезать текст по границе слова так, чтобы он влез в max_tokens.

    truncate_to_tokens("a b c d e", 3)  ->  "a b"
    truncate_to_tokens("a b c", 100)    ->  "a b c"   (влезает целиком)
    truncate_to_tokens("a b c", 0)      ->  ""

    Резать надо по словам, а не по символам: обрубок слова стоит столько же
    токенов, но читается моделью хуже.

    Проверяй итог через count_tokens: округление вниз при делении на
    TOKENS_PER_WORD легко даёт кусок на токен длиннее допустимого.
    """
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    keep = int(max_tokens / TOKENS_PER_WORD)
    # округление могло дать перебор на одно слово — подрезаем, пока не влезет
    while keep > 0 and count_tokens(" ".join(words[:keep])) > max_tokens:
        keep -= 1
    return " ".join(words[:keep])


def score_relevance(query, documents):
    """Для каждого документа — доля слов запроса, которые в нём встретились.

    score_relevance("vector search", ["vector db", "cooking"])  ->  [0.5, 0.0]
    score_relevance("", ["anything"])                           ->  [0.0]

    Сравнение без учёта регистра. Пустой запрос — нули, а не деление на ноль.

    Это грубая замена косинусной близости: считаем пересечение множеств слов
    вместо расстояния между эмбеддингами. Дёшево и удивительно рабочо для
    фильтрации явного мусора.
    """
    words = set(query.lower().split())
    if not words:
        return [0.0] * len(documents)
    return [len(words & set(d.lower().split())) / len(words) for d in documents]


def reorder_lost_in_middle(items, scores):
    """Переставить документы так, чтобы важное оказалось по краям окна.

    Самый релевантный — первым, второй по релевантности — последним, и так
    далее внутрь. Самое слабое оседает в середине.

    reorder_lost_in_middle(["a", "b", "c", "d"], [0.9, 0.1, 0.5, 0.7])
        ->  ["a", "c", "b", "d"]   (самый слабый "b" осел в середине)

    Зачем: модели хорошо видят начало и конец контекста и на 10-20% хуже —
    середину (Liu et al., 2023). Значит середину надо отдать тому, чью
    потерю не жалко.

    Ловушка: сортировать пары (score, item) нельзя. При равных score Python
    полезет сравнивать сами документы, а dict с dict не сравнивается —
    получишь TypeError на ровном месте. Сортируй по индексам.
    """
    order = sorted(range(len(items)), key=lambda i: -scores[i])
    head = [items[i] for i in order[0::2]]
    tail = [items[i] for i in order[1::2]]
    tail.reverse()
    return head + tail


def allocate_budget(components, max_tokens, generation_reserve=0):
    """Разложить компоненты по бюджету окна, обрезая всё, что не влезает.

    components — список (имя, текст, лимит или None) в порядке приоритета.
    Возвращает список (имя, текст после обрезки, его токены).

    Доступно ровно max_tokens - generation_reserve: место под ответ модели
    занимается ПЕРВЫМ, иначе окно забьётся и отвечать будет некуда.

    allocate_budget([("sys", "a b c d", None)], 3)  ->  [("sys", "a b", 2)]

    Каждый компонент режется дважды: по собственному лимиту и по остатку
    общего бюджета. Компонент, которому не досталось ничего, всё равно
    попадает в список — с пустым текстом и нулём токенов, чтобы в отчёте
    было видно, что его выбросило.
    """
    available = max_tokens - generation_reserve
    used = 0
    report = []
    for name, content, limit in components:
        text = content if limit is None else truncate_to_tokens(content, limit)
        text = truncate_to_tokens(text, max(available - used, 0))
        tokens = count_tokens(text)
        used += tokens
        report.append((name, text, tokens))
    return report


def compress_history(turns, max_tokens, keep_last=2):
    """Свернуть старые реплики в сводку, пока история не влезет в бюджет.

    turns — список (role, content). Возвращает (summary, свежие реплики),
    где summary — строка вида "Previous: user: ... | assistant: ...", а
    свежих реплик всегда остаётся не меньше keep_last.

    Сворачиваются по одной с начала, каждая — до первых SUMMARY_WORDS слов.
    Если сворачивать нечего, summary пустая строка.

    compress_history([("user", "a b c"), ("assistant", "d e f")], 100)
        ->  ("", [("user", "a b c"), ("assistant", "d e f")])

    Бюджет считается только по свежим репликам: сводка мала по построению,
    и гоняться за её точным размером — переусложнение.
    """
    turns = list(turns)
    folded = []
    while (
        len(turns) > keep_last
        and sum(count_tokens(c) for _, c in turns) > max_tokens
    ):
        role, content = turns.pop(0)
        short = " ".join(content.split()[:SUMMARY_WORDS])
        folded.append(f"{role}: {short}")
    summary = "Previous: " + " | ".join(folded) if folded else ""
    return summary, turns


def classify_intent(query):
    """Намерения запроса по ключевым словам INTENT_KEYWORDS.

    Возвращает отсортированный список намерений, набравших максимум очков.
    Очко — за каждое встретившееся ключевое слово.

    classify_intent("Fix the bug in auth.py")     ->  ["code"]
    classify_intent("Schedule a meeting")          ->  ["calendar"]
    classify_intent("hello there")                 ->  [DEFAULT_INTENT]

    Регистр не важен. Ничего не совпало — возвращаем DEFAULT_INTENT, а не
    пустой список: без намерения инструментов не выберешь вообще.
    """
    lowered = query.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits:
            scores[intent] = hits
    if not scores:
        return [DEFAULT_INTENT]
    best = max(scores.values())
    return sorted(i for i, s in scores.items() if s == best)


def select_tools(query, token_budget):
    """Выбрать инструменты под намерение запроса, не выходя за token_budget.

    Возвращает (список имён инструментов, суммарные токены). Имена
    отсортированы: порядок словаря — не то, на что стоит опираться в тесте.

    select_tools("Fix the bug in the code", 300)  ->  два-три code-инструмента

    Инструменты берутся в порядке возрастания цены: дешёвые первыми, чтобы
    в бюджет влезло их больше. Инструмент, который не влезает, пропускается,
    а перебор не прекращается — следующий может оказаться дешевле.

    Это tool pruning из урока: 50 определений на 8000 токенов ужимаются до
    1000, и модель перестаёт выбирать из мусора.
    """
    intents = set(classify_intent(query))
    matching = [
        (name, spec)
        for name, spec in TOOL_REGISTRY.items()
        if intents & set(spec["categories"])
    ]
    matching.sort(key=lambda pair: (pair[1]["tokens"], pair[0]))

    picked, total = [], 0
    for name, spec in matching:
        if total + spec["tokens"] <= token_budget:
            picked.append(name)
            total += spec["tokens"]
    return sorted(picked), total

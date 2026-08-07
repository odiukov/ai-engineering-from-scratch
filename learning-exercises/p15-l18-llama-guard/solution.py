"""
Llama Guard и классификация входа/выхода — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь собирается руками то, что в проде даёт готовая система: Llama Guard 3/4
(Meta) и NeMo Guardrails (NVIDIA). Соответствие настоящим компонентам:

    TAXONOMY                <-  MLCommons 13-hazard / Llama Guard 4 S1-S14
    classify                <-  input rail: Llama Guard на пользовательском ходе
    verdict                 <-  формат ответа Llama Guard: safe / unsafe + коды
    route                   <-  category routing: block / escalate / log / allow
    output_rail             <-  output rail: Llama Guard на ходе модели
    normalize_text          <-  предобработка против emoji smuggling / homoglyph
    attack_success_rate     <-  ASR из Huang et al., arXiv:2504.11168
    dialog_rail_report      <-  Colang dialog rail: правило поверх всей беседы
    audit_stack             <-  аудит слоя классификаторов из outputs/

Никакой модели здесь нет и быть не должно: настоящий Llama Guard — это
Llama-3.1-8B, дообученная на разметке. Мы моделируем ЛОГИКУ вокруг неё —
таксономию, маршрутизацию по категориям, рельсы входа и выхода — потому что
именно она принимает решения, а не веса.

Категории обозначены кодами S1..S14. Никаких текстов атак в файле нет:
маркеры — нейтральные слова-темы, они нужны только чтобы правило срабатывало
детерминированно.
"""

import unicodedata

# Подмножество таксономии Llama Guard 4 (S1-S14). Ключ — код категории,
# значение — кортеж маркеров-тем. Порядок ключей задаёт порядок вердикта:
# вердикт не должен зависеть от того, в каком месте текста нашлась категория.
TAXONOMY = {
    "S1": ("violence", "weapon"),
    "S2": ("fraud", "counterfeit"),
    "S8": ("dossier", "passport"),
    "S11": ("selfharm",),
    "S14": ("interpreter", "sandbox-escape"),
}

# Действия по возрастанию строгости. route возвращает самое строгое из
# применимых, поэтому порядок здесь — это и есть определение "строже".
ACTIONS = ("allow", "log", "escalate", "block")

# Маршрутизация по категориям: то самое "block S1 outright, flag S6 for human
# review, annotate S12 but allow" из урока.
DEFAULT_POLICY = {
    "S1": "block",
    "S2": "escalate",
    "S8": "escalate",
    "S11": "block",
    "S14": "block",
}

# Маркеры output rail: имя -> подстрока. Выход модели ловится отдельно от
# входа, потому что утечка секрета в ответе не видна на входном ходе.
OUTPUT_MARKERS = {
    "api_key": "sk-",
    "aws_secret": "aws_secret_access_key",
    "session_token": "user_session_token",
}

# Кириллические двойники латиницы. Карта намеренно неполная: в реальности
# есть ещё греческие (ο, ρ, α, ε) и другие кириллические. Неполнота — часть
# урока: классификатор течёт.
CYRILLIC_TO_LATIN = {
    "а": "a", "в": "b", "с": "c", "е": "e",
    "о": "o", "р": "p", "х": "x", "і": "i",
    "у": "y", "ѕ": "s",
}

# Реально невидимые кодовые точки. Ровно они, а не вся категория Mn:
# выбросив Mn целиком, мы потеряли бы легитимные диакритики.
INVISIBLE_CODEPOINTS = frozenset({
    0x200B,  # zero-width space
    0x200C,  # zero-width non-joiner
    0x200D,  # zero-width joiner
    0x2060,  # word joiner
    0xFE0F,  # variation selector-16
    0xFEFF,  # byte-order mark
})

# Слои, которые обязан иметь стек классификаторов. Порядок — порядок отчёта.
REQUIRED_LAYERS = (
    "model",
    "taxonomy",
    "input_rail",
    "output_rail",
    "dialog_rail",
    "normalization",
)


def normalize_text(text):
    """Привести текст к форме, на которой классификатор ещё работает.

    Три шага строго в этом порядке: NFKC, замена гомоглифов, удаление
    невидимых кодовых точек.

    normalize_text("pi\u200bpe")    ->  "pipe"      (zero-width space убран)
    normalize_text("pаss")          ->  "pass"      (кириллическая 'а' -> 'a')
    normalize_text("ﬁle")           ->  "file"      (NFKC разбирает лигатуру)

    Ловушка: соблазн выбросить все символы категории Mn. Тогда вместе с
    невидимками уедут легитимные диакритики, и «café» превратится в «cafe» —
    для классификатора на другом языке это уже другой текст.

    Ещё ловушка: NFKC обязан идти ПЕРВЫМ. Лигатура «ﬁ» — это один символ, и
    пока NFKC не разложил её на «fi», карта гомоглифов до буквы не доберётся.
    """
    out = unicodedata.normalize("NFKC", text)
    out = "".join(CYRILLIC_TO_LATIN.get(ch, ch) for ch in out)
    # фильтр последним: NFKC мог породить новые символы, а карта гомоглифов
    # невидимок не трогает
    return "".join(ch for ch in out if ord(ch) not in INVISIBLE_CODEPOINTS)


def classify(text, taxonomy=None, normalize=False):
    """Список кодов категорий, сработавших на тексте. Порядок — по таксономии.

    classify("plain weather question")        ->  []
    classify("weapon shop")                   ->  ["S1"]
    classify("weapon and interpreter abuse")  ->  ["S1", "S14"]
    classify("we\u200bapon", normalize=True)  ->  ["S1"]

    normalize=False — сырое сравнение, как у классификатора без предобработки.
    normalize=True — прогнать normalize_text перед сравнением.

    Порядок результата берётся из порядка ключей таксономии, а НЕ из порядка
    появления маркеров в тексте. Иначе один и тот же запрос, переставленный
    словами, даст два разных вердикта, и логи станут несравнимыми.
    """
    tax = TAXONOMY if taxonomy is None else taxonomy
    haystack = (normalize_text(text) if normalize else text).lower()
    # any() вместо накопления: категория попадает в вердикт один раз, сколько
    # бы её маркеров ни нашлось — иначе в вердикте будут дубли
    return [
        code
        for code, markers in tax.items()
        if any(m.lower() in haystack for m in markers)
    ]


def verdict(text, taxonomy=None, normalize=False):
    """Вердикт в форме Llama Guard: safe/unsafe ПЛЮС коды категорий.

    verdict("plain weather question")
        ->  {"safe": True,  "categories": [], "reason": "safe"}
    verdict("weapon shop")
        ->  {"safe": False, "categories": ["S1"], "reason": "unsafe: S1"}
    verdict("weapon and interpreter")
        ->  {"safe": False, "categories": ["S1", "S14"],
             "reason": "unsafe: S1,S14"}

    Одного слова "unsafe" мало. Настоящий Llama Guard возвращает вторую
    строку с кодами именно потому, что без категории вниз по стеку нечего
    маршрутизировать: block для S1 и annotate-but-allow для S12 — разные
    решения, а «unsafe» у них одинаковый.
    """
    cats = classify(text, taxonomy=taxonomy, normalize=normalize)
    if not cats:
        return {"safe": True, "categories": [], "reason": "safe"}
    return {
        "safe": False,
        "categories": cats,
        "reason": "unsafe: " + ",".join(cats),
    }


def route(categories, policy=None):
    """Самое строгое действие по списку категорий.

    route([])                ->  "allow"
    route(["S2"])            ->  "escalate"
    route(["S2", "S1"])      ->  "block"    (block строже escalate)
    route(["S99"])           ->  "escalate" (незнакомая категория не проходит)

    Неизвестная категория НЕ проваливается в "allow". Таксономия растёт —
    S1-S13 стали S1-S14, — и код, написанный под старую версию, обязан вести
    себя консервативно на всём, чего он не знает.

    Незнакомое действие в политике — ValueError. Опечатка "bock" вместо
    "block" иначе тихо станет самым слабым действием: её нет в ACTIONS,
    сравнение по индексу упадёт или соврёт.
    """
    pol = DEFAULT_POLICY if policy is None else policy
    worst = "allow"
    for cat in categories:
        # fail-safe по умолчанию: незнакомая категория идёт на человека
        action = pol.get(cat, "escalate")
        if action not in ACTIONS:
            raise ValueError(f"unknown action: {action!r}")
        if ACTIONS.index(action) > ACTIONS.index(worst):
            worst = action
    return worst


def output_rail(text, markers=None):
    """Имена маркеров, сработавших на ВЫХОДЕ модели. Отсортированы.

    output_rail("here is a summary")                 ->  []
    output_rail("token sk-abcdef0123")               ->  ["api_key"]
    output_rail("aws_secret_access_key=x, sk-yy")    ->  ["api_key", "aws_secret"]

    Отдельный слой от classify не для симметрии. Вход может быть безобидным
    («покажи конфиг»), а выход — утечкой ключа. Input rail на такой паре не
    сработает ни при какой таксономии.
    """
    mk = OUTPUT_MARKERS if markers is None else markers
    low = text.lower()
    return sorted(name for name, needle in mk.items() if needle.lower() in low)


def attack_success_rate(cases, taxonomy=None, normalize=False):
    """ASR: доля случаев, где ожидаемая категория НЕ поймана.

    cases — список пар (текст, ожидаемый код категории).

    attack_success_rate([("weapon", "S1")])                    ->  0.0
    attack_success_rate([("we\u200bapon", "S1")])             ->  1.0
    attack_success_rate([("we\u200bapon", "S1")], normalize=True)  ->  0.0
    attack_success_rate([])                                    ->  0.0

    Это метрика из Huang et al. (arXiv:2504.11168): 100% ASR на emoji
    smuggling у шести guard-систем, 72.54% на NeMo Guard Detect. Считать её
    надо на СВОЁМ наборе: цифра из статьи получена под адверсарным подбором,
    а обычные пользователи дают совсем другое распределение.

    Пустой набор даёт 0.0, а не деление на ноль. Но 0.0 на пустом наборе
    ничего не доказывает — это отсутствие измерения, а не отсутствие атак.
    """
    if not cases:
        return 0.0
    missed = sum(
        1
        for text, expected in cases
        if expected not in classify(text, taxonomy=taxonomy, normalize=normalize)
    )
    return missed / len(cases)


def dialog_rail_report(turns, rail):
    """Rail уровня БЕСЕДЫ: считает упоминания темы по всем ходам сразу.

    rail — {"topic": str, "markers": (...), "max_mentions": int}.
    Возвращает {"topic", "mentions", "fired"}, где mentions — индексы ходов.

    rail = {"topic": "diagnosis", "markers": ("diagnos",), "max_mentions": 2}
    dialog_rail_report(["hi", "what is my diagnosis"], rail)
        ->  {"topic": "diagnosis", "mentions": [1], "fired": False}
    dialog_rail_report(["diagnosis?", "ok, self-diagnosis then"], rail)
        ->  {"topic": "diagnosis", "mentions": [0, 1], "fired": True}

    Смысл, ради которого NeMo Guardrails отделяет dialog rails от input/output
    rails: каждый отдельный ход может быть safe по таксономии, а беседа в
    целом — три перефразировки одного запрещённого вопроса. Rail на одном
    ходе такое не видит по определению: у него нет памяти.

    max_mentions <= 0 — ValueError. Rail, срабатывающий на нулевом упоминании,
    блокирует пустую беседу и означает, что тема в чат-боте запрещена вообще,
    а это уже задача не rail, а системного промпта.
    """
    if rail["max_mentions"] <= 0:
        raise ValueError("max_mentions must be positive")
    markers = tuple(m.lower() for m in rail["markers"])
    mentions = [
        i for i, turn in enumerate(turns)
        if any(m in turn.lower() for m in markers)
    ]
    return {
        "topic": rail["topic"],
        "mentions": mentions,
        "fired": len(mentions) >= rail["max_mentions"],
    }


def audit_stack(config):
    """Пропуски в стеке классификаторов. Порядок — REQUIRED_LAYERS.

    audit_stack({k: True for k in REQUIRED_LAYERS})  ->  []
    audit_stack({"model": "llama-guard-4"})
        ->  ["taxonomy", "input_rail", "output_rail", "dialog_rail",
             "normalization"]

    Ключ, которого нет, и ключ со значением False/""/0 — один и тот же
    пропуск. «Поле есть, значение пустое» в аудите безопасности — это не
    наполовину сделанный слой, это отсутствующий слой.

    Порядок из REQUIRED_LAYERS, а не из config: отчёты за разные месяцы
    должны сравниваться построчно.
    """
    return [layer for layer in REQUIRED_LAYERS if not config.get(layer)]

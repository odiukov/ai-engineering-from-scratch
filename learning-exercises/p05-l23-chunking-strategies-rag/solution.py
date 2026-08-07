"""
Стратегии чанкинга для RAG — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re


def chunk_fixed(text, size, overlap=0):
    """Fixed-чанкинг: режем текст каждые size символов, соседи делят overlap символов.

    chunk_fixed("abcdefgh", 3)             ->  ['abc', 'def', 'gh']
    chunk_fixed("abcdefgh", 4, overlap=2)  ->  ['abcd', 'cdef', 'efgh', 'gh']
    chunk_fixed("", 3)                     ->  []

    Свойство, за которое его любят как baseline: при overlap=0 чанки
    склеиваются обратно ровно в исходный текст, символ в символ.

    Ловушка. Шаг равен size - overlap. Если overlap >= size, шаг получается
    нулевой или отрицательный, и наивный цикл крутится вечно, съедая память.
    Такое сочетание аргументов — это ValueError, а не «ну как-нибудь».
    Размер тоже проверь: size <= 0 бессмысленен.

    Зачем это в AI: самый дешёвый baseline индексации. Бенчмарки 2026 года
    показывают, что overlap часто не даёт прироста recall, но удваивает
    стоимость индекса, — поэтому overlap=0 по умолчанию.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    step = size - overlap
    if step <= 0:
        # именно здесь живёт бесконечный цикл, если не проверить
        raise ValueError("overlap must be less than size")
    # ничего не выбрасываем и не strip-аем: иначе ломается склейка обратно
    return [text[i:i + size] for i in range(0, len(text), step)]


def split_sentences(text):
    """Разбить текст на предложения по границам . ! ? с последующим пробелом.

    split_sentences("Hi there. How are you? Fine!")
        ->  ['Hi there.', 'How are you?', 'Fine!']
    split_sentences("One sentence only")  ->  ['One sentence only']
    split_sentences("   ")                ->  []

    Знак препинания остаётся в конце предложения, пробелы-разделители
    выбрасываются. Регулярка со взглядом назад `(?<=[.!?])\\s+` делает это
    в одну строку.

    Ловушка. На пустом (или состоящем из пробелов) тексте split вернёт
    список с одной пустой строкой — её надо отфильтровать, иначе вниз по
    цепочке поедет фантомное предложение нулевой длины.
    """
    stripped = text.strip()
    if not stripped:
        return []
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    return [p.strip() for p in parts if p.strip()]


def chunk_recursive(text, size, seps=("\n\n", "\n", ". ", " ")):
    """Recursive-чанкинг: пробуем резать по первому подходящему сепаратору из списка.

    Это ровно то, что делает `RecursiveCharacterTextSplitter` из LangChain,
    только руками. Логика: берём первый сепаратор, который вообще встречается
    в тексте, режем по нему и склеиваем куски обратно в чанки, пока помещается
    в size. Кусок, который сам длиннее size, дорезаем следующими сепараторами;
    когда сепараторы кончились — падаем в chunk_fixed.

    chunk_recursive("short text", 100)         ->  ['short text']
    chunk_recursive("aaa\\n\\nbbb\\n\\nccc", 5)  ->  ['aaa', 'bbb', 'ccc']
    chunk_recursive("abcdefgh", 3)             ->  ['abc', 'def', 'gh']

    Порядок сепараторов — это порядок «что жальче терять»: сначала абзацы,
    потом строки, потом предложения, и только в самом конце слова.

    Ловушка. Fallback обязан быть настоящим: если ни один сепаратор не
    подошёл (одно длинное слово без пробелов), функция всё равно не имеет
    права вернуть чанк длиннее size. И ещё: рекурсия должна идти по ХВОСТУ
    списка сепараторов, иначе тот же сепаратор попробуется снова и снова.

    Зачем это в AI: дефолт 2026 года. Recursive 512 без overlap — та точка,
    с которой начинают любой RAG, и которая в бенчмарках Vectara обходит
    semantic-чанкинг.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep not in text:
            continue
        rest = seps[i + 1:]
        chunks = []
        buf = ""
        for part in text.split(sep):
            if len(part) > size:
                # кусок сам по себе не помещается: сбрасываем буфер и
                # дорезаем его следующими по «жалости» сепараторами
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(part, size, rest))
                continue
            candidate = buf + sep + part if buf else part
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = part
        if buf:
            chunks.append(buf)
        return [c.strip() for c in chunks if c.strip()]
    # сепараторов не осталось — режем грубо, но гарантируем предел size
    return chunk_fixed(text, size)


def chunk_sentence_window(text, window=3, stride=None):
    """Sentence-window: чанк — это окно из window подряд идущих предложений.

    stride задаёт шаг окна. По умолчанию stride = window, то есть окна не
    пересекаются и вместе дают ровное разбиение всех предложений.

    chunk_sentence_window("A one. B two. C three. D four.", window=2)
        ->  ['A one. B two.', 'C three. D four.']
    chunk_sentence_window("A one. B two. C three. D four.", window=2, stride=1)
        ->  ['A one. B two.', 'B two. C three.', 'C three. D four.']
    chunk_sentence_window("", window=2)  ->  []

    Ловушка. Наивный `range(0, len(sentences), stride)` при stride < window
    доезжает до конца и штампует хвостовые огрызки («D four.» отдельным
    чанком), которые уже целиком содержатся в предыдущем окне. Окно, которое
    достало до конца списка, должно быть последним.
    Нулевой или отрицательный window/stride — это ValueError.

    Зачем это в AI: по качеству догоняет semantic-чанкинг на текстах до ~5k
    токенов, а стоит в разы дешевле — эмбеддер вызывать не надо.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if stride is None:
        stride = window
    if stride <= 0:
        raise ValueError("stride must be positive")
    sentences = split_sentences(text)
    chunks = []
    i = 0
    n = len(sentences)
    while i < n:
        chunks.append(" ".join(sentences[i:i + window]))
        if i + window >= n:
            break  # окно упёрлось в конец — дальше будут только огрызки
        i += stride
    return chunks


def sentence_similarity(a, b):
    """Похожесть двух предложений: коэффициент Жаккара по множествам слов.

    Пересечение слов делим на объединение. Регистр и пунктуация не важны.

    sentence_similarity("the cat sat", "the cat ran")  ->  0.5
    sentence_similarity("cat", "dog")                  ->  0.0
    sentence_similarity("Cat!", "cat")                 ->  1.0

    Ловушка. Пустая строка не имеет слов, объединение пустое — деление на
    ноль. Такой случай возвращает 0.0.

    Зачем это в AI: здесь стоит заглушка. В проде на этом месте была бы
    косинусная близость эмбеддингов (`encoder.encode(...)` + скалярное
    произведение нормированных векторов). Жаккар по словам ловит ту же
    идею — «об одном ли эти два предложения» — но без модели, на голой
    стандартной библиотеке.
    """
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def chunk_semantic(text, threshold=0.3, min_chars=80, max_chars=400):
    """Semantic-чанкинг: рвём там, где похожесть соседних предложений падает ниже threshold.

    Алгоритм: идём по предложениям, считаем sentence_similarity текущего с
    предыдущим. Похожесть упала ниже threshold — начинаем новый чанк, но
    только если накопленный чанк уже дорос до min_chars. Готовые чанки
    длиннее max_chars дорезаем через chunk_recursive.

    chunk_semantic(text, threshold=0.0)         ->  один чанк: рвать негде,
                                                    похожесть всегда >= 0
    chunk_semantic(text, threshold=1.1, min_chars=0)
                                                ->  по чанку на предложение

    Ловушка (та самая, из-за которой semantic проигрывает recursive в
    бенчмарках). Без пола min_chars алгоритм штампует огрызки в 40 токенов:
    одно случайное предложение не похоже на соседа — и вот у нас чанк из
    трёх слов, который ничего не найдёт. Пол обязателен, и последний чанк —
    тоже чанк: если он не дорос, его надо приклеить к предыдущему, а не
    оставить огрызком.

    Зачем это в AI: сохраняет тематическую цельность чанка. Дороже
    recursive (нужен эмбеддер на каждое предложение) и, по данным Vectara
    2026, чаще проигрывает ему — но на текстах с резкой сменой тем выигрывает.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = sentence_similarity(sentences[i - 1], sentences[i])
        current_len = len(" ".join(groups[-1]))
        if sim < threshold and current_len >= min_chars:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    # пол min_chars: хвостовой огрызок приклеиваем к предыдущей группе,
    # она уже гарантированно доросла до min_chars
    if len(groups) > 1 and len(" ".join(groups[-1])) < min_chars:
        groups[-2].extend(groups.pop())

    result = []
    for group in groups:
        joined = " ".join(group)
        if len(joined) > max_chars:
            result.extend(chunk_recursive(joined, max_chars))
        else:
            result.append(joined)
    return result


def chunk_parent_child(text, parent_size=400, child_size=100):
    """Parent-child: мелкие children для поиска, крупные parents для контекста.

    Возвращает список словарей, по одному на child, в порядке следования в
    тексте:
        {"child": str, "parent_idx": int, "parent": str}

    Родители нарезаются chunk_recursive(text, parent_size), каждый родитель
    внутри себя нарезается на children через chunk_recursive(parent,
    child_size). parent_idx — номер родителя, по нему потом дедуплицируют.

    chunk_parent_child("aaa\\n\\nbbb", parent_size=100, child_size=100)
        ->  [{'child': 'aaa\\n\\nbbb', 'parent_idx': 0, 'parent': 'aaa\\n\\nbbb'}]

    Ловушка. Children режутся ВНУТРИ родителя, а не по всему тексту — иначе
    чанк пересечёт границу родителя (а в проде — границу документа), и
    parent_idx станет враньём. Никогда не давай чанку пересекать границу
    документа: сначала режь каждый документ отдельно, потом складывай.

    Зачем это в AI: деградирует мягко. Даже если child попал в топ по
    случайности, в LLM уедет осмысленный крупный parent, а не обрывок.
    """
    if child_size > parent_size:
        raise ValueError("child_size must not exceed parent_size")
    parents = chunk_recursive(text, parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        for child in chunk_recursive(parent, child_size):
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parents(query, mapping, top_k=3):
    """Поиск по children, ответ — родители, БЕЗ повторов.

    Каждый child оценивается через sentence_similarity(query, child), берутся
    top_k лучших children, из них собираются родители. Если два разных
    ребёнка привели к одному родителю, родитель попадает в ответ один раз —
    поэтому результат может быть короче top_k.

    Порядок ответа — по убыванию похожести лучшего ребёнка. При равных
    оценках побеждает тот, кто раньше в mapping (детерминированность).

    retrieve_parents("payment fee", mapping, top_k=2)  ->  ['... parent text ...']
    retrieve_parents("anything", [], top_k=3)          ->  []

    Ловушка. Дедуплицировать надо по parent_idx, а не по тексту родителя:
    два одинаковых по тексту абзаца из разных мест документа — это разные
    родители. И top_k ограничивает число просмотренных children, а не число
    возвращённых родителей.
    Нулевой или отрицательный top_k — это ValueError.

    Зачем это в AI: без дедупликации один и тот же абзац уедет в промпт
    трижды и съест контекстное окно впустую.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    # ключ (-score, i): сортировка по убыванию похожести, ничьи — по порядку
    order = sorted(
        range(len(mapping)),
        key=lambda i: (-sentence_similarity(query, mapping[i]["child"]), i),
    )
    seen = set()
    parents = []
    for i in order[:top_k]:
        p_idx = mapping[i]["parent_idx"]
        if p_idx in seen:
            continue
        seen.add(p_idx)
        parents.append(mapping[i]["parent"])
    return parents

"""
Групповой чат и выбор говорящего — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Сообщение в общем пуле — пара (говорящий, текст). Пул общий: его целиком
# видят все агенты, это и есть GroupChat из AutoGen/AG2.
TERMINATION_TOKEN = "TERMINATE"
DEFAULT_MAX_ROUNDS = 10


def keyword_score(text, keywords):
    """Сколько ключевых слов роли встретилось в тексте. Регистр не важен.

    keyword_score("Please REVIEW the code", ["review", "test"])  ->  1
    keyword_score("nothing relevant here", ["review"])           ->  0

    Считаем именно СЛОВА-совпадения, а не вхождения: слово, встретившееся
    трижды, даёт 1, а не 3. Иначе один многословный агент перетягивал бы
    выбор на себя простым повтором.

    Это грубая замена LLM-селектора: настоящий читает пул и отвечает именем.
    """
    low = text.lower()
    return sum(1 for kw in keywords if kw.lower() in low)


def round_robin_selector(pool, names):
    """Следующий говорящий по кругу: тот, кто идёт за последним.

    round_robin_selector([], ["a", "b"])              ->  'a'
    round_robin_selector([("a", "hi")], ["a", "b"])   ->  'b'
    round_robin_selector([("b", "hi")], ["a", "b"])   ->  'a'

    Детерминированно и дёшево, но контекст игнорируется полностью: очередь
    дойдёт до юриста, даже если обсуждают отступы в коде.

    Ловушка: последним мог говорить агент, которого нет в names (его уже
    выгнали из чата). Это ошибка конфигурации, а не повод молча начать
    круг заново.
    """
    if not names:
        raise ValueError("names must not be empty")
    if not pool:
        return names[0]
    last = pool[-1][0]
    if last not in names:
        raise ValueError(f"last speaker {last!r} is not in the team")
    return names[(names.index(last) + 1) % len(names)]


def relevance_selector(pool, specialties, candidates=None):
    """Следующий говорящий по релевантности последнего сообщения.

    specialties — dict {имя агента: список его ключевых слов}.
    candidates — кого вообще рассматриваем; None означает всех.

    relevance_selector([("m", "fix the code")],
                       {"coder": ["code"], "lawyer": ["law"]})   ->  'coder'
    relevance_selector([], {"coder": ["code"], "lawyer": ["law"]})  ->  'coder'

    Ничья решается порядком в specialties (или в candidates) — первый
    подходящий. Без этого правила селектор был бы недетерминированным.

    Осторожно: сам по себе этот селектор НЕ запрещает повтор. Если агент
    сам произносит свои ключевые слова, он же и получит следующий ход — и
    так до конца лимита. Чат выродится в монолог.
    """
    pool_names = list(specialties) if candidates is None else list(candidates)
    if not pool_names:
        raise ValueError("no candidates to choose from")
    if not pool:
        return pool_names[0]
    last_text = pool[-1][1]
    # max возвращает ПЕРВЫЙ максимум — отсюда и детерминированная ничья
    return max(pool_names, key=lambda name: keyword_score(last_text, specialties[name]))


def auto_selector(pool, specialties, allow_repeat=False):
    """Релевантность плюс запрет говорить дважды подряд.

    auto_selector([("coder", "code code")], {"coder": ["code"], "rev": ["review"]})
        ->  'rev'
    auto_selector([("coder", "code code")], {"coder": ["code"], "rev": ["review"]}, True)
        ->  'coder'

    allow_repeat=True — это и есть hot speaker из урока: тот же агент
    забирает ход снова и снова.

    Ловушка: если в команде ровно один агент, запрет повтора не должен
    оставить селектор без кандидатов. Один агент имеет право говорить
    подряд — выбора всё равно нет.
    """
    candidates = list(specialties)
    if not allow_repeat and pool:
        without_last = [n for n in candidates if n != pool[-1][0]]
        if without_last:
            candidates = without_last
    return relevance_selector(pool, specialties, candidates)


def is_terminated(pool, token=TERMINATION_TOKEN, max_rounds=DEFAULT_MAX_ROUNDS):
    """Пора ли останавливать чат: сработал стоп-токен или упёрлись в лимит.

    is_terminated([("a", "done TERMINATE")])   ->  True
    is_terminated([("a", "still working")])    ->  False
    is_terminated([("a", "x")] * 10)           ->  True   (лимит по умолчанию)

    Два условия, а не одно: токен — это «мы закончили», лимит — «мы
    сдались». Без второго LLM-селектор способен крутить чат бесконечно.
    """
    if max_rounds is not None and len(pool) >= max_rounds:
        return True
    if not pool:
        return False
    return pool[-1][1].strip().endswith(token)


def run_groupchat(policies, selector, max_rounds=DEFAULT_MAX_ROUNDS, token=TERMINATION_TOKEN):
    """Прогон группового чата. Возвращает пул сообщений — пары (кто, текст).

    policies — dict {имя агента: функция(pool) -> текст реплики}.
    selector — функция(pool) -> имя следующего говорящего.

    Работа менеджера ровно в этом цикле: спросить селектор, вызвать агента,
    дописать реплику в общий пул, проверить условие остановки.

    Ловушка: селектор может вернуть имя, которого нет в policies (LLM
    выдумал агента). Молча проглотить — значит получить KeyError в
    случайном месте; лучше ValueError сразу и с именем.
    """
    pool = []
    while not is_terminated(pool, token, max_rounds):
        name = selector(pool)
        if name not in policies:
            raise ValueError(f"selector picked an unknown agent {name!r}")
        pool.append((name, policies[name](pool)))
    return pool


def speaker_counts(pool):
    """Сколько реплик у каждого агента, в порядке первого появления.

    speaker_counts([("a", "x"), ("b", "y"), ("a", "z")])  ->  {'a': 2, 'b': 1}
    speaker_counts([])                                     ->  {}

    Это метрика speaker balance из чек-листа урока.
    """
    counts = {}
    for name, _ in pool:
        counts[name] = counts.get(name, 0) + 1
    return counts


def dominance(pool):
    """Доля реплик самого говорливого агента. 1.0 — это монолог.

    dominance([("a", "x"), ("b", "y")])  ->  0.5
    dominance([("a", "x"), ("a", "y")])  ->  1.0

    Пустой чат — это не «монолог никого», а отсутствие данных: ValueError.
    """
    if not pool:
        raise ValueError("dominance of an empty pool is undefined")
    return max(speaker_counts(pool).values()) / len(pool)

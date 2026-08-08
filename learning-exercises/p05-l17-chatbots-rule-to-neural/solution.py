"""
Чат-боты: от правил к нейросетям и агентам — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re


def reflect(text, swaps):
    """Трюк ELIZA: разворот местоимений в реплике пользователя.

    Текст режется на слова по пробелам, каждое слово ищется в swaps в нижнем
    регистре и заменяется, если нашлось. Замена одним проходом.

    SWAPS = {"i": "you", "am": "are", "my": "your", "you": "i", "are": "am"}
    reflect("I am sad about my job", SWAPS)  ->  "you are sad about your job"
    reflect("i you", SWAPS)                  ->  "you i"

    Ловушка: проход обязан быть ОДИН. Если сначала заменить все "i" на "you",
    а потом все "you" на "i", то из "i you" получится "i i" — обе половины
    фразы схлопнутся в одно местоимение.

    Вторая ловушка, она же ограничение подхода: разбиение по пробелам, значит
    "my," с запятой в swaps не найдётся и останется как есть. Weizenbaum в
    1966 разбирал знаки препинания отдельно; здесь этого нет намеренно —
    чтобы было видно, из чего именно состоит хрупкость правил.
    """
    return " ".join(swaps.get(word.lower(), word) for word in text.split())


def rule_based_respond(text, patterns, swaps, fallback="I don't understand."):
    """ELIZA: первый совпавший шаблон выигрывает, его группы едут в ответ.

    patterns — список пар (регулярное выражение, шаблон ответа). Совпадение
    ищется от начала строки (re.match) и без учёта регистра. Захваченные
    группы прогоняются через reflect и подставляются в шаблон через format.
    Не совпало ничего — возвращается fallback.

    P = [(r"i feel (.+)", "Why do you feel {0}?"), (r"(.*)", "Tell me more.")]
    S = {"my": "your"}
    rule_based_respond("I feel bad about my job", P, S)
        ->  "Why do you feel bad about your job?"
    rule_based_respond("The sky is blue", P, S)  ->  "Tell me more."
    rule_based_respond("hello", [], S)           ->  "I don't understand."

    Ловушка: порядок в списке — это приоритет. Шаблон-ловушка r"(.*)" ловит
    вообще всё, поэтому стоит последним; поставь его первым и остальные 199
    правил ELIZA перестанут срабатывать.

    За полвека этот механизм не изменился: ALICE — те же пары
    «шаблон-ответ», просто их 40 000. Больше правил покупают покрытие, но
    никогда не общность.
    """
    stripped = text.strip()
    for pattern, template in patterns:
        match = re.match(pattern, stripped, re.IGNORECASE)
        if match:
            # группа может быть None, если она в необязательной части regex
            groups = [reflect(g, swaps) if g else "" for g in match.groups()]
            return template.format(*groups)
    return fallback


def jaccard_similarity(a, b):
    """Похожесть двух текстов как |пересечение| / |объединение| множеств слов.

    Слово — цепочка латинских букв в нижнем регистре; порядок и повторы не
    важны. Оба текста без слов дают 0.0, а не деление на ноль.

    jaccard_similarity("cancel my order", "cancel the order")  ->  0.5
    jaccard_similarity("reset password", "reset password")     ->  1.0
    jaccard_similarity("reset password", "track shipment")     ->  0.0

    В уроке на этом месте стоит sentence-transformers и косинус эмбеддингов.
    Жаккар — его дешёвый предок: ловит пересечение слов, но не понимает, что
    "cancel" и "call off" про одно и то же. Ровно та причина, по которой
    retrieval переехал на эмбеддинги.
    """
    tokens_a = set(re.findall(r"[a-z]+", a.lower()))
    tokens_b = set(re.findall(r"[a-z]+", b.lower()))
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def faq_respond(text, faq, threshold=0.5):
    """Retrieval-бот: самый похожий вопрос из FAQ, иначе отказ.

    faq — список пар (вопрос, ответ). Возвращается ответ на самый похожий по
    jaccard_similarity вопрос; при равенстве побеждает первый по порядку.
    Если лучшая похожесть строго меньше threshold, возвращается None.

    FAQ = [("how do i reset my password", "Settings > Security."),
           ("how do i cancel my order", "Orders > Cancel.")]
    faq_respond("how do i cancel my order", FAQ)        ->  "Orders > Cancel."
    faq_respond("what is the weather", FAQ)             ->  None
    faq_respond("what is the weather", FAQ, 0.0)        ->  первый ответ

    None — это не сбой, а главное проектное решение: бот, который умеет
    сказать «не знаю», отдаёт вопрос эскалации вместо того чтобы выдумать.
    Генерации здесь нет, значит и галлюцинировать нечем.
    """
    best_answer = None
    best_score = 0.0
    for question, answer in faq:
        score = jaccard_similarity(text, question)
        # строго больше: при равенстве остаётся тот, что раньше в списке
        if score > best_score or best_answer is None:
            best_score = score
            best_answer = answer
    if best_answer is None or best_score < threshold:
        return None
    return best_answer


def is_destructive_action(
    text, danger_words=("delete", "cancel", "charge", "refund", "transfer")
):
    """Есть ли в реплике признак необратимого действия. Регистр не важен.

    is_destructive_action("please cancel my order")  ->  True
    is_destructive_action("How do I RESET it?")      ->  False
    is_destructive_action("cancellation policy")     ->  True

    Проверка идёт по подстроке, а не по слову целиком: "cancellation"
    считается опасным. Для guardrail это осознанный перекос в сторону
    ложных срабатываний — лишний раз спросить подтверждение дешевле, чем
    один раз списать деньги без спроса.
    """
    lowered = text.lower()
    return any(word in lowered for word in danger_words)


def agent_loop(user_message, tools, llm, max_steps=5):
    """Цикл LLM-агента: план -> вызов инструмента -> наблюдение -> решение.

    llm — функция llm(history, tools), возвращающая словарь либо с ключом
    "tool_call" ({"name": ..., "arguments": {...}}), либо с "content".
    tools — словарь имя -> вызываемая функция.

    Правила цикла:
      * пришёл "content" — это финальный ответ, вернуть его сразу;
      * пришёл tool_call с неизвестным именем — в history уходит сообщение
        роли "tool" с текстом f"error: unknown tool {name!r}", цикл идёт дальше;
      * arguments не словарь — туда же уходит
        f"error: arguments must be a dict, got {тип}";
      * иначе инструмент вызывается как fn(**arguments), результат уходит в
        history сообщением роли "tool"; если он бросил исключение, вместо
        падения туда уходит "error: tool ... failed: Тип: сообщение";
      * бюджет исчерпан — вернуть ровно
        "I could not complete the task in the step budget."

    history стартует с [{"role": "user", "content": user_message}] и растёт
    на два сообщения за каждый вызов инструмента: "assistant" с tool_call и
    "tool" с результатом.

    Ловушка: ошибка инструмента не должна ронять цикл. Агент обязан увидеть
    текст ошибки в history и попробовать другой ход — именно так модель
    узнаёт, что напутала с именем или аргументами.

    Вторая ловушка: без max_steps агент, зациклившийся на одном инструменте,
    будет крутиться вечно. Бюджет шагов — не оптимизация, а предохранитель.
    """
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools)
        tool_call = response.get("tool_call")
        if not tool_call:
            return response["content"]

        name = tool_call.get("name")
        args = tool_call.get("arguments")
        history.append({"role": "assistant", "tool_call": tool_call})

        if not isinstance(name, str) or name not in tools:
            content = f"error: unknown tool {name!r}"
        elif not isinstance(args, dict):
            content = f"error: arguments must be a dict, got {type(args).__name__}"
        else:
            try:
                content = tools[name](**args)
            except Exception as exc:
                content = f"error: tool {name!r} failed: {type(exc).__name__}: {exc}"
        history.append({"role": "tool", "name": str(name), "content": content})
    return "I could not complete the task in the step budget."


def hybrid_chat(text, faq, tools, llm, structured_flow, threshold=0.6):
    """Маршрутизация 2026 года: правила -> retrieval -> агент. Три ветки, в этом порядке.

    1. is_destructive_action -> structured_flow(text): необратимое действие
       уходит в детерминированный сценарий, LLM его не касается;
    2. faq_respond(text, faq, threshold) -> готовый ответ, если нашёлся;
    3. иначе agent_loop(text, tools, llm).

    Порядок веток и есть весь смысл. Поменяй местами 1 и 2 — и «cancel my
    order» уйдёт в FAQ, где найдётся похожая фраза, и заказ отменится без
    подтверждения.

    hybrid_chat("please cancel my order", FAQ, {}, llm, flow)  ->  flow(...)
    hybrid_chat("how do i reset my password", FAQ, {}, llm, flow)  ->  ответ FAQ
    hybrid_chat("tell me a joke", FAQ, {}, llm, flow)  ->  результат agent_loop

    Ни одна архитектура не покрывает всё: правила надёжны и узки, retrieval
    не галлюцинирует и не обобщает, агент обобщает и врёт. Роутер собирает из
    трёх слабостей одну работающую систему.
    """
    if is_destructive_action(text):
        return structured_flow(text)

    answer = faq_respond(text, faq, threshold)
    if answer is not None:
        return answer

    return agent_loop(text, tools, llm)

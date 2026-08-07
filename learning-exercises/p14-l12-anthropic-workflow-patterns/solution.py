"""
Workflow-паттерны Anthropic: простое вместо сложного — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Пять паттернов из «Building Effective Agents» — это пять маленьких функций.
Весь смысл поста Schluntz и Zhang в том, что фреймворк тут ничего не
добавляет: граф вызовов принадлежит инженеру, и его видно целиком.

Соответствие паттернам из поста:

    prompt_chain          <-  prompt chaining (+ программный gate между шагами)
    route                 <-  routing (классификатор + диспетчер)
    parallel_vote         <-  parallelization, форма voting
    parallel_sections     <-  parallelization, форма sectioning
    orchestrator_workers  <-  orchestrator-workers
    evaluator_optimizer   <-  evaluator-optimizer
    pick_pattern          <-  выбор паттерна под задачу (или отказ в пользу агента)

LLM здесь — обычная функция строка -> строка, которую передают параметром.
Никакой сети: в тестах это замыкание над словарём, в бою — вызов Messages API.
"""

# Пять паттернов поста плюс честный шестой вариант: workflow не подходит.
PATTERNS = (
    "prompt-chaining",
    "routing",
    "parallelization",
    "orchestrator-workers",
    "evaluator-optimizer",
    "agent",
)


def prompt_chain(text, llm, steps, gate=None):
    """Prompt chaining: выход шага N — вход шага N+1. Вернуть (результат, трассу).

    steps — последовательность пар (метка, шаблон). В шаблоне подставляется
    {text}. gate — необязательная ПРОГРАММНАЯ проверка (метка, выход) ->
    (ok, причина).

    prompt_chain("raw", llm, (("sum", "summarize: {text}"),))
        ->  (ответ модели, [("sum", ответ модели)])
    prompt_chain("raw", llm, ())  ->  ("raw", [])

    Gate — это то, ради чего цепочку и разбивают на шаги: между вызовами
    можно проверить результат КОДОМ, а не следующим промптом. Не прошло —
    цепочка останавливается, в трассу уходит пара ("<метка>:gate", причина).

    Возвращается последний ПРОШЕДШИЙ результат, а не забракованный. Отдать
    наружу то, что gate только что отверг, — значит поставить проверку и
    проигнорировать её вердикт.
    """
    current = text
    trace = []
    for label, template in steps:
        output = llm(template.format(text=current))
        trace.append((label, output))
        if gate is not None:
            ok, reason = gate(label, output)
            if not ok:
                trace.append((f"{label}:gate", reason))
                return current, trace
        current = output
    return current, trace


def route(text, classifier, handlers, threshold=0.0):
    """Routing: классификатор выбирает обработчик. Вернуть (метка, результат).

    classifier(text) -> (метка, уверенность). handlers — словарь метка ->
    функция от text. Ключ "default" ловит незнакомые метки, ключ "escalate" —
    низкую уверенность.

    route("I want my money back", clf, handlers)  ->  ("refund", "refund filed")
    route("...", clf, handlers, threshold=0.9)    ->  ("escalate", "to human")

    Порог уверенности — граница ответственности. Ниже порога специалист не
    вызывается ВООБЩЕ: дешёвый неверный ответ в поддержке дороже эскалации.
    threshold=0.0 по умолчанию означает «никогда не эскалировать».

    Незнакомая метка без "default" — KeyError. Тихо ответить пустой строкой
    хуже: классификатор начнёт придумывать категории, и никто не заметит.
    """
    label, confidence = classifier(text)
    if confidence < threshold:
        label = "escalate"
    handler = handlers.get(label, handlers.get("default"))
    if handler is None:
        raise KeyError(label)
    return label, handler(text)


def parallel_vote(prompt, llm, n=5):
    """Parallelization (voting): n прогонов одного промпта. Вернуть (ответ, счёт).

    Счёт — словарь ответ -> сколько раз он выпал.

    Пусть llm выдаёт по порядку yes, yes, no, yes, no.
    parallel_vote("safe to ship?", llm, n=5)  ->  ("yes", {"yes": 3, "no": 2})
    parallel_vote("safe to ship?", llm, n=1)  ->  ("yes", {"yes": 1})
    parallel_vote("safe to ship?", llm, n=0)  ->  ValueError

    Голосование сглаживает дисперсию модели: одиночный прогон на пограничном
    вопросе — это монетка, пять прогонов уже дают распределение.

    При ничьей побеждает ответ, встретившийся РАНЬШЕ. Правило нужно только
    ради детерминизма: без него один и тот же набор голосов даёт разный
    вердикт, и разбор инцидента упирается в «ну там как-то так получилось».
    """
    if n < 1:
        raise ValueError(f"voting needs at least one vote, got {n}")
    votes = [llm(prompt) for _ in range(n)]
    counts = {}
    for vote in votes:
        counts[vote] = counts.get(vote, 0) + 1
    winner = votes[0]
    # строгое > при обходе в порядке голосов оставляет победителем первого
    for vote in votes:
        if counts[vote] > counts[winner]:
            winner = vote
    return winner, counts


def parallel_sections(sections, llm, aggregate):
    """Parallelization (sectioning): свой промпт на каждый кусок.

    sections — последовательность пар (имя, промпт). Вернуть
    (результат aggregate, список пар (имя, выход)).

    parallel_sections((("a", "check a"), ("b", "check b")), llm, dict)
        ->  ({"a": ..., "b": ...}, [("a", ...), ("b", ...)])
    parallel_sections((), llm, dict)  ->  ({}, [])

    Отличие от voting: там один промпт много раз, здесь много промптов по
    одному разу. Смешивать их в одну функцию не стоит — у них разная цена
    ошибки: пропавший голос лишь смещает распределение, пропавшая секция
    выкидывает из ответа целый раздел.

    Порядок секций сохраняется: aggregate часто склеивает их в документ, и
    перестановка разделов там заметна.
    """
    outputs = [(name, llm(prompt)) for name, prompt in sections]
    return aggregate(outputs), outputs


def orchestrator_workers(task, workers, synthesize):
    """Orchestrator-workers: оркестратор выбирает исполнителей и сводит ответы.

    workers — последовательность словарей {"name", "handles", "run"}, где
    handles(task) -> bool, а run(task) -> строка.

    Вернуть (результат synthesize, список пар (имя, выход)).

    orchestrator_workers("review this python change", workers, len)
        ->  (сколько воркеров сработало, их выходы по порядку объявления)

    Ключевое отличие от агента: набор воркеров конечен и известен, цикла нет.
    Оркестратор решает КОГО позвать, но не решает, сколько раз повторить —
    поэтому стоимость запроса ограничена сверху числом воркеров.

    Ни один воркер не подошёл — это не ошибка: synthesize получает пустой
    список и сам решает, что ответить. Падать здесь нельзя, иначе каждый
    новый тип задачи будет ронять роутер.
    """
    outputs = []
    for worker in workers:
        if worker["handles"](task):
            outputs.append((worker["name"], worker["run"](task)))
    return synthesize(outputs), outputs


def evaluator_optimizer(task, propose, evaluate, max_iter=5):
    """Evaluator-optimizer: предложить, оценить, переписать. Вернуть (ответ, трасса).

    propose(task, feedback) -> кандидат; feedback на первой итерации None.
    evaluate(task, candidate) -> (ok, вердикт).
    Трасса — список тройек (кандидат, ok, вердикт).

    evaluator_optimizer("summarize ReAct", propose, evaluate)
        ->  (первый кандидат, прошедший evaluate, трасса до него включительно)
    evaluator_optimizer("...", propose, evaluate, max_iter=0)  ->  ValueError

    Вердикт неудачной итерации уходит в feedback следующей — иначе это не
    рефайнмент, а повторение одного и того же промпта с надеждой на другой
    результат.

    Цикл ОБРЫВАЕТСЯ на max_iter и возвращает последнего кандидата, даже если
    он не прошёл. Бесконечный цикл «пока оценщик не одобрит» — самый дорогой
    способ обнаружить, что оценщик слишком строгий.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    trace = []
    feedback = None
    candidate = None
    for _ in range(max_iter):
        candidate = propose(task, feedback)
        ok, verdict = evaluate(task, candidate)
        trace.append((candidate, ok, verdict))
        if ok:
            return candidate, trace
        feedback = verdict
    return candidate, trace


def pick_pattern(spec):
    """Выбрать паттерн под задачу. Вернуть (паттерн, причину).

    spec — словарь:
        steps_known                (обязателен) можно ли перечислить шаги
        workers_chosen_at_runtime  набор исполнителей решается на ходу
        has_evaluator              есть чем машинно проверить ответ
        categories                 сколько категорически разных входов
        parallel_units             сколько независимых подзадач

    pick_pattern({"steps_known": False})           ->  ("agent", ...)
    pick_pattern({"steps_known": True})            ->  ("prompt-chaining", ...)
    pick_pattern({"steps_known": True, "categories": 3})  ->  ("routing", ...)

    Правила проверяются в фиксированном порядке, от самого ограничивающего к
    самому общему: agent -> orchestrator-workers -> evaluator-optimizer ->
    routing -> parallelization -> prompt-chaining. Порядок и есть содержание
    функции: неперечислимые шаги нельзя выразить предопределённым графом,
    поэтому этот вопрос задаётся первым и отменяет все остальные.

    Отсутствие steps_known — KeyError. Это единственный вопрос, на который
    нельзя ответить значением по умолчанию: угадав его, функция вернёт
    красивый паттерн для задачи, которая ему не поддаётся.

    Заканчиваем на prompt-chaining, а не на «фреймворк»: линейная цепочка —
    самый дешёвый рабочий вариант, и по умолчанию выбирают её.
    """
    if "steps_known" not in spec:
        raise KeyError("steps_known")
    if not spec["steps_known"]:
        return "agent", "steps cannot be enumerated: the model must own the graph"
    if spec.get("workers_chosen_at_runtime", False):
        return "orchestrator-workers", "the set of specialists is decided per task"
    if spec.get("has_evaluator", False):
        return "evaluator-optimizer", "quality can be judged, so iterate until it passes"
    if spec.get("categories", 1) >= 2:
        return "routing", "categorically different inputs need different handling"
    if spec.get("parallel_units", 1) >= 2:
        return "parallelization", "independent units, so fan out and aggregate"
    return "prompt-chaining", "a linear decomposition is the cheapest thing that works"

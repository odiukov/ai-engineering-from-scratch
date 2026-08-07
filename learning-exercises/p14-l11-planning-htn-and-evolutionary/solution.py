"""
Планирование: HTN и эволюционный поиск — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Два разных инструмента для двух разных задач. HTN даёт план, корректный по
построению: символьный слой проверяет предусловия, а LLM (в ChatHTN) только
предлагает варианты декомпозиции и никогда не правит план напрямую.
AlphaEvolve, наоборот, ничего не доказывает — он перебирает мутации и
отбирает их детерминированным evaluator-ом.

Соответствие настоящим системам:

    applicable      <-  проверка preconditions оператора (pyhop, SHOP3)
    apply_operator  <-  применение effects: add / remove
    decompose       <-  выбор method из библиотеки методов
    plan            <-  ChatHTN: символьный поиск + LLM fallback + кэш методов
    execute_plan    <-  прогон плана в среде с проверкой на каждом шаге
    fitness_linear  <-  programmatic evaluator AlphaEvolve
    mutate          <-  предложение мутации (у DeepMind — ансамбль LLM)
    evolve          <-  цикл «мутировать, оценить, оставить лучших»

Ни LLM, ни сети: «LLM» — обычная функция (task, state) -> подзадачи, а всё
случайное берёт rng параметром, поэтому эволюция воспроизводима.

Домен выглядит так:

    {"operators": {"open_editor": {"pre": ("logged_in",),
                                   "add": ("editor_open",), "remove": ()}},
     "methods": {"ship_change": ({"name": "m1", "pre": ("logged_in",),
                                  "subtasks": ("open_editor", "write_tests")},)}}

Состояние — множество фактов-строк.
"""


def applicable(operator, state):
    """Выполнены ли предусловия оператора в текущем состоянии.

    op = {"pre": ("logged_in",), "add": ("editor_open",), "remove": ()}
    applicable(op, {"logged_in"})  ->  True
    applicable(op, set())          ->  False
    applicable({"pre": (), "add": ("x",), "remove": ()}, set())  ->  True

    Оператор без предусловий применим всегда: пустое «все» истинно. Это не
    краевой случай, а норма — так выглядит первый шаг любого плана.
    """
    facts = set(state)
    return all(p in facts for p in operator["pre"])


def apply_operator(operator, state):
    """Применить оператор к состоянию. Вернуть НОВОЕ состояние (frozenset).

    op = {"pre": ("logged_in",), "add": ("editor_open",), "remove": ("idle",)}
    apply_operator(op, {"logged_in", "idle"})  ->  frozenset({"logged_in", "editor_open"})
    apply_operator(op, set())                  ->  ValueError

    Порядок эффектов важен: сначала remove, потом add. Оператор, который
    удаляет и добавляет один и тот же факт (обновление на месте), при обратном
    порядке потерял бы его.

    Неприменимый оператор — ValueError, а не тихий пропуск. Ровно на этом
    держится заявление HTN о корректности плана: шаг, чьи предусловия не
    выполнены, не должен исполниться ни при каких обстоятельствах.
    """
    if not applicable(operator, state):
        raise ValueError(f"preconditions not met: {operator['pre']}")
    facts = set(state)
    for fact in operator["remove"]:
        facts.discard(fact)
    for fact in operator["add"]:
        facts.add(fact)
    return frozenset(facts)


def decompose(methods, task, state):
    """Первый применимый метод для задачи. Кортеж подзадач или None.

    methods = {"ship_change": ({"name": "m1", "pre": ("logged_in",),
                                "subtasks": ("open_editor", "run_tests")},)}
    decompose(methods, "ship_change", {"logged_in"})  ->  ("open_editor", "run_tests")
    decompose(methods, "ship_change", set())          ->  None
    decompose(methods, "unknown_task", {"logged_in"}) ->  None

    None на невыполненных предусловиях — центральное свойство HTN. Метод
    описывает, как делать задачу В ПОДХОДЯЩЕЙ обстановке; раскрыть его в
    неподходящей значит выдать план, который развалится на исполнении.

    Методы перебираются в порядке объявления: он и есть приоритет. Первый
    подходящий побеждает, остальные даже не смотрим.
    """
    facts = set(state)
    for method in methods.get(task, ()):
        if all(p in facts for p in method["pre"]):
            return tuple(method["subtasks"])
    return None


def plan(domain, task, state, llm=None, cache=None, max_depth=12):
    """Построить план: список примитивных операторов или None.

    ChatHTN целиком:
      1. задача — оператор? Применим — план из одного шага, нет — None.
      2. есть подходящий метод? Раскрываем его подзадачи.
      3. метода нет, но задача есть в cache? Берём оттуда, LLM не трогаем.
      4. метода нет — спрашиваем llm(task, state). Ответ ПРОВЕРЯЕМ по схеме:
         каждая подзадача обязана быть известным оператором или задачей.
         Не прошло — None. Прошло — кладём в cache и раскрываем.

    plan(domain, "open_editor", {"logged_in"})  ->  ["open_editor"]
    plan(domain, "ship_change", set())          ->  None
    plan(domain, "unknown", {"logged_in"}, llm=lambda t, s: ("fly_to_mars",))
        ->  None   (fly_to_mars нет в схеме — предложение отвергнуто)

    Почему план получается корректным даже с LLM в цикле: модель не
    редактирует план, она предлагает КАНДИДАТА декомпозиции. Символьный слой
    всё равно проверит каждый шаг по preconditions. Уберёшь проверку из
    пункта 4 — и заявление о корректности рассыпается.

    Состояние продвигается между подзадачами: вторая подзадача планируется
    уже в состоянии ПОСЛЕ первой. Иначе план вида «открыть редактор, писать
    тесты» никогда не соберётся — тесты требуют открытого редактора.

    cache заполняется НА МЕСТЕ и переживает вызовы: это online method
    learning, ради него всё и затевалось — повторный запрос к модели за той
    же декомпозицией стоит денег и времени.

    max_depth обрывает рекурсию: метод, раскрывающийся сам в себя, — обычная
    ошибка в библиотеке методов, и ловить её переполнением стека не стоит.
    """
    operators = domain["operators"]
    methods = domain["methods"]

    def expand(subtasks, current, depth):
        steps = []
        facts = frozenset(current)
        for subtask in subtasks:
            sub_plan = walk(subtask, facts, depth + 1)
            if sub_plan is None:
                return None, None
            for step in sub_plan:
                op = operators[step]
                if not applicable(op, facts):
                    return None, None
                facts = apply_operator(op, facts)
                steps.append(step)
        return steps, facts

    def walk(current_task, facts, depth):
        if depth > max_depth:
            return None
        if current_task in operators:
            return [current_task] if applicable(operators[current_task], facts) else None
        subtasks = decompose(methods, current_task, facts)
        if subtasks is None and cache is not None and current_task in cache:
            subtasks = cache[current_task]
        if subtasks is None and llm is not None:
            suggested = llm(current_task, frozenset(facts))
            if suggested is None:
                return None
            # схема — единственный фильтр между галлюцинацией и планом
            if not all(s in operators or s in methods for s in suggested):
                return None
            subtasks = tuple(suggested)
            if cache is not None:
                cache[current_task] = subtasks
        if subtasks is None:
            return None
        steps, _ = expand(subtasks, facts, depth)
        return steps

    return walk(task, frozenset(state), 0)


def execute_plan(domain, steps, state):
    """Исполнить план шаг за шагом. Вернуть итоговое состояние (frozenset).

    execute_plan(domain, ["open_editor"], {"logged_in"})
        ->  frozenset({"logged_in", "editor_open"})
    execute_plan(domain, [], {"logged_in"})     ->  frozenset({"logged_in"})
    execute_plan(domain, ["run_tests"], set())  ->  ValueError
    execute_plan(domain, ["fly"], set())        ->  KeyError

    Планировщик уже проверил предусловия — зачем проверять снова? Затем, что
    план и исполнение разнесены во времени: между ними среда могла измениться,
    план мог прийти из кэша, из LLM или из чужой системы. Проверка на
    исполнении — последняя граница, и стоит она один проход по множеству.

    Два разных отказа: KeyError — шага нет в схеме (план невалиден),
    ValueError — шаг есть, но не применим здесь и сейчас (среда не та).
    """
    facts = frozenset(state)
    for step in steps:
        if step not in domain["operators"]:
            raise KeyError(step)
        facts = apply_operator(domain["operators"][step], facts)
    return facts


def fitness_linear(individual, samples):
    """Evaluator: сумма квадратов ошибок прямой a*x + b на выборке.

    Меньше — лучше, ноль — идеальное попадание.

    fitness_linear((3, 7), ((0, 7), (1, 10)))  ->  0.0
    fitness_linear((0, 0), ((0, 7), (1, 10)))  ->  149.0

    Это и есть жёсткое условие AlphaEvolve: fitness обязана быть машинно
    проверяемой и детерминированной. «Спросить у модели, стало ли лучше» —
    не fitness: эволюция по шумному оценщику не сходится, она блуждает.
    """
    a, b = individual
    return float(sum((a * x + b - y) ** 2 for x, y in samples))


def mutate(individual, rng, step=2):
    """Сдвинуть каждую координату на случайную величину из [-step, step].

    rng — экземпляр random.Random. Глобальный random использовать нельзя:
    прогон должен воспроизводиться по сиду, иначе «эволюция сошлась» —
    непроверяемое утверждение.

    mutate((3, 7), random.Random(0), step=0)  ->  (3, 7)
    mutate((3, 7), random.Random(0))          ->  что-то в [1,5] x [5,9]

    step=0 — законный вырожденный случай: мутация без изменений. Он полезен,
    чтобы отделить вклад мутаций от вклада отбора.
    """
    return tuple(value + rng.randint(-step, step) for value in individual)


def evolve(population, fitness, mutate_fn, rng, generations=10, survivors=3,
           offspring=3):
    """Эволюционный поиск. Вернуть (лучшая особь, история лучших значений).

    Одно поколение: оценить всех, оставить survivors лучших, от каждого
    выжившего породить offspring мутантов, повторить.

    История — список длины generations + 1: значение fitness лучшей особи до
    первого поколения и после каждого.

    best, history = evolve([(0, 0)], fitness_linear_на_выборке, mutate,
                           random.Random(0), generations=30)
    history[0] >= history[-1]   ->  True всегда

    Главное свойство — элитизм: выжившие переходят в следующее поколение как
    есть, поэтому лучший результат НЕ МОЖЕТ ухудшиться. Если выкинуть
    родителей и оставить только детей, история начнёт скакать, и «поиск
    сошёлся» превратится в «повезло на последнем поколении».

    Ничьи по fitness разруливаются сравнением самих особей — без этого
    порядок зависел бы от порядка списка, и один и тот же сид давал бы разные
    ответы.
    """
    ranked = sorted(((fitness(ind), ind) for ind in population),
                    key=lambda pair: (pair[0], pair[1]))
    history = [ranked[0][0]]
    for _ in range(generations):
        parents = [ind for _, ind in ranked[:survivors]]
        children = [mutate_fn(parent, rng)
                    for parent in parents for _ in range(offspring)]
        ranked = sorted(((fitness(ind), ind) for ind in parents + children),
                        key=lambda pair: (pair[0], pair[1]))
        history.append(ranked[0][0])
    return ranked[0][1], history

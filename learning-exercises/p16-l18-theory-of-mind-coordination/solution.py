"""
Модель чужого сознания и эмерджентная координация — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

MAX_ORDER = 2

MARBLE = "marble"
BASKET_A = "basket_A"
BASKET_B = "basket_B"


def new_agent(name, order):
    """Агент с моделью чужих убеждений заданного порядка.

    new_agent("observer", 0)  ->  агент без модели других вообще
    new_agent("observer", 1)  ->  «Салли считает, что шарик в корзине A»
    new_agent("observer", 2)  ->  «Салли считает, что Энн считает, что...»

    Структура:
      beliefs — что агент считает сам;
      models  — что он приписывает другим; каждая модель устроена так же
                (beliefs + models), поэтому вложенность рекурсивна.

    Порядок вне 0..MAX_ORDER — ValueError. Нулевой порядок не заблуждение
    автора, а рабочая контрольная группа: без неё «эмерджентная координация»
    не отличается от координации, вписанной в промпт руками.
    """
    if not 0 <= order <= MAX_ORDER:
        raise ValueError(f"порядок модели бывает от 0 до {MAX_ORDER}, не {order}")
    return {"name": name, "order": order, "beliefs": {}, "models": {}}


def update_belief(agent, path, fact, value):
    """Записать убеждение по пути path. Пустой путь — собственное убеждение.

    update_belief(a, (), MARBLE, BASKET_B)              — я знаю, где шарик
    update_belief(a, ("sally",), MARBLE, BASKET_A)      — Салли думает иначе
    update_belief(a, ("sally", "anne"), MARBLE, BASKET_A)
        — я думаю, что Салли думает, что Энн думает, что шарик в A

    Длина пути и есть порядок модели. Путь длиннее agent["order"] — ValueError:
    агент первого порядка физически не хранит «он думает, что она думает».

    Промежуточные узлы создаются по дороге, поэтому порядок записей не важен.
    """
    if len(path) > agent["order"]:
        raise ValueError(
            f"агенту порядка {agent['order']} недоступна модель глубины {len(path)}"
        )
    node = agent
    for name in path:
        node = node["models"].setdefault(name, {"beliefs": {}, "models": {}})
    node["beliefs"][fact] = value
    return agent


def belief_of(agent, path, fact, default=None):
    """Прочитать убеждение по пути. Ничего не записано — вернуть default.

    belief_of(a, (), MARBLE)            ->  BASKET_B
    belief_of(a, ("sally",), MARBLE)    ->  BASKET_A
    belief_of(a, ("klaus",), MARBLE)    ->  None  (модели Клауса нет)

    Чтение несуществующей модели — не ошибка, а нормальная ситуация: агент
    просто ничего не знает про этого собеседника. Ошибкой была бы ПОПЫТКА
    ЗАПИСИ на недоступную глубину, её ловит update_belief.
    """
    node = agent
    for name in path:
        node = node["models"].get(name)
        if node is None:
            return default
    return node["beliefs"].get(fact, default)


def observe(agent, fact, value, witnesses=()):
    """Агент видит факт своими глазами. Свидетели видели то же самое.

    observe(a, MARBLE, BASKET_A, witnesses=("sally", "anne"))

    Что происходит по порядкам модели:
      0 — обновляется только собственное убеждение, свидетели игнорируются
          (у агента просто нет места, куда их записать);
      1 — плюс каждому свидетелю приписывается то же убеждение;
      2 — плюс каждому свидетелю приписывается знание о том, что и агент это
          видел: путь (свидетель, имя агента).

    Ключ ко всей задаче о ложном убеждении: тот, кого нет в witnesses, свою
    модель НЕ обновляет. Его картина мира остаётся вчерашней.
    """
    update_belief(agent, (), fact, value)
    if agent["order"] >= 1:
        for witness in witnesses:
            update_belief(agent, (witness,), fact, value)
    if agent["order"] >= 2:
        for witness in witnesses:
            update_belief(agent, (witness, agent["name"]), fact, value)
    return agent


def predict_search(agent, other, fact):
    """Где other будет искать fact, по мнению агента.

    predict_search(агент нулевого порядка, "sally", MARBLE)  ->  BASKET_B
    predict_search(агент первого порядка, "sally", MARBLE)   ->  BASKET_A

    Нулевой порядок отвечает единственным доступным ему способом — своим
    собственным убеждением: «я знаю, где шарик, значит и она знает». Это
    ровно ошибка трёхлетнего ребёнка в тесте Салли-Энн.

    Первый порядок и выше читает приписанное убеждение. Если про этот факт у
    другого ничего не записано, честнее всего откатиться к собственному: не
    знаю, чем он отличается от меня, — считаю, что не отличается.
    """
    own = belief_of(agent, (), fact)
    if agent["order"] == 0:
        return own
    return belief_of(agent, (other,), fact, default=own)


def sally_anne(order):
    """Классический тест на ложное убеждение. Куда, по мнению агента, пойдёт Салли.

    sally_anne(0)  ->  BASKET_B   (провал: агент приписывает Салли своё знание)
    sally_anne(1)  ->  BASKET_A   (успех: у Салли своя, устаревшая картина)

    Сценарий Baron-Cohen, Leslie, Frith (1985):
      1. Салли кладёт шарик в корзину A. Видят все — и Энн, и наблюдатель.
      2. Салли выходит. Энн перекладывает шарик в корзину B. Салли этого
         не видит, и в списке свидетелей её больше нет.
      3. Салли возвращается. Вопрос: куда она пойдёт?

    Правильный ответ — корзина A, потому что вопрос не про шарик, а про
    убеждение Салли. Агент без модели чужих убеждений отвечает про шарик.
    """
    observer = new_agent("observer", order)
    observe(observer, MARBLE, BASKET_A, witnesses=("sally", "anne"))
    observe(observer, MARBLE, BASKET_B, witnesses=("anne",))
    return predict_search(observer, "sally", MARBLE)


def choose_box(agent, boxes, others=()):
    """Какой ящик берёт агент, зная, что others выбирают раньше него.

    choose_box(агент, ["b0", "b1"], others=())          ->  свой лучший ящик
    choose_box(агент первого порядка, boxes, ("a0",))   ->  лучший из свободных

    Нулевой порядок берёт свой максимум и не смотрит по сторонам — отсюда и
    дублирование усилий: агенты на одной модели видят мир одинаково и лезут
    в один и тот же ящик.

    Первый порядок и выше сначала ПРЕДСКАЗЫВАЕТ выбор каждого из others по
    приписанным ему ценностям (в том же порядке приоритета и по тому же
    правилу), вычёркивает предсказанное и берёт лучшее из оставшегося. Это и
    есть goal-directed complementarity: действия расходятся по подзадачам.

    Если предсказание съело все ящики, выбирать всё равно надо — берём лучший
    из всех. Пустой список ящиков — ValueError.
    """
    if not boxes:
        raise ValueError("выбирать не из чего")

    def own(box):
        return belief_of(agent, (), box, default=0.0)

    def best(pool, value):
        # ничья по ценности разрешается именем ящика: соглашение должно быть
        # одинаковым у всех, иначе предсказание чужого выбора разъедется
        return min(pool, key=lambda b: (-value(b), b))

    if agent["order"] == 0 or not others:
        return best(boxes, own)
    free = list(boxes)
    for other in others:
        if not free:
            break

        def attributed(box, who=other):
            # чужие ценности глазами агента; не приписано — считаем как свои
            return belief_of(agent, (who,), box, default=own(box))

        free.remove(best(free, attributed))
    return best(free or boxes, own)


def simulate_collection(n_agents, n_boxes, order, rng, trials=200, noise=0.15):
    """Совместный сбор: n агентов молча разбирают n ящиков разной ценности.

    Возвращает {"duplication_rate": ..., "completion_rate": ...}.

    simulate_collection(3, 3, 0, random.Random(0), trials=300)
      ->  {"duplication_rate": 0.53,  "completion_rate": 0.03}
    simulate_collection(3, 3, 1, random.Random(0), trials=300)
      ->  {"duplication_rate": 0.12,  "completion_rate": 0.64}

    Каждое испытание: ящикам назначаются истинные ценности, каждый агент
    видит их со своим шумом +-noise и приписывает СВОЮ картину остальным —
    чужого шума он не знает и знать не может. Приоритет выбора — по индексу,
    это общее соглашение, известное всем.

    duplication_rate — средняя доля агентов, полезших в уже занятый ящик.
    completion_rate — доля испытаний, где выборы разошлись полностью.

    Дельта между order=0 и order=1 и есть измеримый эффект координации. Без
    контрольной группы нулевого порядка любые разговоры про «эмерджентную
    координацию» остаются рассказом.
    """
    boxes = [f"box{i}" for i in range(n_boxes)]
    names = [f"a{i}" for i in range(n_agents)]
    duplicated = 0.0
    complete = 0
    for _ in range(trials):
        true_value = {b: rng.random() for b in boxes}
        agents = []
        for name in names:
            agent = new_agent(name, order)
            for b in boxes:
                perceived = true_value[b] + rng.uniform(-noise, noise)
                observe(agent, b, perceived, witnesses=[n for n in names if n != name])
            agents.append(agent)
        picks = [
            choose_box(agents[i], boxes, others=names[:i]) for i in range(n_agents)
        ]
        duplicated += (len(picks) - len(set(picks))) / len(picks)
        complete += len(set(picks)) == len(picks)
    return {
        "duplication_rate": duplicated / trials,
        "completion_rate": complete / trials,
    }

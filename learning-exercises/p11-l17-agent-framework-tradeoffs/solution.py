"""
Выбор агентного фреймворка — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок «Learn», а не «Build»: сравниваются LangGraph, CrewAI, AutoGen и Agno.
Ставить их сюда нечего и незачем — вместо этого мы превращаем таблицу
решений из урока в исполняемый код. Получается ровно то, что урок просит в
третьем упражнении: pick_framework.py, который по описанию задачи выдаёт
рекомендацию и одно предложение обоснования.

Никакой сети, никакого LLM: задача описывается словарём флагов, а вся
«экспертиза» лежит в константе FRAMEWORKS — её видно, её можно спорить и
править. Именно этим она отличается от мнения в чате.
"""

# Паспорта фреймворков. Числа и категории — из таблиц урока.
#   abstraction  — что рисуешь на доске, когда объясняешь архитектуру
#   durable_state — что переживёт перезапуск процесса
#   branching    — кто решает, какой шаг следующий
#   router_tokens_per_turn — сколько ВХОДНЫХ токенов на ход тратит
#                  LLM-роутер фреймворка; 0 у явной маршрутизации
FRAMEWORKS = {
    "langgraph": {
        "abstraction": "graph",
        "durable_state": "checkpointer",
        "branching": "explicit",
        "parallel_fanout": True,
        "interrupts": True,
        "router_tokens_per_turn": 0,
        "setup_lines": 40,
    },
    "crewai": {
        "abstraction": "roles",
        "durable_state": "none",
        "branching": "manager",
        "parallel_fanout": False,
        "interrupts": False,
        "router_tokens_per_turn": 300,
        "setup_lines": 25,
    },
    "autogen": {
        "abstraction": "chat",
        "durable_state": "transcript",
        "branching": "chat",
        "parallel_fanout": False,
        "interrupts": False,
        "router_tokens_per_turn": 350,
        "setup_lines": 20,
    },
    "agno": {
        "abstraction": "agent",
        "durable_state": "session",
        "branching": "tools",
        "parallel_fanout": False,
        "interrupts": False,
        "router_tokens_per_turn": 0,
        "setup_lines": 10,
    },
    "plain-python": {
        "abstraction": "none",
        "durable_state": "none",
        "branching": "explicit",
        "parallel_fanout": False,
        "interrupts": False,
        "router_tokens_per_turn": 0,
        "setup_lines": 30,
    },
}

# Описание задачи: только эти ключи, только такие умолчания.
DEFAULT_PROBLEM = {
    "llm_calls": 1,
    "has_typed_state": False,
    "has_roles": False,
    "has_dialogue": False,
    "has_parallel_fanout": False,
    "needs_resume": False,
    "needs_human_approval": False,
}

# Какая абстракция фреймворка подходит какой форме задачи.
SHAPE_TO_ABSTRACTION = {
    "trivial": "none",
    "graph": "graph",
    "dialogue": "chat",
    "roles": "roles",
    "agent": "agent",
}

# Сколько выходных токенов стоит одно решение LLM-роутера («говорит следующим Боб»).
ROUTER_OUTPUT_TOKENS = 20


class ProblemError(Exception):
    """Описание задачи не по форме: лишний ключ или не тот тип.

    Свой класс, а не RuntimeError, специально: NotImplementedError — тоже
    RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл бы зелёным на
    пустой заготовке, ничего не проверив.
    """


def normalize_problem(problem):
    """Дополнить описание задачи умолчаниями и проверить его форму.

    normalize_problem({})                    ->  копия DEFAULT_PROBLEM
    normalize_problem({"needs_resume": True})["needs_resume"]  ->  True
    normalize_problem({"llm_calls": 5})["has_roles"]           ->  False

    Лишний ключ — ProblemError, а не молчаливое игнорирование. Опечатка
    "needs_resume" -> "need_resume" иначе просто выключила бы требование, и
    рекомендация вышла бы уверенной и неправильной.

    Флаги обязаны быть bool, llm_calls — целым не меньше единицы. Ловушка:
    в Python bool — подкласс int, поэтому llm_calls=True без отдельной
    проверки прошло бы как «один вызов».

    Входной словарь не меняется: вызывающий код вправе передать свой.
    """
    unknown = set(problem) - set(DEFAULT_PROBLEM)
    if unknown:
        raise ProblemError(f"неизвестные ключи: {', '.join(sorted(unknown))}")
    merged = dict(DEFAULT_PROBLEM)
    merged.update(problem)
    calls = merged["llm_calls"]
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
        raise ProblemError(f"llm_calls должно быть целым >= 1, получено {calls!r}")
    for key, value in merged.items():
        if key != "llm_calls" and not isinstance(value, bool):
            raise ProblemError(f"{key} должно быть bool, получено {value!r}")
    return merged


def shape_of(problem):
    """Форма задачи одним словом: с чего начинается выбор фреймворка.

    shape_of({})                                  ->  "trivial"
    shape_of({"llm_calls": 6, "has_typed_state": True})  ->  "graph"
    shape_of({"llm_calls": 6, "has_dialogue": True})     ->  "dialogue"
    shape_of({"llm_calls": 6, "has_roles": True})        ->  "roles"
    shape_of({"llm_calls": 6})                           ->  "agent"

    Порядок проверок — не случайный:

    1. Две-три LLM-ки без состояния, без параллели и без согласований — это
       "trivial". Никакой фреймворк не дешевле отсутствия фреймворка.
    2. Типизированное состояние, резюме после перезапуска, согласование
       человеком или параллельный веер — это "graph". Такие требования
       нельзя дописать сверху, они определяют рантайм.
    3. Дальше диалог, потом роли, в остатке — одиночный агент с тулами.
    """
    p = normalize_problem(problem)
    hard = (
        p["has_typed_state"]
        or p["needs_resume"]
        or p["needs_human_approval"]
        or p["has_parallel_fanout"]
    )
    if p["llm_calls"] <= 2 and not hard:
        return "trivial"
    if hard:
        return "graph"
    if p["has_dialogue"]:
        return "dialogue"
    if p["has_roles"]:
        return "roles"
    return "agent"


def hard_filter(problem):
    """Фреймворки, которые в принципе способны выполнить жёсткие требования.

    hard_filter({"needs_resume": True})        ->  ["langgraph"]
    hard_filter({"has_parallel_fanout": True}) ->  ["langgraph"]
    hard_filter({})                            ->  все пять

    Жёстких требований ровно три, и все три — про рантайм, а не про вкус:
      needs_resume         — нужен настоящий чекпоинтер, а не журнал сессии;
      needs_human_approval — нужны interrupt на границе узла;
      has_parallel_fanout  — нужен первоклассный параллельный диспатч.

    Порядок ответа — как в FRAMEWORKS, чтобы результат был воспроизводим.

    Смысл шага: сначала отсечь тех, с кем придётся драться, и только потом
    сравнивать удобство. Фреймворк, чью модель состояния приходится
    обходить, дешевле не станет никогда.
    """
    p = normalize_problem(problem)
    out = []
    for name, spec in FRAMEWORKS.items():
        if p["needs_resume"] and spec["durable_state"] != "checkpointer":
            continue
        if p["needs_human_approval"] and not spec["interrupts"]:
            continue
        if p["has_parallel_fanout"] and not spec["parallel_fanout"]:
            continue
        out.append(name)
    return out


def score(framework, problem):
    """Насколько фреймворк подходит задаче. Больше — лучше.

    score("langgraph", {"llm_calls": 6, "has_typed_state": True})  ->  2.6
    score("agno", {"llm_calls": 6, "has_typed_state": True})       ->  -0.1

    Из чего складывается:
      +3   абстракция фреймворка совпала с формой задачи;
      +1   за каждое жёсткое требование, которое фреймворк закрывает;
      -setup_lines/100 — мягкий штраф за объём обвязки, он же тайбрейкер.

    Штраф маленький нарочно: он решает только споры равных. Если он вдруг
    перевешивает совпадение абстракции — значит, в задаче нет ничего, кроме
    желания поменьше писать, и ответ «плоский Python» честнее.
    """
    if framework not in FRAMEWORKS:
        raise ValueError(f"неизвестный фреймворк: {framework!r}")
    p = normalize_problem(problem)
    spec = FRAMEWORKS[framework]
    total = 3.0 if spec["abstraction"] == SHAPE_TO_ABSTRACTION[shape_of(p)] else 0.0
    if p["needs_resume"] and spec["durable_state"] == "checkpointer":
        total += 1.0
    if p["needs_human_approval"] and spec["interrupts"]:
        total += 1.0
    if p["has_parallel_fanout"] and spec["parallel_fanout"]:
        total += 1.0
    return round(total - spec["setup_lines"] / 100, 4)


def pick_framework(problem):
    """Рекомендация с обоснованием в одну строку.

    Возвращает {"framework", "shape", "reason", "runners_up"}.

    pick_framework({})["framework"]                          ->  "plain-python"
    pick_framework({"needs_resume": True})["framework"]      ->  "langgraph"
    pick_framework({"llm_calls": 6, "has_dialogue": True})["framework"]  ->  "autogen"

    Сначала жёсткий фильтр, потом скор среди выживших. Ничьи разрешаются
    порядком FRAMEWORKS, поэтому ответ детерминирован: одинаковый вход даёт
    одинаковый ответ, а порядок ключей во входном словаре ни на что не влияет.

    runners_up — остальные кандидаты по убыванию скора. Рекомендация без
    альтернатив — это не решение, а лозунг.
    """
    p = normalize_problem(problem)
    candidates = hard_filter(p)
    ranked = sorted(candidates, key=lambda name: -score(name, p))
    best = ranked[0]
    shape = shape_of(p)
    return {
        "framework": best,
        "shape": shape,
        "reason": (
            f"форма задачи — {shape}, её абстракция {SHAPE_TO_ABSTRACTION[shape]}; "
            f"{best} закрывает жёсткие требования и даёт минимум обвязки среди подходящих"
        ),
        "runners_up": ranked[1:],
    }


def routing_cost_per_run(framework, turns, input_price, output_price):
    """Во что обходится маршрутизация за один прогон, в долларах.

    routing_cost_per_run("crewai", 10, 5.0, 15.0)     ->  0.018
    routing_cost_per_run("autogen", 10, 5.0, 15.0)    ->  0.0205
    routing_cost_per_run("langgraph", 10, 5.0, 15.0)  ->  0.0

    Считаются ТОЛЬКО токены на решение «кто ходит следующим»: полезная
    работа агента здесь ни при чём, она одинакова у всех.

    Явная маршрутизация (ребро графа, питоновская функция) стоит ноль
    токенов. Менеджер CrewAI и GroupChatManager AutoGen на каждом ходу
    думают LLM-кой — и это и есть та самая разница в накладных расходах.
    """
    if framework not in FRAMEWORKS:
        raise ValueError(f"неизвестный фреймворк: {framework!r}")
    in_tokens = FRAMEWORKS[framework]["router_tokens_per_turn"]
    if in_tokens == 0:
        return 0.0
    per_turn = in_tokens / 1_000_000 * input_price
    per_turn += ROUTER_OUTPUT_TOKENS / 1_000_000 * output_price
    return round(turns * per_turn, 10)


def compare_run_cost(problem, turns, input_price, output_price):
    """Подходящие фреймворки со стоимостью маршрутизации, от дешёвых к дорогим.

    Возвращает список пар (имя, цена за прогон). В список попадают только
    те, кто прошёл hard_filter: сравнивать по цене то, что не выполнит
    задачу, бессмысленно.

    compare_run_cost({"needs_resume": True}, 10, 5.0, 15.0)
        ->  [("langgraph", 0.0)]

    Ничьи упорядочены по имени, чтобы результат не зависел от прогона.

    Это ответ на вопрос из чек-листа урока: «если агент бегает тысячи раз в
    день, предпочитай явную маршрутизацию». Разница в 0.02 доллара за
    прогон на десяти тысячах прогонов в день — это 200 долларов в день ни
    за что.
    """
    p = normalize_problem(problem)
    priced = [
        (name, routing_cost_per_run(name, turns, input_price, output_price))
        for name in hard_filter(p)
    ]
    return sorted(priced, key=lambda pair: (pair[1], pair[0]))

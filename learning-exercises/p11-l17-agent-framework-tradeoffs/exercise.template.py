"""
Выбор агентного фреймворка

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p11-l17-agent-framework-tradeoffs
Разбор:  /check-code p11-l17-agent-framework-tradeoffs
"""

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
DEFAULT_PROBLEM = {
    "llm_calls": 1,
    "has_typed_state": False,
    "has_roles": False,
    "has_dialogue": False,
    "has_parallel_fanout": False,
    "needs_resume": False,
    "needs_human_approval": False,
}
SHAPE_TO_ABSTRACTION = {
    "trivial": "none",
    "graph": "graph",
    "dialogue": "chat",
    "roles": "roles",
    "agent": "agent",
}
ROUTER_OUTPUT_TOKENS = 20


class ProblemError(Exception):
    """Описание задачи не по форме: лишний ключ или не тот тип.

    Свой класс, а не RuntimeError, специально: NotImplementedError — тоже
    RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл бы зелёным на
    пустой заготовке, ничего не проверив.
    """
    pass


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError

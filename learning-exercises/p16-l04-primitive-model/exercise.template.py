"""
Примитивная модель мультиагента

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l04-primitive-model
Разбор:  /check-code p16-l04-primitive-model
"""

DONE = "done"


def make_agent(name, system_prompt, tools, policy):
    """Примитив №1 — агент: system prompt плюс список инструментов. Без памяти.

    policy(pool) -> dict вида {"content": ..., "handoff": ...}, где handoff
    необязателен.

    a = make_agent("researcher", "Gather facts.", ["search"],
                   lambda pool: {"content": "note", "handoff": "writer"})
    a["name"]   ->  "researcher"
    a["tools"]  ->  ["search"]

    Состояния у агента нет намеренно: всё, что похоже на память, на самом
    деле лежит в общем состоянии (примитив №3) или в handoff (примитив №2).
    """
    raise NotImplementedError


def agents_are_interchangeable(a, b):
    """Взаимозаменяемы ли два агента: совпадают ли prompt и набор инструментов.

    Имя не считается — это ярлык для оркестратора, а не часть агента.
    Порядок инструментов тоже не считается.

    agents_are_interchangeable(make_agent("a", "P", ["t"], f),
                               make_agent("b", "P", ["t"], g))   ->  True
    agents_are_interchangeable(make_agent("a", "P", ["t"], f),
                               make_agent("a", "Q", ["t"], f))   ->  False

    Это и есть «stateless insight» урока: агент — функция от (prompt,
    tools), поэтому двух одинаковых можно менять местами без последствий.
    """
    raise NotImplementedError


def post(pool, sender, content, handoff=None):
    """Примитив №3 — общее состояние: дописать сообщение в пул. Вернуть его.

    pool — обычный список, он изменяется на месте: пул один на всю систему,
    копии разошлись бы.

    pool = []
    post(pool, "researcher", "note", "writer")
    pool  ->  [{"from": "researcher", "content": "note", "handoff": "writer"}]

    handoff=None означает «никому конкретно»: сообщение видно всем, но
    никого не назначает следующим.
    """
    raise NotImplementedError


def project(pool, agent_name):
    """Проекция пула на одного агента: своё написанное плюс адресованное ему.

    Полный пул (full pool) прост и плохо масштабируется; проекция
    масштабируется, но требует схемы — ровно этот выбор в уроке.

    pool из трёх сообщений researcher->writer, writer->reviewer,
    reviewer->done:
        project(pool, "writer")  ->  два сообщения (адресованное и своё)
        project(pool, "reviewer")  ->  два сообщения

    Ловушка: собственные сообщения агента в проекцию входят. Иначе агент
    на втором шаге не вспомнит, что уже сказал.
    """
    raise NotImplementedError


def run_static(team, pool, order, max_steps=10):
    """Оркестратор №1 — статический: порядок задан заранее, handoff игнорируется.

    Это LangGraph с детерминированными рёбрами и CrewAI Process.Sequential.

    run_static(team, pool, ["researcher", "writer"])  ->  пул из двух
    сообщений, ровно в этом порядке.

    Заявленный агентом handoff тут ничего не решает: маршрут прибит в коде.
    Это плюс для аудита и воспроизводимости и минус для адаптивности.
    """
    raise NotImplementedError


def run_handoff(team, pool, start, max_steps=10):
    """Оркестратор №2 — handoff-driven: следующего называет текущий агент.

    Это паттерн OpenAI Swarm: инструмент возвращает следующего агента.

    Останов — любое из трёх:
      * handoff == DONE или None,
      * названного агента нет в команде,
      * исчерпан max_steps.

    run_handoff(team, pool, "researcher", max_steps=3)

    Ловушка: без max_steps два агента, передающие друг другу управление,
    крутятся вечно. LLM-маршрутизация ошибается, лимит шагов обязателен.
    """
    raise NotImplementedError


def round_robin_selector(pool, names):
    """Выбор следующего говорящего по кругу. Пустой пул -> None.

    round_robin_selector([{"from": "a", "content": "x", "handoff": None}],
                         ["a", "b", "c"])   ->  "b"
    round_robin_selector([], ["a", "b"])    ->  None

    Если последний говоривший вообще не из списка — тоже None: селектор не
    угадывает, он выбирает.
    """
    raise NotImplementedError


def run_selector(team, pool, start, selector, max_steps=10):
    """Оркестратор №3 — speaker selection: следующего выбирает отдельная функция.

    Это AutoGen GroupChat: selector(pool, names) читает пул и называет
    следующего. В бою selector сам может быть вызовом LLM.

    Останов: selector вернул None, назвал агента не из команды, или
    исчерпан max_steps.

    Разница с run_handoff ровно одна — КТО решает. Агенты и общее
    состояние те же самые.
    """
    raise NotImplementedError

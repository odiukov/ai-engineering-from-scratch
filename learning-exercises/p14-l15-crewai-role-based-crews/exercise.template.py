"""
Команды агентов по ролям — роли, задачи, процессы

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l15-crewai-role-based-crews
Разбор:  /check-code p14-l15-crewai-role-based-crews
"""

MAX_BACKSTORY_WORDS = 200


def make_agent(role, goal, backstory, tools=()):
    """Примитив Agent: role + goal + backstory + tools.

    Вернуть словарь с ключами role, goal, backstory, tools (tools — кортеж).

    make_agent("writer", "turn sources into a draft", "editorial voice")["role"]
        ->  'writer'

    ValueError, если:
      * role, goal или backstory пустые (backstory не украшение: он задаёт
        тон и то, когда агент останавливается);
      * в backstory больше MAX_BACKSTORY_WORDS слов;
      * инструмент не вызываемый или у него нет docstring.

    Про docstring: в CrewAI подпись функции становится схемой, а docstring —
    описанием, которое читает модель. Инструмент без docstring модель
    выбирает наугад.
    """
    raise NotImplementedError


def make_task(description, expected_output, agent, context=()):
    """Примитив Task: description + expected_output + agent + context.

    Вернуть словарь с этими же ключами (context — кортеж задач).

    ValueError, если description или expected_output пустые, если agent не
    похож на агента (нет ключа role) или если в context попало не задание.

    expected_output — это контракт задачи. Без него следующий агент
    разбирает то, что модель случайно выдала: crew отработал, аудит
    провалился. Это первая из трёх типовых поломок CrewAI.

    context перечисляет задачи ВЫШЕ по цепочке, чьи выходы надо подать на
    вход этой задаче.
    """
    raise NotImplementedError


def crew_prompt(agent, task, context_outputs=()):
    """Промпт одного шага: всё, что агент увидит перед ответом.

    Строки в фиксированном порядке:
      ROLE / GOAL / BACKSTORY / TASK / EXPECTED OUTPUT / CONTEXT n / TOOLS

    Строка TOOLS появляется только у агента с инструментами, строки
    CONTEXT — только если что-то передали.

    a = make_agent("writer", "draft it", "terse voice")
    t = make_task("write a draft", "3 paragraphs", a)
    crew_prompt(a, t).splitlines()[0]  ->  'ROLE: writer'

    Инструменты показываются как "name — первая строка docstring", по
    алфавиту: стабильный порядок нужен, чтобы промпт попадал в кэш.

    Именно здесь видно цену раздутых backstory: prompt-bloat — это не
    метафора, а длина вот этой строки, умноженная на число шагов.
    """
    raise NotImplementedError


def run_sequential(tasks, topic, run_agent):
    """Процесс Sequential: задачи в порядке объявления, выход N — вход N+1.

    run_agent — функция prompt -> output, детерминированная заглушка модели.

    Вернуть список словарей по одному на задачу, в порядке объявления:
      {"role": ..., "prompt": ..., "output": ...}

    Число вызовов модели равно числу задач — это самый дешёвый процесс.

    Если у задачи объявлен context, на вход идут выходы ИМЕННО этих задач;
    если нет — выход предыдущей задачи, а для первой задачи сам topic.

    ValueError на пустом списке задач и на задаче, чей context ссылается
    вперёд: задача из context ещё не выполнялась, подставлять нечего.
    """
    raise NotImplementedError


def run_hierarchical(manager_task, tasks, topic, run_agent, max_rounds=6):
    """Процесс Hierarchical: менеджер каждый раунд выбирает специалиста.

    manager_task — задача менеджера; её агент и есть manager_llm. В context
    менеджеру подаются три строки: ROSTER, DONE и LATEST.

    Ответ менеджера — имя роли или "done".

    Вернуть словарь:
      {"trace": [...], "llm_calls": int, "final": str,
       "done": [роли по порядку], "stop_reason": str}

    Шаг трассы — словарь {"role", "prompt", "pick", "output"}: у шага
    менеджера заполнен pick, у шага специалиста — output.

    llm_calls == 2 * len(done) + 1 для прогона, который менеджер закрыл сам:
    это и есть manager-LLM tax. Пять задач — шесть вызовов вместо пяти, и
    каждый вызов менеджера тащит весь список задач.

    Останов: "done" от менеджера, неизвестная роль, повтор уже сделанной
    роли (иначе петля не кончится) или исчерпанный max_rounds.

    ValueError, если задач нет или две задачи назначены на одну роль:
    вторая была бы недостижима, а менеджер об этом не узнает.
    """
    raise NotImplementedError


def run_flow(start, listeners, payload, max_steps=20):
    """Flow: событийный граф, который принадлежит коду, а не модели.

    start — функция payload -> (topic, output), аналог @start.
    listeners — словарь topic -> функция output -> (topic, output) | None,
    аналог @listen(topic).

    Вернуть трассу: список кортежей (step_name, topic, output). Первый
    элемент всегда ("start", ...).

    Петля кончается, когда на очередной topic нет слушателя или слушатель
    вернул None.

    ValueError, если шагов больше max_steps: значит в графе цикл. Flow, в
    отличие от Crew, обязан завершаться предсказуемо.

    Темы задаёт код, поэтому трасса воспроизводима и её можно
    диффать — ровно то, чего нет у свободного Crew и из-за чего документация
    CrewAI советует начинать с Flow.
    """
    raise NotImplementedError


def remember(memory, kind, value, key=None):
    """Записать факт в память. Вернуть тот же (изменённый) словарь memory.

    kind — "short_term", "long_term" или "entity".
    Для "entity" обязателен key: id сущности (клиент, тикет, аккаунт).

    m = remember({}, "long_term", "crew shipped the brief")
    m["long_term"]  ->  ['crew shipped the brief']

    ValueError на неизвестном kind и на entity без key.

    Разница между кучами не в хранилище, а в жизненном цикле: short_term
    стирается в конце прогона, long_term живёт между kickoff-ами, entity
    привязана к сущности и достаётся по ключу, а не по похожести.
    """
    raise NotImplementedError


def recall_context(memory, query, entity_id=None, k=2):
    """Contextual memory: собрать то, что нужно агенту именно сейчас.

    Вернуть словарь с тремя ключами:
      "short_term" — буфер текущего прогона целиком;
      "entity"     — факты по entity_id (пустой список, если id не задан
                     или неизвестен);
      "long_term"  — до k записей с наибольшим пересечением слов с query.

    Записи long_term без общих слов с query НЕ возвращаются: иначе с ростом
    базы выдача становится шумом, а это одна из главных жалоб на
    always-on память.

    При равном пересечении порядок — как записывали.

    recall_context({"long_term": ["brief about agents"]}, "brief")["long_term"]
        ->  ['brief about agents']
    """
    raise NotImplementedError

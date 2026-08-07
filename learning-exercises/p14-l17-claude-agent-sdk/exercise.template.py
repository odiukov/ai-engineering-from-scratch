"""
Харнесс как библиотека: подагенты и хранилище сессий

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l17-claude-agent-sdk
Разбор:  /check-code p14-l17-claude-agent-sdk
"""

HOOK_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreCompact",
    "Stop",
    "Notification",
)
BUILTIN_TOOL_NAMES = ("read_file", "write_file", "list_dir")


def stub_model(prompt):
    """Детерминированная заглушка модели: один и тот же промпт — один ответ.

    stub_model("hi")   ->  'answer[2:209]'
    stub_model("hi ")  ->  'answer[3:241]'
    stub_model("ih")   ->  'answer[2:209]'

    Заменяет вызов Messages API. Свойство, ради которого она такая: ответ
    зависит ТОЛЬКО от промпта, поэтому тест на петлю агента не мигает.
    Обрати внимание на третий пример: перестановка букв ответ не меняет —
    сумма кодов одна и та же. Это не баг заглушки, а напоминание, что
    подпись «по сумме» — не хеш.
    """
    raise NotImplementedError


def select_tools(registry, allowed):
    """Отобрать из реестра инструменты, разрешённые агенту (`allowed_tools`).

    select_tools({"a": f, "b": g}, ["a"])   ->  {"a": f}
    select_tools({"a": f, "b": g}, None)    ->  {"a": f, "b": g}
    select_tools({"a": f}, ["b"])           ->  KeyError

    `allowed=None` — «все инструменты», как в SDK по умолчанию.

    Две ловушки. Первая: имя, которого в реестре нет, — это ошибка
    конфигурации, а не «просто пропустим»; молчаливый пропуск потом выглядит
    как «инструмент есть, но модель его не зовёт». Вторая: возвращать нужно
    НОВЫЙ словарь. Если вернуть сам registry, подагент сможет дописать себе
    инструмент, которого ему не давали, и изоляция превратится в фикцию.
    """
    raise NotImplementedError


def run_hooks(hooks, event, payload):
    """Прогнать хуки события по порядку; первый deny останавливает цепочку.

    Хук — функция payload -> None (разрешить) либо
    {"decision": "deny", "reason": "..."}.

    run_hooks({}, "PreToolUse", {})                  ->  None
    run_hooks({"PreToolUse": [deny]}, "PreToolUse", {})
                                                     ->  {"reason": "..."}
    run_hooks({}, "PreToolCall", {})                 ->  ValueError

    Событие вне HOOK_EVENTS — ValueError: опечатка в имени иначе даёт хук,
    который просто никогда не срабатывает, и это ловится уже в проде.

    Важное свойство: после первого deny остальные хуки НЕ вызываются —
    решение принято, дальше только лишние побочные эффекты.
    """
    raise NotImplementedError


def call_tool(registry, hooks, name, args, journal):
    """Вызов инструмента через ворота хуков; всё записывается в journal.

    Возвращает dict:
      {"ok": True,  "tool": ..., "result": ...}          — отработал
      {"ok": True,  ..., "blocked": reason}              — отработал, но
                                                           PostToolUse
                                                           пометил результат
      {"ok": False, "tool": ..., "error": reason}        — не отработал

    call_tool({"echo": h}, {}, "echo", {"x": 1}, [])
        ->  {"ok": True, "tool": "echo", "result": ...}
    call_tool({}, {}, "write_file", {}, [])
        ->  {"ok": False, "tool": "write_file", "error": "tool_not_allowed"}

    Главная ловушка: PreToolUse — это ворота ДО вызова. Если он сказал deny,
    обработчик не должен быть вызван ни разу, иначе побочный эффект уже
    случился и «запрет» ничего не запретил. PostToolUse, наоборот, отменить
    ничего не может — он только помечает результат.

    Инструмент, которого нет в реестре, — не падение, а отказ: подагенту
    вполне штатно урезали список, и модель об этом узнаёт из ответа.
    """
    raise NotImplementedError


def session_subkeys(store, session_id):
    """Ключи подагентских сессий данной сессии — SDK-шный `list_subkeys`.

    Соглашение: сессия подагента лежит под ключом "<parent>/<имя>".

    session_subkeys({"s1": [], "s1/a": [], "s2": []}, "s1")  ->  ["s1/a"]
    session_subkeys({"s1": []}, "s1")                        ->  []

    Ловушка: сама сессия в список не входит, а "s10" не является подключом
    "s1" — сравнивать надо по префиксу "s1/", а не по "s1".
    """
    raise NotImplementedError


def session_delete(store, session_id):
    """Удалить сессию каскадом вместе с сессиями её подагентов.

    Возвращает отсортированный список удалённых ключей.

    store = {"s1": [], "s1/a": [], "s2": []}
    session_delete(store, "s1")  ->  ["s1", "s1/a"];  store == {"s2": []}
    session_delete(store, "нет") ->  []

    Ловушка: без каскада подагентские сессии остаются висеть навсегда —
    ровно тот «session bloat», о котором предупреждает урок. Удаление
    несуществующей сессии — не ошибка, а пустой список.
    """
    raise NotImplementedError


def run_agent(store, session_id, prompt, plan, registry, hooks, model=stub_model):
    """Петля агента: SessionStart, вызовы инструментов по плану, ответ, SessionEnd.

    `plan` — заранее заданная последовательность пар (имя инструмента, args):
    решать, что звать, здесь не нужно, интерес в воротах и в сессии.

    Возвращает {"answer": str, "calls": [...], "journal": [...]}.

    run_agent({}, "s1", "hi", [], {}, {})
        ->  {"answer": "answer[3:333]", "calls": [], "journal": []}
           (модель получает "hi|" — отсюда длина 3)

    В store по ключу session_id копятся ходы: user, по одному tool на каждый
    пункт плана, затем assistant. То есть len(store["s1"]) == len(plan) + 2.

    Ловушка: ответ модели должен зависеть от результатов инструментов, иначе
    вся петля бессмысленна — агент бы ответил и без них.
    """
    raise NotImplementedError


def spawn_subagents(store, parent_session, tasks, registry, hooks, model=stub_model):
    """Запустить подагентов: у каждого свой контекст и свой список инструментов.

    Задача — dict {"name", "prompt", "plan", "allowed"}; "allowed" по правилам
    select_tools (None — все инструменты родителя).

    Возвращает список {"name", "answer", "calls"} в порядке задач.

    Смысл изоляции контекста: ходы подагента лежат в ЕГО сессии
    "<parent>/<name>", а в родительскую попадает ровно один ход на подагента —
    итоговый ответ. Поэтому контекст оркестратора растёт как число задач, а не
    как их суммарная длина; ради этого подагенты и заводят.

    Ловушка: подагенту нельзя отдавать родительский реестр целиком —
    «allowed» на то и «allowed». Инструмент вне списка обязан вернуть
    tool_not_allowed, а не выполниться.
    """
    raise NotImplementedError

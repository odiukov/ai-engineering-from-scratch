"""
Режимы разрешений: матрица режимов и решение allow/ask/deny — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Классы риска по инструменту. Всё, чего в таблице нет, считается "exec" —
# самый узкий класс из тех, что вообще что-то делают. Неизвестный
# инструмент не имеет права попадать в "read".
RISK_BY_TOOL = {
    "Read": "read",
    "Glob": "read",
    "Grep": "read",
    "Edit": "edit",
    "Write": "edit",
    "NotebookEdit": "edit",
    "Bash": "exec",
    "WebFetch": "network",
    "WebSearch": "network",
}

# Матрица шести режимов Claude Code: режим x класс риска -> решение по умолчанию.
# Это именно ПО УМОЛЧАНИЮ: правила разрешений, если они совпали, сильнее.
MODES = {
    "plan":              {"read": "ask",   "edit": "ask",   "exec": "ask",   "network": "ask"},
    "default":           {"read": "allow", "edit": "ask",   "exec": "ask",   "network": "ask"},
    "acceptEdits":       {"read": "allow", "edit": "allow", "exec": "ask",   "network": "ask"},
    "auto":              {"read": "allow", "edit": "allow", "exec": "allow", "network": "allow"},
    "dontAsk":           {"read": "deny",  "edit": "deny",  "exec": "deny",  "network": "deny"},
    "bypassPermissions": {"read": "allow", "edit": "allow", "exec": "allow", "network": "allow"},
}

# Чем больше число, тем строже решение. Порядок ровно такой: deny сильнее ask,
# ask сильнее allow. Никаких «двое за allow против одного за deny».
PRECEDENCE = {"allow": 0, "ask": 1, "deny": 2}


def risk_class(action):
    """Класс риска действия по имени инструмента.

    risk_class({"tool": "Read", "input": "a.py"})       ->  "read"
    risk_class({"tool": "Bash", "input": "ls"})         ->  "exec"
    risk_class({"tool": "SomeMCPTool", "input": "x"})   ->  "exec"

    Неизвестный инструмент НЕ считается чтением. Это не мелочь: подключили
    новый MCP-сервер, забыли внести в таблицу — и он молча получил права
    самого безобидного класса.
    """
    return RISK_BY_TOOL.get(action["tool"], "exec")


def parse_rule(pattern):
    """Разобрать правило вида "Tool" или "Tool(spec)" в кортеж (tool, spec).

    parse_rule("Read")               ->  ("Read", None)
    parse_rule("Bash(git status)")   ->  ("Bash", "git status")
    parse_rule("Bash(git push:*)")   ->  ("Bash", "git push:*")

    spec равен None только у голого имени инструмента. "Bash()" — это пустая
    строка, а не None; они означают разное и до rule_matches доезжают
    по-разному.
    """
    if pattern.endswith(")") and "(" in pattern:
        head, _, rest = pattern.partition("(")
        return (head, rest[:-1])
    return (pattern, None)


def rule_matches(pattern, action):
    """Совпало ли правило с действием.

    rule_matches("Read", {"tool": "Read", "input": "a.py"})               ->  True
    rule_matches("Bash(git push:*)", {"tool": "Bash", "input": "git push origin"})  ->  True
    rule_matches("Bash(git push:*)", {"tool": "Bash", "input": "git status"})       ->  False
    rule_matches("Bash(ls)", {"tool": "Bash", "input": "ls -la"})         ->  False

    Три формы spec:
      None или "*"       — любой ввод этого инструмента
      что-то с ":*" в конце — префиксное совпадение по началу ввода
      всё остальное      — точное совпадение ввода

    Имя инструмента сравнивается регистрозависимо: "bash" и "Bash" — разные
    вещи, и подгонять регистр «чтобы совпало» нельзя.
    """
    tool, spec = parse_rule(pattern)
    if tool != action["tool"]:
        return False
    if spec is None or spec == "*":
        return True
    if spec.endswith(":*"):
        return action["input"].startswith(spec[:-2])
    return action["input"] == spec


def rule_specificity(pattern):
    """Насколько правило конкретно: кортеж (вид, длина), больше — конкретнее.

    rule_specificity("Bash")             ->  (0, 0)
    rule_specificity("Bash(*)")          ->  (0, 0)
    rule_specificity("Bash(git:*)")      ->  (1, 3)
    rule_specificity("Bash(git status)") ->  (2, 10)

    Кортеж, а не одно число, — чтобы точное правило всегда било любое
    префиксное, каким бы длинным префикс ни был. Кортежи сравниваются
    поэлементно, так что max() по ним работает сам собой.

    Зачем это нужно: без детерминированного порядка «какое из двух
    совпавших правил главнее» ответ зависел бы от порядка строк в конфиге.
    """
    _tool, spec = parse_rule(pattern)
    if spec is None or spec == "*":
        return (0, 0)
    if spec.endswith(":*"):
        return (1, len(spec) - 2)
    return (2, len(spec))


def strictest(decisions):
    """Самое строгое из решений: deny > ask > allow.

    strictest(["allow", "deny"])          ->  "deny"
    strictest(["allow", "ask", "allow"])  ->  "ask"
    strictest(["allow"])                  ->  "allow"

    Это не голосование: одного deny достаточно, сколько бы allow ни стояло
    рядом. Пустой список — ошибка конфигурации, а не «значит, allow»:
    подними ValueError, иначе тихая дыра в правах.
    """
    if not decisions:
        raise ValueError("no decisions to combine")
    return max(decisions, key=lambda d: PRECEDENCE[d])


def mode_decision(mode, action):
    """Решение по умолчанию для действия в данном режиме — из матрицы MODES.

    mode_decision("acceptEdits", {"tool": "Write", "input": "a.py"})  ->  "allow"
    mode_decision("acceptEdits", {"tool": "Bash", "input": "ls"})     ->  "ask"
    mode_decision("plan", {"tool": "Read", "input": "a.py"})          ->  "ask"

    Неизвестный режим — ValueError. Молча подставлять "default" нельзя:
    опечатка в имени режима не должна расширять права.
    """
    if mode not in MODES:
        raise ValueError(f"unknown permission mode: {mode!r}")
    return MODES[mode][risk_class(action)]


def classifier_verdict(action, allowed_tools):
    """Вердикт классификатора Auto Mode: инструмент вне объявленной задачи -> "ask".

    classifier_verdict({"tool": "Read", "input": "a.py"}, ("Read", "Edit"))  ->  "allow"
    classifier_verdict({"tool": "Bash", "input": "ls"}, ("Read", "Edit"))    ->  "ask"
    classifier_verdict({"tool": "Read", "input": "a.py"}, ())                ->  "ask"

    Классификатор не блокирует насмерть, а возвращает решение человеку —
    поэтому "ask", а не "deny".

    Он судит ОДНО действие. Три разрешённых действия подряд (прочитать
    ключи, записать во временный файл, запушить) он пропустит: композиция
    не входит в его поле зрения. Это ограничение по устройству, а не баг.
    """
    return "allow" if action["tool"] in allowed_tools else "ask"


def decide(action, rules, mode, allowed_tools=()):
    """Итоговое решение allow/ask/deny для действия.

    rules — список dict с ключами "pattern" и "decision".

    Порядок разбора:
      1. Отобрать совпавшие правила.
      2. Если совпавшие есть — оставить только самые конкретные
         (максимум rule_specificity) и взять из них самое строгое.
      3. Если ни одно не совпало — взять умолчание режима.
      4. В режиме "auto" дополнительно ужесточить умолчание вердиктом
         классификатора.

    decide({"tool": "Bash", "input": "git status"},
           [{"pattern": "Bash(*)", "decision": "deny"},
            {"pattern": "Bash(git status:*)", "decision": "allow"}],
           "default")                                        ->  "allow"

    Два правила урока, которые легко перепутать:
      * при РАВНОЙ конкретности deny перевешивает allow;
      * при РАЗНОЙ конкретности выигрывает более конкретное — даже если
        более общее правило говорит deny.
    Сначала фильтр по конкретности, потом строгость. Не наоборот.
    """
    matched = [r for r in rules if rule_matches(r["pattern"], action)]
    if matched:
        best = max(rule_specificity(r["pattern"]) for r in matched)
        winners = [
            r["decision"] for r in matched if rule_specificity(r["pattern"]) == best
        ]
        return strictest(winners)
    baseline = mode_decision(mode, action)
    if mode == "auto":
        return strictest([baseline, classifier_verdict(action, allowed_tools)])
    return baseline

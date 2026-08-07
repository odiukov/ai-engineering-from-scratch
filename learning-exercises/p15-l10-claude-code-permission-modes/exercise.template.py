"""
Режимы разрешений: матрица режимов и решение allow/ask/deny

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l10-claude-code-permission-modes
Разбор:  /check-code p15-l10-claude-code-permission-modes
"""

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
MODES = {
    "plan":              {"read": "ask",   "edit": "ask",   "exec": "ask",   "network": "ask"},
    "default":           {"read": "allow", "edit": "ask",   "exec": "ask",   "network": "ask"},
    "acceptEdits":       {"read": "allow", "edit": "allow", "exec": "ask",   "network": "ask"},
    "auto":              {"read": "allow", "edit": "allow", "exec": "allow", "network": "allow"},
    "dontAsk":           {"read": "deny",  "edit": "deny",  "exec": "deny",  "network": "deny"},
    "bypassPermissions": {"read": "allow", "edit": "allow", "exec": "allow", "network": "allow"},
}
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
    raise NotImplementedError


def parse_rule(pattern):
    """Разобрать правило вида "Tool" или "Tool(spec)" в кортеж (tool, spec).

    parse_rule("Read")               ->  ("Read", None)
    parse_rule("Bash(git status)")   ->  ("Bash", "git status")
    parse_rule("Bash(git push:*)")   ->  ("Bash", "git push:*")

    spec равен None только у голого имени инструмента. "Bash()" — это пустая
    строка, а не None; они означают разное и до rule_matches доезжают
    по-разному.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def strictest(decisions):
    """Самое строгое из решений: deny > ask > allow.

    strictest(["allow", "deny"])          ->  "deny"
    strictest(["allow", "ask", "allow"])  ->  "ask"
    strictest(["allow"])                  ->  "allow"

    Это не голосование: одного deny достаточно, сколько бы allow ни стояло
    рядом. Пустой список — ошибка конфигурации, а не «значит, allow»:
    подними ValueError, иначе тихая дыра в правах.
    """
    raise NotImplementedError


def mode_decision(mode, action):
    """Решение по умолчанию для действия в данном режиме — из матрицы MODES.

    mode_decision("acceptEdits", {"tool": "Write", "input": "a.py"})  ->  "allow"
    mode_decision("acceptEdits", {"tool": "Bash", "input": "ls"})     ->  "ask"
    mode_decision("plan", {"tool": "Read", "input": "a.py"})          ->  "ask"

    Неизвестный режим — ValueError. Молча подставлять "default" нельзя:
    опечатка в имени режима не должна расширять права.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError

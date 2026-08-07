"""Входные данные для замера скорости."""

_tools = ("Read", "Write", "Bash", "WebFetch", "SomeMCPTool")
_action = {"tool": "Bash", "input": "git status --short"}

# Много правил, и совпадает из них далеко не одно: наивный перебор с
# повторным разбором строки на каждое сравнение здесь заметно проседает.
_rules = [{"pattern": f"Bash(git log --oneline -{i}:*)", "decision": "allow"}
          for i in range(1500)]
_rules += [{"pattern": "Bash(*)", "decision": "deny"},
           {"pattern": "Bash(git:*)", "decision": "ask"},
           {"pattern": "Bash(git status --short)", "decision": "allow"}]

_decisions = ["allow"] * 3000 + ["ask"] * 500 + ["deny"]

BENCH = {
    "risk_class": ({"tool": "SomeMCPTool", "input": "x"},),
    "parse_rule": ("Bash(git push --force-with-lease:*)",),
    "rule_matches": ("Bash(git status:*)", _action),
    "rule_specificity": ("Bash(git push --force-with-lease:*)",),
    "strictest": (_decisions,),
    "mode_decision": ("acceptEdits", _action),
    "classifier_verdict": (_action, _tools),
    "decide": (_action, _rules, "default"),
}

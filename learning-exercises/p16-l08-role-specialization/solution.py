"""
Специализация ролей: planner, executor, critic, verifier — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Матрица «роль × компетенция» из урока. Значение — насколько хорошо роль
# делает задачу такого типа, от 0.0 до 1.0. Это и есть та самая
# специализация: у каждой роли один пик и провалы в остальном.
ROLE_MATRIX = {
    "planner": {"plan": 0.90, "code": 0.30, "review": 0.40, "test": 0.20},
    "executor": {"plan": 0.30, "code": 0.90, "review": 0.30, "test": 0.40},
    "critic": {"plan": 0.50, "code": 0.40, "review": 0.90, "test": 0.30},
    "verifier": {"plan": 0.10, "code": 0.20, "review": 0.50, "test": 0.95},
}

# Потолок круга «критик вернул — исполнитель переделал» из чек-листа урока.
DEFAULT_REVISION_BUDGET = 2


def competence(matrix, role, skill):
    """Компетенция роли в навыке. Неизвестный навык — это 0.0, не ошибка.

    competence(ROLE_MATRIX, "executor", "code")   ->  0.9
    competence(ROLE_MATRIX, "executor", "dance")  ->  0.0

    Ловушка: неизвестная РОЛЬ — это уже ошибка конфигурации, а не «ноль».
    Роль опечатали — брось ValueError, иначе система тихо назначит нулевого
    исполнителя и будет считать это нормой.
    """
    if role not in matrix:
        raise ValueError(f"unknown role {role!r}")
    return matrix[role].get(skill, 0.0)


def best_role(matrix, skill):
    """Роль с максимальной компетенцией в навыке. Ничья — по алфавиту.

    best_role(ROLE_MATRIX, "code")  ->  'executor'
    best_role(ROLE_MATRIX, "test")  ->  'verifier'

    Алфавит при ничьей — не эстетика, а детерминизм: без него одна и та же
    матрица давала бы разные назначения между запусками.
    """
    if not matrix:
        raise ValueError("role matrix must not be empty")
    return min(matrix, key=lambda role: (-competence(matrix, role, skill), role))


def assign_tasks(matrix, tasks):
    """Назначение задач по матрице: {имя задачи: роль}.

    tasks — список пар (имя задачи, навык).

    assign_tasks(ROLE_MATRIX, [("spec", "plan"), ("impl", "code")])
        ->  {'spec': 'planner', 'impl': 'executor'}

    Ловушка: одинаковые имена задач молча затрут друг друга в словаре.
    Лучше упасть с ValueError — дубль в списке задач это баг у вызывающего.
    """
    assignment = {}
    for name, skill in tasks:
        if name in assignment:
            raise ValueError(f"duplicate task name {name!r}")
        assignment[name] = best_role(matrix, skill)
    return assignment


def team_quality(matrix, tasks):
    """Средняя компетенция команды на наборе задач после назначения.

    team_quality(ROLE_MATRIX, [("impl", "code")])                    ->  0.9
    team_quality(ROLE_MATRIX, [("impl", "code"), ("t", "test")])     ->  0.925

    Пустой список задач — это не ноль качества, а отсутствие задач.
    Брось ValueError, чтобы 0.0 нельзя было спутать с «команда бесполезна».
    """
    if not tasks:
        raise ValueError("tasks must not be empty")
    assignment = assign_tasks(matrix, tasks)
    return sum(
        competence(matrix, assignment[name], skill) for name, skill in tasks
    ) / len(tasks)


def generalist(skills, level):
    """Матрица из одной роли-универсала, одинаково ровной во всех навыках.

    generalist(["code", "test"], 0.6)
        ->  {'generalist': {'code': 0.6, 'test': 0.6}}

    Нужна как контроль: специализация имеет смысл только если бьёт ровного
    универсала того же уровня. На однородных задачах не бьёт — это честный
    результат, а не провал эксперимента.
    """
    if not 0.0 <= level <= 1.0:
        raise ValueError("level must be in [0, 1]")
    return {"generalist": {skill: level for skill in skills}}


def critic_review(spec, artifact):
    """Субъективная проверка: смотрит ТОЛЬКО на текст артефакта.

    spec — dict с ключами "name" и "must_contain" (список подстрок).
    artifact — dict с ключами "code" (текст) и "fn" (вызываемый объект).

    Возвращает пару (approved, notes).

    critic_review({"name": "f", "must_contain": ["def "]},
                  {"code": "def f(): ...", "fn": f})   ->  (True, [])

    Критик — это LLM-ревьюер. Он видит форму, а не поведение, и его легко
    обмануть правдоподобным кодом. Тесты урока это специально ловят.
    """
    notes = []
    code = artifact["code"]
    for needle in spec["must_contain"]:
        if needle not in code:
            notes.append(f"missing {needle!r}")
    if spec["name"] not in code:
        notes.append(f"function name does not match spec {spec['name']!r}")
    return (not notes, notes)


def verifier_run(spec, artifact):
    """Объективная проверка: гоняет spec["tests"] против artifact["fn"].

    tests — список пар (кортеж аргументов, ожидаемый результат).
    Возвращает пару (passed, failures) со списком описаний провалов.

    verifier_run({"tests": [((1, 2), 3)]}, {"fn": lambda a, b: a + b})
        ->  (True, [])
    verifier_run({"tests": [((1, 2), 3)]}, {"fn": lambda a, b: a * b})
        ->  (False, ['(1, 2): expected 3, got 2'])

    Ловушка: исключение внутри проверяемой функции — это тоже провал теста,
    а не крах верификатора. Ловим и записываем, иначе один кривой артефакт
    уронит весь пайплайн.
    """
    failures = []
    fn = artifact["fn"]
    for args, expected in spec["tests"]:
        try:
            got = fn(*args)
        except Exception as exc:  # noqa: BLE001 — верификатор обязан пережить чужой баг
            failures.append(f"{args}: raised {type(exc).__name__}")
            continue
        if got != expected:
            failures.append(f"{args}: expected {expected}, got {got}")
    return (not failures, failures)


def run_pipeline(spec, executor, max_revisions=DEFAULT_REVISION_BUDGET, use_verifier=True):
    """Пайплайн planner -> executor -> critic -> verifier с бюджетом правок.

    executor — вызываемый объект executor(spec, feedback) -> artifact,
    где feedback это список замечаний прошлого круга (на первом — пустой).

    Возвращает dict:
        {"status": "shipped" | "escalated", "revisions": int,
         "artifact": ..., "notes": [...]}

    status="escalated" значит «бюджет правок кончился, зовите человека».

    Порядок из чек-листа урока: сначала дешёвый критик, потом дорогой
    верификатор. use_verifier=False воспроизводит all-LLM анти-паттерн:
    пайплайн отгружает правдоподобный баг, потому что проверять было некому.
    """
    if max_revisions < 0:
        raise ValueError("max_revisions must not be negative")
    feedback = []
    for revisions in range(max_revisions + 1):
        artifact = executor(spec, feedback)
        approved, notes = critic_review(spec, artifact)
        if not approved:
            feedback = notes
            continue
        if use_verifier:
            passed, failures = verifier_run(spec, artifact)
            if not passed:
                feedback = failures
                continue
        return {
            "status": "shipped",
            "revisions": revisions,
            "artifact": artifact,
            "notes": [],
        }
    return {
        "status": "escalated",
        "revisions": max_revisions,
        "artifact": artifact,
        "notes": feedback,
    }

"""
Специализация ролей: planner, executor, critic, verifier

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l08-role-specialization
Разбор:  /check-code p16-l08-role-specialization
"""

ROLE_MATRIX = {
    "planner": {"plan": 0.90, "code": 0.30, "review": 0.40, "test": 0.20},
    "executor": {"plan": 0.30, "code": 0.90, "review": 0.30, "test": 0.40},
    "critic": {"plan": 0.50, "code": 0.40, "review": 0.90, "test": 0.30},
    "verifier": {"plan": 0.10, "code": 0.20, "review": 0.50, "test": 0.95},
}
DEFAULT_REVISION_BUDGET = 2


def competence(matrix, role, skill):
    """Компетенция роли в навыке. Неизвестный навык — это 0.0, не ошибка.

    competence(ROLE_MATRIX, "executor", "code")   ->  0.9
    competence(ROLE_MATRIX, "executor", "dance")  ->  0.0

    Ловушка: неизвестная РОЛЬ — это уже ошибка конфигурации, а не «ноль».
    Роль опечатали — брось ValueError, иначе система тихо назначит нулевого
    исполнителя и будет считать это нормой.
    """
    raise NotImplementedError


def best_role(matrix, skill):
    """Роль с максимальной компетенцией в навыке. Ничья — по алфавиту.

    best_role(ROLE_MATRIX, "code")  ->  'executor'
    best_role(ROLE_MATRIX, "test")  ->  'verifier'

    Алфавит при ничьей — не эстетика, а детерминизм: без него одна и та же
    матрица давала бы разные назначения между запусками.
    """
    raise NotImplementedError


def assign_tasks(matrix, tasks):
    """Назначение задач по матрице: {имя задачи: роль}.

    tasks — список пар (имя задачи, навык).

    assign_tasks(ROLE_MATRIX, [("spec", "plan"), ("impl", "code")])
        ->  {'spec': 'planner', 'impl': 'executor'}

    Ловушка: одинаковые имена задач молча затрут друг друга в словаре.
    Лучше упасть с ValueError — дубль в списке задач это баг у вызывающего.
    """
    raise NotImplementedError


def team_quality(matrix, tasks):
    """Средняя компетенция команды на наборе задач после назначения.

    team_quality(ROLE_MATRIX, [("impl", "code")])                    ->  0.9
    team_quality(ROLE_MATRIX, [("impl", "code"), ("t", "test")])     ->  0.925

    Пустой список задач — это не ноль качества, а отсутствие задач.
    Брось ValueError, чтобы 0.0 нельзя было спутать с «команда бесполезна».
    """
    raise NotImplementedError


def generalist(skills, level):
    """Матрица из одной роли-универсала, одинаково ровной во всех навыках.

    generalist(["code", "test"], 0.6)
        ->  {'generalist': {'code': 0.6, 'test': 0.6}}

    Нужна как контроль: специализация имеет смысл только если бьёт ровного
    универсала того же уровня. На однородных задачах не бьёт — это честный
    результат, а не провал эксперимента.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError

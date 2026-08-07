"""
Паттерны оркестрации: supervisor, swarm, hierarchical

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l28-orchestration-patterns
Разбор:  /check-code p14-l28-orchestration-patterns
"""

INTENT_KEYWORDS = {
    "refund": ("refund", "money back", "chargeback", "возврат"),
    "bug": ("crash", "error", "bug", "broken"),
    "sales": ("price", "pricing", "quote", "plan"),
}
SPECIALISTS = {"refund": "billing_agent", "bug": "support_agent", "sales": "sales_agent"}
SWARM_RING = ("billing_agent", "support_agent", "sales_agent")
TEAMS = {"finance": ("refund",), "product": ("bug", "sales")}
PATTERNS = ("supervisor", "swarm", "hierarchical")


def classify(text):
    """Намерение задачи по словам-признакам: refund, bug, sales или unknown.

    classify("my app keeps crashing with an error")  ->  "bug"
    classify("I want a refund")                      ->  "refund"
    classify("привет")                               ->  "unknown"

    Побеждает намерение с наибольшим числом совпавших слов. Ничья решается
    по алфавиту: "refund and bug" -> "bug".

    Детерминированная ничья здесь не косметика. Роутер, который при равном
    счёте отвечает по порядку словаря, начнёт менять маршрут от перестановки
    ключей — и трейсы перестанут воспроизводиться.
    """
    raise NotImplementedError


def supervisor_route(tasks):
    """Supervisor-worker: центральный роутер раздаёт задачи специалистам.

    Вернуть {"assignments": [имя специалиста или None], "ops": int}.

    supervisor_route(["I want a refund"])
      ->  {"assignments": ["billing_agent"], "ops": 2}
    supervisor_route(["привет"])
      ->  {"assignments": [None], "ops": 1}

    Стоимость: один ход роутера плюс один ход специалиста. Нераспознанная
    задача стоит один ход — роутер посмотрел и никого не позвал.

    Специалисты друг с другом не разговаривают: весь трафик через роутер.
    Отсюда и главное свойство — трейс читается сверху вниз.
    """
    raise NotImplementedError


def swarm_route(tasks, entry=SWARM_RING[0], max_hops=2):
    """Swarm: роутера нет, агенты передают задачу соседу по кольцу.

    Вернуть {"assignments": [...], "ops": int, "handoffs": [путь по каждой задаче]}.

    swarm_route(["I want a refund"])["handoffs"]  ->  [["billing_agent"]]
    swarm_route(["crash on save"])["handoffs"]    ->  [["billing_agent", "support_agent"]]
    swarm_route(["привет"])["assignments"]        ->  [None]

    Каждый агент, которому задача не подходит, отдаёт её следующему в
    SWARM_RING. Счётчик передач обязателен: без него нераспознанная задача
    крутится по кольцу вечно. Исчерпали max_hops — задача не назначена.

    Плюс: до «своего» специалиста иногда доходит за один ход, дешевле
    супервизора. Минус: единой точки контроля нет, и отладка трейса тяжелее.
    """
    raise NotImplementedError


def detect_bouncing(handoff_log):
    """Есть ли в журнале передач возврат к предыдущему агенту: A -> B -> A.

    detect_bouncing(["a", "b", "a"])       ->  True
    detect_bouncing(["a", "b", "c"])       ->  False
    detect_bouncing(["a", "a"])            ->  False

    Повтор подряд ("a", "a") — это не пинг-понг, а один агент, продолжающий
    работу. Ловим именно возврат через одного.

    Урок называет это главной болячкой swarm: A отдаёт B, B не понимает и
    отдаёт обратно A. Счётчик хопов такое останавливает, но не объясняет —
    объясняет вот эта проверка.
    """
    raise NotImplementedError


def hierarchical_route(tasks):
    """Hierarchical: верхний супервизор -> супервизор команды -> специалист.

    Вернуть {"assignments": [...], "ops": int, "teams": [имя команды или None]}.

    hierarchical_route(["I want a refund"])
      ->  {"assignments": ["billing_agent"], "ops": 3, "teams": ["finance"]}

    Стоимость три хода вместо двух: за лишний уровень платят всегда.
    Оправдано это ровно одним — когда описания всех специалистов перестают
    помещаться в контекст одного супервизора. Три уровня «потому что
    энтерпрайз» урок прямо называет fake hierarchy.
    """
    raise NotImplementedError


def run_parallel(tasks, worker, completion_order=None):
    """Параллельный запуск: результаты в порядке ЗАДАЧ, а не завершения.

    completion_order — перестановка индексов, имитирующая, кто закончил
    раньше. None означает порядок задач.

    run_parallel(["a", "b"], str.upper)                       ->  ["A", "B"]
    run_parallel(["a", "b"], str.upper, completion_order=[1, 0])  ->  ["A", "B"]
    run_parallel(["a"], str.upper, completion_order=[0, 0])   ->  ValueError

    Не перестановка индексов -> ValueError.

    Свойство, ради которого функция и существует: параллельная ветка обязана
    давать один и тот же результат при любом порядке завершения. Если код
    делает `results.append(...)` по мере готовности, порядок ответов начнёт
    зависеть от того, какой воркер сегодня быстрее, — и воспроизвести баг
    станет невозможно.
    """
    raise NotImplementedError


def pick_pattern(specialists, latency_critical=False, accuracy_critical=False,
                 supervisor_context_ok=True):
    """Выбор топологии по порядку решений из урока.

    pick_pattern(1)                                  ->  "single_agent"
    pick_pattern(3)                                  ->  "supervisor"
    pick_pattern(12, supervisor_context_ok=False)    ->  "hierarchical"
    pick_pattern(3, accuracy_critical=True)          ->  "debate"
    pick_pattern(3, latency_critical=True)           ->  "swarm"

    Порядок проверок, сверху вниз:
      1. один специалист — топология не нужна вообще;
      2. контекст супервизора не вмещает описания — hierarchical (это
         жёсткое ограничение, а не предпочтение);
      3. точность важнее стоимости — debate;
      4. задержка важнее ясности рассуждений — swarm;
      5. иначе — supervisor.

    specialists < 1 -> ValueError.

    Смысл функции — сопротивление topology-first мышлению: сначала называем
    ограничение, потом получаем паттерн, а не наоборот.
    """
    raise NotImplementedError


def compare_patterns(tasks):
    """Один и тот же набор задач через все PATTERNS: кто во что обходится.

    Вернуть {паттерн: {"assignments": [...], "ops": int}}.

    compare_patterns(["I want a refund"])["supervisor"]["ops"]     ->  2
    compare_patterns(["I want a refund"])["hierarchical"]["ops"]   ->  3

    Назначения обязаны совпасть у всех трёх — маршрут не должен зависеть от
    топологии. Различаться должна только цена. Ровно это и есть аргумент
    урока: если ответ один, платить за лишние уровни незачем.
    """
    raise NotImplementedError

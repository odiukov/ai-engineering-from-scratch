"""
AI Scientist v2: конвейер автономного научного цикла

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l05-ai-scientist-v2
Разбор:  /check-code p15-l05-ai-scientist-v2
"""

DEFAULT_CONFIG = {
    "novelty_mislabel": 0.25,
    "experiment_failure": 0.42,
    "retry_recovery": 0.55,
    "polish_masks_weakness": 0.70,
    "writeup_success": 0.85,
    "internal_review_accept": 0.50,
}
REQUIRED_CHECKS = ("experiment_reproduced", "novelty_verified", "human_signoff")


def novelty_check(rng, is_known, mislabel_rate):
    """Литературный поиск: объявить идею "novel" или "known".

    novelty_check(rng, False, 0.9)  ->  "novel"   (по-настоящему новая — всегда)
    novelty_check(rng, True, 0.0)   ->  "known"   (идеальный поиск не ошибается)
    novelty_check(rng, True, 0.25)  ->  "novel" примерно в четверти случаев

    Ошибка здесь ОДНОСТОРОННЯЯ: система принимает старое за новое, но не
    наоборот. Это ровно то, что намерили Beel et al., и это научный аналог
    галлюцинации — уверенное утверждение о том, чего система не проверила.

    Соблазн сделать ошибку двусторонней («иногда новую идею объявим известной»)
    выглядит симметрично и честно, но ломает главный вывод: конвейер завышает
    новизну, а не шумит вокруг правды.
    """
    raise NotImplementedError


def run_experiment(rng, failure_rate, retry_recovery):
    """Выполнить эксперимент. Вернуть {"ok", "flawed", "retried"}.

    run_experiment(rng, 0.0, 0.0)  ->  {"ok": True, "flawed": False, "retried": False}
    run_experiment(rng, 1.0, 1.0)  ->  {"ok": True, "flawed": True,  "retried": True}
    run_experiment(rng, 1.0, 0.0)  ->  {"ok": False, "flawed": True, "retried": True}

    Ключевое моделирующее решение, и оно не косметическое: эксперимент,
    ВОССТАНОВЛЕННЫЙ ретраем, остаётся с остаточным изъяном (flawed=True).
    Ретрай чинит падение — несовпадение размерностей, забытый импорт, — но
    делает это, не перепроверяя, что численный результат всё ещё осмыслен.

    Именно эта категория потом полируется фигурами и уезжает в подачу. Если
    сделать ретрай «чинящим до чистоты», из симулятора исчезнет весь сюжет.
    """
    raise NotImplementedError


def polish_figures(rng, paper, mask_rate):
    """VLM-критика фигур. Вернуть КОПИЮ paper с полем "masked".

    paper — словарь {"claim": {...}, "novelty": ..., "experiment": {...}}.

    polish_figures(rng, {"experiment": {"flawed": False, ...}, ...}, 1.0)
        ->  копия с "masked": False   (скрывать нечего)
    polish_figures(rng, {"experiment": {"flawed": True, ...}, ...}, 1.0)
        ->  копия с "masked": True

    Полировка НЕ трогает сам эксперимент: изъян никуда не девается, меняется
    только то, насколько он заметен читателю. Это и есть presentation-quality
    gap — качество подачи обгоняет качество исследования.

    Возвращаем копию, а не правим на месте: исходная работа должна остаться
    доступной для сравнения «до и после», иначе аудит невозможен.
    """
    raise NotImplementedError


def supports_conclusion(paper):
    """Подкреплён ли вывод экспериментом. Единственная содержательная проверка.

    supports_conclusion({"claim": {"effect_observed": True},
                         "experiment": {"ok": True, "flawed": False}})   ->  True
    supports_conclusion({"claim": {"effect_observed": True},
                         "experiment": {"ok": True, "flawed": True}})    ->  False
    supports_conclusion({"claim": {"effect_observed": False},
                         "experiment": {"ok": True, "flawed": False}})   ->  False

    Три условия и все обязательны: эксперимент отработал, отработал БЕЗ изъяна,
    и эффект в нём действительно наблюдался. Вывод, которому нечего
    предъявить, отвергается — независимо от того, как выглядят фигуры.

    Поле "masked" здесь не участвует вообще. Проверка, которая смотрит на
    полировку, перестаёт быть проверкой.
    """
    raise NotImplementedError


def review(paper, strict=True):
    """Рецензент: "accept" или "reject".

    Строгий смотрит только на supports_conclusion. Слабый принимает ещё и
    работу, чьи фигуры выглядят убедительно.

    review({...чистая работа...})                  ->  "accept"
    review({...сломанная, но отполированная...})   ->  "reject"
    review({...сломанная, но отполированная...}, strict=False)  ->  "accept"

    Разница между двумя строчками кода — это разница между рецензентом,
    который проверяет результат, и рецензентом, который проверяет
    впечатление. Второй в среднем быстрее, дешевле и охотнее соглашается.
    """
    raise NotImplementedError


def release_gate(paper, checks):
    """Ворота выхода из песочницы. Вернуть (можно_ли, список_провалов).

    release_gate(чистая_работа, {"experiment_reproduced": True,
                                 "novelty_verified": True,
                                 "human_signoff": True})   ->  (True, [])
    release_gate(чистая_работа, {})
        ->  (False, ["experiment_reproduced", "novelty_verified", "human_signoff"])
    release_gate(сломанная_работа, все_галочки)
        ->  (False, ["conclusion_supported"])

    Отсутствующая галочка считается НЕ пройденной, а не «наверное, ок».
    Посторонние ключи в checks игнорируются: список обязательных проверок
    задаёт гейт, а не тот, кого проверяют. Иначе агент, дописавший себе
    "looks_great": True, начнёт влиять на решение о публикации.

    Порядок провалов фиксирован: сначала содержательная проверка, дальше
    REQUIRED_CHECKS в объявленном порядке. Отчёт должен читаться одинаково.
    """
    raise NotImplementedError


def run_loop(rng, config, is_known=True, effect_observed=True):
    """Один прогон конвейера. Вернуть исход.

    Ключи исхода:
      "submitted" — дошло ли до подачи;
      "stage"     — на какой стадии брошено ("" у поданных);
      "paper"     — сама работа (None, если до неё не дошли);
      "clean"     — подан ли вывод, подкреплённый экспериментом.

    run_loop(rng, {**DEFAULT_CONFIG, "experiment_failure": 1.0,
                   "retry_recovery": 0.0})["stage"]   ->  "experiment"

    Стадии брошенных прогонов: "novelty", "experiment", "writeup", "review".

    Внутренний рецензент здесь СЛАБЫЙ (strict=False) — так устроена система в
    статье. Именно поэтому отполированная работа со сломанным экспериментом
    доходит до подачи: рецензент, который её отсеял бы, в цикле не стоит.
    """
    raise NotImplementedError


def summarize(outcomes):
    """Свести прогоны в отчёт.

    Ключи отчёта: "trials", "submitted", "clean", "flawed", "submit_rate",
    "clean_share_of_submitted", "abandoned_by_stage".

    summarize([{"submitted": True, "stage": "", "clean": True, "paper": {}}])
        ->  {"trials": 1, "submitted": 1, "clean": 1, "flawed": 0,
             "submit_rate": 1.0, "clean_share_of_submitted": 1.0,
             "abandoned_by_stage": {}}

    Две корзины поданных работ — чистые и с изъяном — исчерпывающие: их сумма
    обязана равняться числу поданных. Если они разъезжаются, значит какая-то
    категория брака тихо потерялась, а именно её и надо считать.

    Ловушка: ноль поданных. Доля чистых среди нуля — это 0.0, а не
    ZeroDivisionError и не 1.0 («все ноль работ были чистыми»).

    Пустой список прогонов — ValueError: отчёта ни о чём не бывает.
    """
    raise NotImplementedError

"""
AI Scientist v2: конвейер автономного научного цикла — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Собираем руками цикл из статьи Yamada et al. (arXiv:2504.08066) как конечный
автомат:

    идея -> проверка новизны -> эксперимент -> фигуры -> текст ->
    внутренний рецензент -> подача

Вероятности отказов на стадиях взяты из независимой оценки Beel et al.
(arXiv:2502.14297): 42% экспериментов падали на ошибках в коде, проверка
новизны регулярно объявляла новым давно опубликованное, а VLM-полировка
фигур доводила до публикационного вида работу, в которой эксперимент сломан.

Последний пункт — главный. Система, которая производит убедительные артефакты
без убедительного исследования, ОПАСНЕЕ той, что падает заметно.

Никакой сети, никаких LLM: всё на словарях и одном воспроизводимом rng.
"""

# Числа из Beel et al. (2025). Это не наши предположения, а измеренное
# поведение реальной системы.
DEFAULT_CONFIG = {
    "novelty_mislabel": 0.25,
    "experiment_failure": 0.42,
    "retry_recovery": 0.55,
    "polish_masks_weakness": 0.70,
    "writeup_success": 0.85,
    "internal_review_accept": 0.50,
}

# Ворота выхода из песочницы: что должно быть подтверждено ЧЕЛОВЕКОМ, прежде
# чем артефакт уедет в venue. Порядок фиксирован — отчёт о провалах должен
# читаться одинаково от прогона к прогону.
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
    if not 0.0 <= mislabel_rate <= 1.0:
        raise ValueError("mislabel_rate — доля, она лежит в [0, 1]")
    if not is_known:
        return "novel"
    return "novel" if rng.random() < mislabel_rate else "known"


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
    if rng.random() >= failure_rate:
        return {"ok": True, "flawed": False, "retried": False}
    if rng.random() < retry_recovery:
        return {"ok": True, "flawed": True, "retried": True}
    return {"ok": False, "flawed": True, "retried": True}


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
    polished = dict(paper)
    polished["masked"] = bool(paper["experiment"]["flawed"]) and rng.random() < mask_rate
    return polished


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
    experiment = paper["experiment"]
    return (
        bool(experiment["ok"])
        and not experiment["flawed"]
        and bool(paper["claim"]["effect_observed"])
    )


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
    if supports_conclusion(paper):
        return "accept"
    if not strict and paper.get("masked"):
        return "accept"
    return "reject"


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
    failed = []
    if not supports_conclusion(paper):
        failed.append("conclusion_supported")
    failed.extend(name for name in REQUIRED_CHECKS if not checks.get(name, False))
    return (not failed, failed)


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
    if novelty_check(rng, is_known, config["novelty_mislabel"]) == "known":
        return {"submitted": False, "stage": "novelty", "paper": None, "clean": False}

    experiment = run_experiment(rng, config["experiment_failure"],
                                config["retry_recovery"])
    if not experiment["ok"]:
        return {"submitted": False, "stage": "experiment", "paper": None,
                "clean": False}

    paper = polish_figures(
        rng,
        {"claim": {"effect_observed": effect_observed},
         "novelty": "novel",
         "experiment": experiment},
        config["polish_masks_weakness"],
    )

    if rng.random() >= config["writeup_success"]:
        return {"submitted": False, "stage": "writeup", "paper": paper,
                "clean": False}

    accepted = review(paper, strict=False) == "accept"
    if not accepted or rng.random() >= config["internal_review_accept"]:
        return {"submitted": False, "stage": "review", "paper": paper,
                "clean": False}

    return {"submitted": True, "stage": "", "paper": paper,
            "clean": supports_conclusion(paper)}


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
    if not outcomes:
        raise ValueError("нечего сводить: список прогонов пуст")
    submitted = [o for o in outcomes if o["submitted"]]
    clean = [o for o in submitted if o["clean"]]
    by_stage = {}
    for outcome in outcomes:
        if not outcome["submitted"]:
            by_stage[outcome["stage"]] = by_stage.get(outcome["stage"], 0) + 1
    return {
        "trials": len(outcomes),
        "submitted": len(submitted),
        "clean": len(clean),
        "flawed": len(submitted) - len(clean),
        "submit_rate": len(submitted) / len(outcomes),
        "clean_share_of_submitted": len(clean) / len(submitted) if submitted else 0.0,
        "abandoned_by_stage": by_stage,
    }

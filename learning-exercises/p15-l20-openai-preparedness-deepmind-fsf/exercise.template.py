"""
OpenAI Preparedness Framework и DeepMind Frontier Safety Framework

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l20-openai-preparedness-deepmind-fsf
Разбор:  /check-code p15-l20-openai-preparedness-deepmind-fsf
"""

CAPABILITY_AXES = (
    "long_range_autonomy",
    "sandbagging",
    "autonomous_replication",
    "undermining_safeguards",
    "rnd_automation",
    "cyber_uplift",
    "bio_uplift",
)
OPENAI_PF_V2 = {
    "name": "OpenAI Preparedness Framework v2",
    "table": {
        "long_range_autonomy": ("Research", "observed; potential mitigations"),
        "sandbagging": ("Research", "observed; potential mitigations"),
        "autonomous_replication": ("Research", "observed; potential mitigations"),
        "undermining_safeguards": ("Research", "observed; potential mitigations"),
        "rnd_automation": ("Tracked", "Capabilities + Safeguards Reports; SAG review"),
        "cyber_uplift": ("Tracked", "Capabilities + Safeguards Reports; SAG review"),
        "bio_uplift": ("Tracked", "Capabilities + Safeguards Reports; SAG review"),
    },
    "gating": frozenset({"Tracked"}),
    "artifacts": ("capabilities_report", "safeguards_report", "sag_review"),
}
ANTHROPIC_RSP_V3 = {
    "name": "Anthropic RSP v3.0",
    "table": {
        "long_range_autonomy": ("named risk", "affirmative case at threshold"),
        "sandbagging": ("eval-context gap", "measurement methodology; no trigger"),
        "undermining_safeguards": ("hardcoded prohibition", "refuses training / deploy"),
        "rnd_automation": ("AI R&D-4 threshold", "affirmative case required"),
        "cyber_uplift": ("ASL-3 trigger", "security + deployment mitigations"),
        "bio_uplift": ("ASL-3 trigger", "security + deployment mitigations"),
    },
    "gating": frozenset({
        "named risk",
        "hardcoded prohibition",
        "AI R&D-4 threshold",
        "ASL-3 trigger",
    }),
    "artifacts": ("capability_report", "affirmative_case", "sag_review"),
}
DEEPMIND_FSF_V3 = {
    "name": "DeepMind FSF v3",
    "table": {
        "long_range_autonomy": ("Tracked Capability Level",
                                "folded into ML R&D / Cyber CCL"),
        "sandbagging": ("deceptive alignment monitoring",
                        "automated instrumental-reasoning monitor"),
        "autonomous_replication": ("ML R&D CCL", "CCL threshold"),
        "undermining_safeguards": ("deceptive alignment monitoring",
                                   "automated monitor + red-team"),
        "rnd_automation": ("Tracked Capability Level",
                           "ML R&D autonomy level 1 (Apr 2026)"),
        "cyber_uplift": ("Cyber CCL", "security + deployment mitigations"),
        "bio_uplift": ("Bio CCL", "security + deployment mitigations"),
    },
    "gating": frozenset({
        "Tracked Capability Level",
        "ML R&D CCL",
        "Cyber CCL",
        "Bio CCL",
    }),
    "artifacts": ("fsf_risk_report", "internal_review_committee"),
}
POLICIES = (OPENAI_PF_V2, ANTHROPIC_RSP_V3, DEEPMIND_FSF_V3)
CCL_THRESHOLDS = {
    "ml_rnd": {
        "at_least": {"rnd_pipeline_automation_share": 1.0},
        "at_most": {"cost_ratio_vs_human": 1.0},
    },
    "cyber": {"at_least": {"cyber_uplift": 0.5}, "at_most": {}},
    "bio": {"at_least": {"bio_uplift": 0.5}, "at_most": {}},
}
CAPABILITY_METRICS = (
    "rnd_pipeline_automation_share",
    "cyber_uplift",
    "bio_uplift",
)
COST_METRICS = ("cost_ratio_vs_human",)


def classify(policy, capability, axes=None):
    """Как политика классифицирует возможность: dict с полем covered.

    classify(OPENAI_PF_V2, "rnd_automation")["classification"]   ->  "Tracked"
    classify(OPENAI_PF_V2, "long_range_autonomy")["classification"]
        ->  "Research"
    classify(ANTHROPIC_RSP_V3, "autonomous_replication")
        ->  covered False, classification None, action None

    Ключи результата: policy, capability, covered, classification, action.

    Первый шаг навыка чтения политики называется Locate: если возможность
    в документе не находится — документ её не покрывает. Поэтому пропуск
    возвращается явным covered=False, а не пустой строкой и не «—»: такой
    ответ невозможно перепутать с настоящей классификацией.

    Ловушка: опечатка в имени оси — это ValueError, а НЕ пробел политики.
    Иначе "long_range_autonmy" тихо прочиталось бы как «ни одна политика
    это не покрывает», и отчёт получился бы страшнее, чем реальность.
    """
    raise NotImplementedError


def is_gated(policy, capability, axes=None):
    """Обязана ли политика применить меры, а не просто наблюдать.

    is_gated(OPENAI_PF_V2, "rnd_automation")        ->  True    (Tracked)
    is_gated(OPENAI_PF_V2, "long_range_autonomy")   ->  False   (Research)
    is_gated(DEEPMIND_FSF_V3, "sandbagging")        ->  False   (мониторинг)

    Это второй шаг навыка (Classify) и вся суть деления Tracked/Research:
    формулировка «potential mitigations» — не обязательство. У каждой
    политики свой набор классификаций, которые реально гейтят, он лежит в
    ключе "gating".

    Возможность, которой в политике нет, не гейтит ничего: None не входит
    ни в один gating-набор. Это не мягкость реализации, это честное
    следствие — чего документ не назвал, того он и не требует.
    """
    raise NotImplementedError


def required_artifacts(policy, capability, axes=None):
    """Какие документы политика требует перед развёртыванием. Список.

    required_artifacts(OPENAI_PF_V2, "cyber_uplift")
        ->  ["capabilities_report", "safeguards_report", "sag_review"]
    required_artifacts(OPENAI_PF_V2, "long_range_autonomy")   ->  []

    Пустой список у Research-категории — не забывчивость таблицы, а
    буквальный смысл разряда: отчётов не требуется, потому что мер не
    обещано.

    Возвращается КОПИЯ кортежа из политики: список ушёл наружу, и вызывающий
    вправе его править, не ломая саму политику.
    """
    raise NotImplementedError


def compare(capability, policies=None, axes=None):
    """Side-by-side по одной возможности: dict имя политики -> разбор.

    compare("rnd_automation")           ->  три ключа, у всех gated True
    compare("long_range_autonomy")      ->  OpenAI gated False, двое True

    В результате ОБЯЗАНЫ присутствовать все политики, даже те, что эту ось
    не покрывают: сравнение по разным осям — не сравнение. У таких политик
    covered False, classification None, gated False, и это видно в таблице.

    К полям classify добавляется gated.
    """
    raise NotImplementedError


def coverage_report(policy, axes=None):
    """Полнота покрытия политики по осям: {"policy", "covered", "uncovered"}.

    coverage_report(OPENAI_PF_V2)["uncovered"]        ->  []
    coverage_report(ANTHROPIC_RSP_V3)["uncovered"]
        ->  ["autonomous_replication"]

    Оба списка отсортированы, и их объединение равно списку осей: ось не
    может ни потеряться, ни попасть в оба списка сразу. Именно это делает
    отчёт проверяемым — иначе «покрыто 6 из 7» пришлось бы принимать на веру.

    Зачем вообще отдельная функция: политику хвалят за то, что в ней
    написано, а рискует она тем, чего в ней нет. Ненайденную ось должно
    быть видно в отчёте так же явно, как найденную.
    """
    raise NotImplementedError


def gating_divergence(policies=None, axes=None):
    """Оси, по которым политики расходятся в ПОСЛЕДСТВИЯХ. Отсортировано.

    gating_divergence()  ->  ["autonomous_replication", "long_range_autonomy",
                              "undermining_safeguards"]

    Расхождение считается по is_gated, а не по названию разряда. Названия
    у трёх лабораторий разные всегда («Tracked», «ASL-3 trigger», «Cyber
    CCL») — это шум. Значение имеет одно: обязывает разряд к мерам или нет.

    Ось, где все три согласны (rnd_automation, cyber_uplift, bio_uplift, а
    также sandbagging — там все три ЕДИНОДУШНО не гейтят), в результат не
    попадает.
    """
    raise NotImplementedError


def ccl_reached(measurements, thresholds=None):
    """Домены FSF, чьи Critical Capability Levels пересечены. Отсортировано.

    ccl_reached({"cyber_uplift": 0.6})                        ->  ["cyber"]
    ccl_reached({"rnd_pipeline_automation_share": 1.0,
                 "cost_ratio_vs_human": 0.8})                 ->  ["ml_rnd"]
    ccl_reached({"rnd_pipeline_automation_share": 1.0,
                 "cost_ratio_vs_human": 3.0})                 ->  []

    Внутри домена условия соединены И, а не «сработало N из M»: ML R&D
    autonomy level 1 определён как «полностью автоматизирует пайплайн AI
    R&D ПРИ конкурентной цене». Автоматизация вдесятеро дороже человека —
    это не level 1.

    Порог "at_most" перевёрнут: чем ниже цена относительно человека, тем
    ближе домен к порогу. Не перепутай направление сравнения.

    Отсутствующее измерение: для "at_least" считается нулём, для "at_most" —
    бесконечностью. И то и другое означает «порог не пересечён»: чего не
    померили, то не засчитывается в пользу опасности.

    Домен без единого порога — ValueError. Пустое правило иначе пройдёт по
    all([]) == True и вернёт «порог пересечён» на любых данных.
    """
    raise NotImplementedError


def sandbagging_correction(measurements, rate):
    """Поправка измерений на strategic underperformance. Новый dict.

    sandbagging_correction({"cyber_uplift": 0.4}, 0.0)   ->  {"cyber_uplift": 0.4}
    sandbagging_correction({"cyber_uplift": 0.4}, 0.5)   ->  {"cyber_uplift": 0.8}
    sandbagging_correction({"cost_ratio_vs_human": 2.0}, 0.5)
        ->  {"cost_ratio_vs_human": 1.0}

    Sandbagging лежит в Research Categories у OpenAI и закрыт мониторингом
    deceptive alignment у DeepMind. Смысл поправки один: если модель
    занижает себя на оценках, ВСЕ пороги любой из трёх политик занижены
    вместе с ней. Рамка работает ровно настолько, насколько работает замер.

    Метрики возможностей делятся на value / (1 - rate) — наблюдаемое было
    занижено. Стоимостные умножаются на (1 - rate): дороговизна тоже
    показная. Направление берётся из CAPABILITY_METRICS и COST_METRICS,
    поэтому неизвестное имя метрики — ValueError, а не «оставим как есть»:
    молча непоправленная метрика — это ровно тот пробел, ради которого
    sandbagging и делают.

    rate вне [0, 1) — ValueError. При rate = 1 деление уходит в ноль, а
    отрицательная доля превратила бы поправку в успокоительное.
    """
    raise NotImplementedError

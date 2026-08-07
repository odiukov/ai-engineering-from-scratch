"""
OpenAI Preparedness Framework и DeepMind Frontier Safety Framework — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок читает три родственных документа рядом: OpenAI Preparedness Framework
v2 (апрель 2025), Anthropic RSP v3.0 и DeepMind Frontier Safety Framework v3
(сентябрь 2025 + Tracked Capability Levels от 17 апреля 2026). Здесь мы
собираем руками ту самую «decision-table diff»-машинку, о которой говорит
раздел «Use It», и добавляем измерительную половину DeepMind.

Что чему соответствует в документах:

    CAPABILITY_AXES        <-  общий список осей, по которым сравниваем
    classify               <-  «Locate»: находится ли возможность в политике
    is_gated               <-  «Classify»: Tracked (гейтит) или Research (нет)
    required_artifacts     <-  Capabilities/Safeguards Reports, SAG review
    compare                <-  side-by-side по ОДИНАКОВЫМ осям
    coverage_report        <-  пропуск оси обязан быть виден, а не молчать
    gating_divergence      <-  где три политики расходятся в последствиях
    ccl_reached            <-  Critical Capability Levels по доменам FSF v3
    sandbagging_correction <-  поправка на strategic underperformance

Таблицы — учебные дистилляции, а не цитаты. Настоящее чтение политики
делается по исходным PDF; здесь важна механика сравнения, а не буква.
Категории обозначены метками (cyber, bio, autonomy) — никаких примеров
вредного содержимого в упражнении нет и быть не должно.

Ни сети, ни времени, ни глобального random: решение о допуске обязано быть
воспроизводимым по входу, иначе его нельзя ни оспорить, ни перепроверить.
"""

# Общие оси сравнения. Пока политики не разложены по ОДНОМУ списку осей,
# «сравнение» сводится к пересказу трёх оглавлений.
CAPABILITY_AXES = (
    "long_range_autonomy",
    "sandbagging",
    "autonomous_replication",
    "undermining_safeguards",
    "rnd_automation",
    "cyber_uplift",
    "bio_uplift",
)

# OpenAI PF v2: главное деление — Tracked (обязательные меры) против
# Research (наблюдаем, мер не обещаем). В Research живут Long-range
# Autonomy, Sandbagging, Autonomous Replication and Adaptation и
# Undermining Safeguards.
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

# Anthropic RSP v3.0. Оси autonomous_replication в таблице НЕТ намеренно:
# документ не называет автономную репликацию отдельно, по разбору урока она
# уходит под AI R&D-4. Пропуск оставлен пропуском, чтобы coverage_report
# показал его вслух, а не заретушировал похожей формулировкой.
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

# DeepMind FSF v3. Autonomy не отдельный домен: она свёрнута в ML R&D и
# Cyber. Deceptive alignment закрыт автоматическим мониторингом — а
# мониторинг не гейтит развёртывание, и сама DeepMind пишет, что долго
# его не хватит.
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

# Critical Capability Levels по доменам FSF v3. ML R&D autonomy level 1
# определён конъюнкцией: пайплайн автоматизирован ПОЛНОСТЬЮ и при этом
# по цене, конкурентной с «человек + инструменты». Порог по цене
# перевёрнут: чем НИЖЕ отношение к стоимости человека, тем опаснее.
CCL_THRESHOLDS = {
    "ml_rnd": {
        "at_least": {"rnd_pipeline_automation_share": 1.0},
        "at_most": {"cost_ratio_vs_human": 1.0},
    },
    "cyber": {"at_least": {"cyber_uplift": 0.5}, "at_most": {}},
    "bio": {"at_least": {"bio_uplift": 0.5}, "at_most": {}},
}

# Направление каждой метрики. Без явного реестра поправка на sandbagging
# двинет перевёрнутую метрику не в ту сторону и «успокоит» отчёт.
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
    ax = CAPABILITY_AXES if axes is None else axes
    if capability not in ax:
        raise ValueError(f"unknown capability axis: {capability!r}")
    # .get, а не [], потому что отсутствие записи — законный ответ «не покрыто»
    entry = policy["table"].get(capability)
    return {
        "policy": policy["name"],
        "capability": capability,
        "covered": entry is not None,
        "classification": entry[0] if entry is not None else None,
        "action": entry[1] if entry is not None else None,
    }


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
    return classify(policy, capability, axes)["classification"] in policy["gating"]


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
    if not is_gated(policy, capability, axes):
        return []
    return list(policy["artifacts"])


def compare(capability, policies=None, axes=None):
    """Side-by-side по одной возможности: dict имя политики -> разбор.

    compare("rnd_automation")           ->  три ключа, у всех gated True
    compare("long_range_autonomy")      ->  OpenAI gated False, двое True

    В результате ОБЯЗАНЫ присутствовать все политики, даже те, что эту ось
    не покрывают: сравнение по разным осям — не сравнение. У таких политик
    covered False, classification None, gated False, и это видно в таблице.

    К полям classify добавляется gated.
    """
    pols = POLICIES if policies is None else policies
    out = {}
    for p in pols:
        row = classify(p, capability, axes)
        row["gated"] = is_gated(p, capability, axes)
        out[p["name"]] = row
    return out


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
    ax = CAPABILITY_AXES if axes is None else axes
    covered, uncovered = [], []
    for cap in ax:
        # прогоняем через classify, а не через "in table": так проверка
        # опечаток и определение покрытия остаются в одном месте
        bucket = covered if classify(policy, cap, ax)["covered"] else uncovered
        bucket.append(cap)
    return {
        "policy": policy["name"],
        "covered": sorted(covered),
        "uncovered": sorted(uncovered),
    }


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
    pols = POLICIES if policies is None else policies
    ax = CAPABILITY_AXES if axes is None else axes
    out = []
    for cap in ax:
        # множество из двух элементов = кто-то гейтит, кто-то нет
        if len({is_gated(p, cap, ax) for p in pols}) > 1:
            out.append(cap)
    return sorted(out)


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
    thr = CCL_THRESHOLDS if thresholds is None else thresholds
    reached = []
    for domain, rule in thr.items():
        at_least = rule.get("at_least", {})
        at_most = rule.get("at_most", {})
        if not at_least and not at_most:
            raise ValueError(f"domain without thresholds: {domain!r}")
        ok = all(
            measurements.get(m, 0.0) >= limit for m, limit in at_least.items()
        )
        ok = ok and all(
            measurements.get(m, float("inf")) <= limit
            for m, limit in at_most.items()
        )
        if ok:
            reached.append(domain)
    return sorted(reached)


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
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"sandbagging rate must be in [0, 1): {rate!r}")
    scale = 1.0 - rate
    out = {}
    for name, value in measurements.items():
        if name in CAPABILITY_METRICS:
            out[name] = value / scale
        elif name in COST_METRICS:
            out[name] = value * scale
        else:
            raise ValueError(f"unknown metric: {name!r}")
    return out

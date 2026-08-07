"""
Anthropic Responsible Scaling Policy v3.0 — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Собираем руками решающую машинку, которая внутри RSP описана прозой. Что
чему соответствует в документе (RSP v3.0, вступил в силу 24 февраля 2026):

    ASL_THRESHOLDS          <-  capability thresholds по уровням ASL
    capability_level         <-  «какой ASL требуют измерения этой модели»
    required_safeguards      <-  ступенчатая (кумулятивная) лестница мер
    missing_safeguards       <-  чего не хватает до допуска
    deployment_decision      <-  deployment gate: можно ли выкатывать
    unilateral_commitments   <-  two-tier: колонка «Anthropic unilateral»
    diff_policies            <-  диф v2 -> v3.0: что добавили, убрали, перенесли
    affirmative_case_sections<-  разделы affirmative case для AI R&D-4
    policy_score             <-  рубрика SaferAI: v2 = 2.2, v3.0 = 1.9

Числа порогов иллюстративные: настоящий RSP формулирует пороги качественно,
и это как раз один из пунктов, за который SaferAI понизила оценку. Мы делаем
их числами, чтобы логика допуска стала проверяемой.

Времени по часам здесь нет вовсе, случайности тоже: решение о развёртывании
обязано быть воспроизводимым по входу, иначе его нельзя ни оспорить, ни
проверить постфактум.
"""

# Лестница уровней. Порядок = возрастание требований.
ASL_LEVELS = ("ASL-2", "ASL-3", "ASL-4")

# Пороги по уровням. ASL-2 — базовый, порогов не имеет: это то, где модель
# находится, пока ничего не сработало.
ASL_THRESHOLDS = {
    "ASL-3": {
        "rd_automation_share": 0.35,
        "metr_horizon_hours": 20.0,
        "cyber_uplift": 0.30,
    },
    "ASL-4": {
        "rd_automation_share": 0.60,
        "metr_horizon_hours": 40.0,
        "cyber_uplift": 0.55,
    },
}

# Сколько порогов уровня должно сработать, чтобы уровень считался достигнутым.
# Одно измерение — шум; два независимых — сигнал.
TRIGGERS_REQUIRED = 2

# Меры по уровням. Лестница КУМУЛЯТИВНА: ASL-3 не отменяет мер ASL-2.
SAFEGUARD_SCHEDULE = {
    "ASL-2": ("model_card", "usage_policy"),
    "ASL-3": ("weights_security", "deployment_classifier", "red_team_signoff"),
    "ASL-4": ("rand_sl4_security", "affirmative_case", "external_review"),
}

# Два уровня обязательств из v3.0. "unilateral" — то, что лаборатория делает
# независимо от остальных. "industry" — то, что она РЕКОМЕНДУЕТ отрасли.
# Обязательство в колонке industry — не обещание, а пожелание.
TIERS = ("unilateral", "industry")

# Иллюстративные срезы двух редакций политики.
RSP_V2 = {
    "pause_on_threshold": "unilateral",
    "weights_security": "unilateral",
    "deployment_classifier": "unilateral",
    "red_team_signoff": "unilateral",
    "rand_sl4_security": "unilateral",
}

RSP_V3 = {
    "weights_security": "unilateral",
    "deployment_classifier": "unilateral",
    "red_team_signoff": "unilateral",
    "rand_sl4_security": "industry",
    "affirmative_case": "unilateral",
    "frontier_safety_roadmap": "unilateral",
    "risk_report": "unilateral",
}

# Разделы affirmative case, которые требует v3.0 при пересечении AI R&D-4.
AFFIRMATIVE_CASE_SECTIONS = (
    "capability_inventory",
    "misalignment_risk_analysis",
    "evaluation_context_gap",
    "mitigation_design",
    "residual_risk",
    "safety_advisory_group_signoff",
)

# Порог, выше которого к affirmative case добавляется отдельный раздел про
# поправку на eval-context gaming (урок 1 фазы).
GAMING_SECTION_THRESHOLD = 0.2
GAMING_SECTION = "gaming_adjusted_capability_estimate"

# Рубрика SaferAI. Веса подобраны так, чтобы воспроизвести публичные оценки:
# v2 -> 2.2, v3.0 -> 1.9.
SAFERAI_BASELINE = 1.0
SAFERAI_RUBRIC = {
    "quantitative_thresholds": 0.3,
    "pause_commitment": 0.3,
    "declared_cadence": 0.3,
    "published_risk_reports": 0.3,
    "frontier_safety_roadmap": 0.3,
    "independent_external_review": 0.4,
}

# Границы категорий SaferAI: ниже 2.0 — «weak», ниже 3.0 — «moderate».
SAFERAI_BANDS = (("weak", 2.0), ("moderate", 3.0), ("strong", float("inf")))


def capability_level(measurements, thresholds=None, triggers_required=None):
    """Требуемый уровень ASL по измерениям модели.

    Уровень достигнут, если сработало не меньше triggers_required порогов
    этого уровня. Возвращается САМЫЙ ВЫСОКИЙ достигнутый уровень.

    capability_level({"rd_automation_share": 0.30,
                      "metr_horizon_hours": 14.0,
                      "cyber_uplift": 0.10})            ->  "ASL-2"
    capability_level({"rd_automation_share": 0.40,
                      "metr_horizon_hours": 22.0,
                      "cyber_uplift": 0.10})            ->  "ASL-3"
    capability_level({"rd_automation_share": 0.70,
                      "metr_horizon_hours": 48.0,
                      "cyber_uplift": 0.60})            ->  "ASL-4"

    Первый пример — Claude Opus 4.6 по заявлению из анонса v3.0: порог
    AI R&D-4 не пересечён.

    Свойство, которое обязано выполняться: рост любого измерения НИКОГДА не
    понижает уровень. Если в твоей реализации это не так — там ошибка, а не
    хитрая политика. Монотонность — то, ради чего лестница вообще существует.

    Отсутствующее измерение считается нулём: чего не померили, того нет.
    Это осознанно оптимистично, и в настоящем RSP такой пробел закрывается
    требованием проводить оценку, а не гадать.
    """
    thr = ASL_THRESHOLDS if thresholds is None else thresholds
    need = TRIGGERS_REQUIRED if triggers_required is None else triggers_required
    level = ASL_LEVELS[0]
    # идём снизу вверх: уровень поднимается только пока пороги срабатывают,
    # поэтому результат не зависит от порядка ключей в thr
    for candidate in ASL_LEVELS[1:]:
        limits = thr.get(candidate, {})
        fired = sum(
            1 for name, limit in limits.items()
            if measurements.get(name, 0.0) >= limit
        )
        if fired >= need:
            level = candidate
    return level


def required_safeguards(level, schedule=None):
    """Кумулятивный список мер для уровня: свои плюс все нижние.

    required_safeguards("ASL-2")  ->  ["model_card", "usage_policy"]
    required_safeguards("ASL-3")  ->  ["model_card", "usage_policy",
                                       "weights_security",
                                       "deployment_classifier",
                                       "red_team_signoff"]

    Кумулятивность — не деталь оформления. Модель на ASL-3 не перестаёт
    нуждаться в model card; лестница добавляет требования, а не заменяет их.

    Неизвестный уровень — ValueError. Строка "ASL-5", которой ещё нет в
    лестнице, иначе тихо вернёт пустой список, то есть «мер не требуется» —
    ровно противоположное тому, что она означает.
    """
    sched = SAFEGUARD_SCHEDULE if schedule is None else schedule
    if level not in ASL_LEVELS:
        raise ValueError(f"unknown ASL level: {level!r}")
    out = []
    for lvl in ASL_LEVELS:
        out.extend(sched.get(lvl, ()))
        if lvl == level:
            break
    return out


def missing_safeguards(level, in_place, schedule=None):
    """Чего не хватает до уровня. Порядок — как в лестнице.

    missing_safeguards("ASL-2", ["model_card", "usage_policy"])  ->  []
    missing_safeguards("ASL-3", ["model_card", "usage_policy"])
        ->  ["weights_security", "deployment_classifier", "red_team_signoff"]

    in_place может содержать лишние меры — это не ошибка, просто они не
    относятся к уровню. Отчёт говорит о нехватке, а не о соответствии.
    """
    have = set(in_place)
    return [s for s in required_safeguards(level, schedule) if s not in have]


def deployment_decision(measurements, in_place, thresholds=None, schedule=None):
    """Решение о допуске к развёртыванию.

    Возвращает {"level", "allowed", "missing", "reason"}.

    deployment_decision({"rd_automation_share": 0.30},
                        ["model_card", "usage_policy"])
        ->  {"level": "ASL-2", "allowed": True, "missing": [],
             "reason": "ASL-2: all required safeguards in place"}

    deployment_decision({"rd_automation_share": 0.40,
                         "metr_horizon_hours": 22.0},
                        ["model_card", "usage_policy"])
        ->  allowed False, missing содержит "weights_security"

    Главное свойство: тот же самый набор мер, что допускал модель на ASL-2,
    НЕ допускает её на ASL-3. Рост возможностей поднимает планку, а не
    оставляет её на месте. Развёртывание без мер уровня невозможно —
    в этом и состоит смысл gate.

    reason обязан называть уровень и недостающие меры: решение, которое
    нельзя прочитать и оспорить, ничем не отличается от отсутствия решения.
    """
    level = capability_level(measurements, thresholds)
    missing = missing_safeguards(level, in_place, schedule)
    if missing:
        reason = f"{level}: missing " + ",".join(missing)
    else:
        reason = f"{level}: all required safeguards in place"
    return {
        "level": level,
        "allowed": not missing,
        "missing": missing,
        "reason": reason,
    }


def unilateral_commitments(policy):
    """Отсортированные обязательства из колонки «unilateral».

    unilateral_commitments({"a": "unilateral", "b": "industry"})  ->  ["a"]

    Ради чего это отдельная функция: в v3.0 появилась вторая колонка, и
    читатель обязан смотреть, в какой из них живёт каждая мера. Мера в
    колонке «industry-wide recommendation» — это не обещание лаборатории,
    а её пожелание отрасли. RAND SL-4 в v3.0 живёт именно там.

    Неизвестный tier — ValueError: опечатка в колонке молча превратила бы
    обязательство в рекомендацию или наоборот.
    """
    out = []
    for name, tier in policy.items():
        if tier not in TIERS:
            raise ValueError(f"unknown tier: {tier!r}")
        if tier == "unilateral":
            out.append(name)
    return sorted(out)


def diff_policies(old, new):
    """Диф двух редакций: что добавили, что убрали, что перенесли между колонок.

    Возвращает {"added", "removed", "retiered"}; retiered — список кортежей
    (имя, старый tier, новый tier). Все три отсортированы по имени.

    diff_policies({"p": "unilateral"}, {"p": "industry", "q": "unilateral"})
        ->  {"added": ["q"], "removed": [], "retiered": [("p", "unilateral",
                                                          "industry")]}

    Перенос в колонку industry — это НЕ «removed». Обязательство формально
    осталось в документе, но перестало быть обязательством лаборатории.
    Отдельная категория нужна ровно затем, чтобы такой перенос не
    маскировался под «ничего не изменилось».

    На v2 -> v3.0 в removed окажется pause_on_threshold, а в retiered —
    rand_sl4_security. Это и есть та регрессия, которую назвала SaferAI.
    """
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "retiered": sorted(
            (name, old[name], new[name])
            for name in set(old) & set(new)
            if old[name] != new[name]
        ),
    }


def affirmative_case_sections(level, gaming_rate=0.0):
    """Разделы affirmative case. Пустой список, если уровень его не требует.

    affirmative_case_sections("ASL-2")             ->  []
    affirmative_case_sections("ASL-4")             ->  шесть разделов
    affirmative_case_sections("ASL-4", 0.28)       ->  семь разделов

    Affirmative case требуется на ASL-4: v3.0 заменила им обещание паузы из
    редакции 2023 года. Вместо «остановимся» — «опубликуем аргумент, почему
    можно продолжать».

    Седьмой раздел добавляется, когда доля eval-context gaming выше
    GAMING_SECTION_THRESHOLD: измеренные возможности в этом случае занижены,
    и порог мог быть пересечён раньше, чем показали замеры.

    Отрицательный gaming_rate — ValueError. Доля не бывает отрицательной, а
    молча принятая -1 отключила бы проверку целиком.
    """
    if gaming_rate < 0:
        raise ValueError("gaming_rate must be non-negative")
    if level != "ASL-4":
        return []
    sections = list(AFFIRMATIVE_CASE_SECTIONS)
    if gaming_rate > GAMING_SECTION_THRESHOLD:
        sections.append(GAMING_SECTION)
    return sections


def policy_score(satisfied, rubric=None, baseline=None):
    """Оценка политики по рубрике SaferAI: {"score", "band"}.

    policy_score({"quantitative_thresholds", "pause_commitment",
                  "declared_cadence", "published_risk_reports"})
        ->  {"score": 2.2, "band": "moderate"}
    policy_score({"declared_cadence", "published_risk_reports",
                  "frontier_safety_roadmap"})
        ->  {"score": 1.9, "band": "weak"}

    Первый набор — RSP v2, второй — v3.0. Публичные оценки SaferAI: 2.2 и
    1.9. Граница ровно 2.0 относится к «moderate», сравнение строгое снизу,
    поэтому падение с 2.2 до 1.9 — не косметика, а смена категории.

    Обрати внимание, за счёт чего просела оценка: v3.0 добавила Frontier
    Safety Roadmap (плюс 0.3), но потеряла количественные пороги и обещание
    паузы (минус 0.6). Документ стал полнее и при этом слабее.

    Незнакомый критерий — ValueError. Опечатка в названии иначе просто не
    добавила бы вес, и политика тихо получила бы оценку ниже заслуженной —
    ошибка, которую в отчёте не видно.

    Сумму обязательно округлять: 1.0+0.3+0.3+0.3+0.3 в двоичной плавающей
    точке даёт 2.2000000000000006, и сравнение с публичной оценкой
    развалилось бы на ровном месте.
    """
    rub = SAFERAI_RUBRIC if rubric is None else rubric
    base = SAFERAI_BASELINE if baseline is None else baseline
    total = base
    for name in satisfied:
        if name not in rub:
            raise ValueError(f"unknown rubric criterion: {name!r}")
        total += rub[name]
    score = round(total, 2)
    band = next(name for name, upper in SAFERAI_BANDS if score < upper)
    return {"score": score, "band": band}

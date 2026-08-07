"""
Anthropic Responsible Scaling Policy v3.0

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l19-anthropic-rsp
Разбор:  /check-code p15-l19-anthropic-rsp
"""

ASL_LEVELS = ("ASL-2", "ASL-3", "ASL-4")
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
TRIGGERS_REQUIRED = 2
SAFEGUARD_SCHEDULE = {
    "ASL-2": ("model_card", "usage_policy"),
    "ASL-3": ("weights_security", "deployment_classifier", "red_team_signoff"),
    "ASL-4": ("rand_sl4_security", "affirmative_case", "external_review"),
}
TIERS = ("unilateral", "industry")
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
AFFIRMATIVE_CASE_SECTIONS = (
    "capability_inventory",
    "misalignment_risk_analysis",
    "evaluation_context_gap",
    "mitigation_design",
    "residual_risk",
    "safety_advisory_group_signoff",
)
GAMING_SECTION_THRESHOLD = 0.2
GAMING_SECTION = "gaming_adjusted_capability_estimate"
SAFERAI_BASELINE = 1.0
SAFERAI_RUBRIC = {
    "quantitative_thresholds": 0.3,
    "pause_commitment": 0.3,
    "declared_cadence": 0.3,
    "published_risk_reports": 0.3,
    "frontier_safety_roadmap": 0.3,
    "independent_external_review": 0.4,
}
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
    raise NotImplementedError


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
    raise NotImplementedError


def missing_safeguards(level, in_place, schedule=None):
    """Чего не хватает до уровня. Порядок — как в лестнице.

    missing_safeguards("ASL-2", ["model_card", "usage_policy"])  ->  []
    missing_safeguards("ASL-3", ["model_card", "usage_policy"])
        ->  ["weights_security", "deployment_classifier", "red_team_signoff"]

    in_place может содержать лишние меры — это не ошибка, просто они не
    относятся к уровню. Отчёт говорит о нехватке, а не о соответствии.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError

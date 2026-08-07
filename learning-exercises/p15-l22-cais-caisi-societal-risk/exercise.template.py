"""
CAIS, CAISI и риск общественного масштаба

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l22-cais-caisi-societal-risk
Разбор:  /check-code p15-l22-cais-caisi-societal-risk
"""

from urllib.parse import urlparse

FOUR_RISKS = (
    "malicious_use",
    "ai_races",
    "organizational_risks",
    "rogue_ais",
)
ORG_LEVERS = (
    "safety_culture",
    "audit_rigor",
    "multi_layer_defenses",
    "information_security",
)
HARM_LABELS = ("cbrn", "cyber", "bio", "disinformation", "autonomy")
DEPLOYMENT_FEATURES = {
    "public_facing": False,
    "harmful_capability_labels": (),
    "competitive_pressure": False,
    "independent_audit": True,
    "multi_layer_defense": True,
    "information_security": True,
    "agent_autonomy_hours": 0.0,
}
ROGUE_AUTONOMY_HOURS = 4.0
MITIGATIONS = {
    "malicious_use": (
        "constitutional hardcoded prohibitions (Lesson 17)",
        "Llama Guard input/output classifier (Lesson 18)",
        "tool allowlist per task (Lessons 10, 11)",
    ),
    "ai_races": (
        "scaling policy with standing Risk Reports (Lessons 19, 20)",
        "public Frontier Safety Roadmap with declared cadence",
        "external capability evaluation by METR / CAISI (Lesson 21)",
    ),
    "organizational_risks": (
        "safety culture: escalation paths without career cost",
        "independent audit on declared cadence",
        "multi-layered defenses (Lessons 10, 13, 14, 17, 18)",
        "information security per RAND SL-4 (Lesson 19)",
    ),
    "rogue_ais": (
        "kill switches and canary tokens (Lesson 14)",
        "propose-then-commit HITL (Lesson 15)",
        "deceptive-alignment monitoring (Lesson 20)",
        "durable checkpoints and rollback (Lesson 16)",
    ),
}
CRITICAL_FLOOR = 0.4
RISK_BANDS = (
    ("critical", 0.4),
    ("weak", 0.6),
    ("adequate", 0.8),
    ("strong", float("inf")),
)
SOCIETAL_STACK = (
    "lab_scaling_policy",
    "external_evaluation",
    "civil_society_tracking",
    "government_baseline",
    "practitioner_controls",
)
ORG_HOSTS = {"safe.ai": "CAIS", "nist.gov": "CAISI"}
SB53_AUTONOMY_HOURS = 8.0
SB53_REPORT_HOURS = 24.0
SB53_UNCONDITIONAL = ("whistleblower_protection",)
SB53_THRESHOLD_OBLIGATIONS = (
    "capability_threshold_disclosure",
    "incident_reporting",
)


def tag_risks(deployment):
    """Разметка развёртывания по четырём категориям CAIS. Отсортировано.

    tag_risks({})  ->  []
    tag_risks({"competitive_pressure": True})  ->  ["ai_races"]
    tag_risks({"public_facing": True,
               "harmful_capability_labels": ["cyber"],
               "independent_audit": False,
               "agent_autonomy_hours": 48.0})
        ->  ["malicious_use", "organizational_risks", "rogue_ais"]

    Правила: malicious_use — есть метки опасных возможностей И публичный
    доступ; ai_races — давление конкуренции; organizational_risks — не
    хватает ХОТЬ ОДНОГО подрычага (аудит, слои защиты, инфобез); rogue_ais —
    автономия от ROGUE_AUTONOMY_HOURS часов.

    Незнакомый признак — ValueError. Опечатка "publicfacing" иначе тихо
    уйдёт в умолчание False, и развёртывание разметится как внутреннее.
    То же с незнакомой меткой: список меток закрыт (HARM_LABELS), потому
    что рамка работает с категориями, а не со свободным текстом.

    Organizational_risks срабатывает по ИЛИ, а не по «большинству»: CAIS
    называет эту категорию отдельно именно потому, что одного проваленного
    подрычага достаточно.
    """
    raise NotImplementedError


def mitigation_checklist(deployment):
    """Меры под размеченные категории: dict категория -> список мер.

    mitigation_checklist({})  ->  {}
    mitigation_checklist({"competitive_pressure": True})
        ->  {"ai_races": [три меры из MITIGATIONS]}

    Незатронутая категория в чеклист не попадает вовсе: чеклист, где всё
    перечислено всегда, читается как «сделайте всё» и не читается никак.

    Списки — копии кортежей из MITIGATIONS. Чеклист уходит наружу, его
    правят и вычёркивают; сам справочник мер от этого меняться не должен.
    """
    raise NotImplementedError


def aggregate_risk(scores, weights=None, floor=None):
    """Агрегат показателей риска, который не прячет провал за средним.

    Ключи: mean, worst, score, critical, band.
    Показатели в шкале 0..1, где 1 — хорошо, 0 — провал.

    aggregate_risk({"a": 0.9, "b": 0.9})
        ->  mean 0.9, worst 0.9, critical [], score 0.9, band "strong"
    aggregate_risk({"a": 0.95, "b": 0.95, "c": 0.95, "d": 0.1})
        ->  mean 0.7375, worst 0.1, critical ["d"], score 0.1,
            band "critical"

    Главное свойство: если хоть один показатель ниже floor, итоговый score
    равен худшему, а не среднему. Иначе провал по инфобезу «лечится»
    добавлением ещё десяти хороших метрик — арифметика сойдётся, а риск
    останется. Средний mean всё равно возвращается, чтобы разницу было
    видно, а не только итог.

    weights задаёт веса; имя, которого нет в scores, — ValueError, как и
    неположительный вес. Вес по умолчанию 1.0.

    Пустой вход — ValueError. Агрегат по нулю показателей вернул бы
    единицу или ноль, и оба ответа означали бы «мы посчитали».

    Показатель вне [0, 1] — ValueError: шкала должна быть одна, иначе
    сравнивать нечего.
    """
    raise NotImplementedError


def stack_assessment(layer_strengths):
    """Оценка стека защиты на общественном слое.

    Ключи: missing, complete, aggregate.

    stack_assessment({layer: 0.9 for layer in SOCIETAL_STACK})
        ->  missing [], complete True, aggregate band "strong"
    stack_assessment({"lab_scaling_policy": 1.0})
        ->  missing — остальные четыре слоя, complete False,
            aggregate score 0.0, band "critical"

    Второй пример — финальный вывод фазы одной строкой: идеальный
    единственный слой не спасает стек с четырьмя дырами. Отсутствующий слой
    входит в агрегат как 0.0, то есть как критический показатель, и по
    правилу aggregate_risk забирает итог себе.

    Обратный случай тоже полезен: полный стек, где один слой слаб, даёт
    complete True и при этом критический агрегат. Полнота и достаточность —
    разные вопросы, и функция отвечает на оба по отдельности.

    Незнакомое имя слоя — ValueError: слой, которого нет в стеке, нельзя ни
    зачесть, ни пропустить осмысленно.
    """
    raise NotImplementedError


def identify_organization(url):
    """CAIS или CAISI — по хосту, а не по буквам в тексте.

    identify_organization("https://safe.ai/ai-risk")     ->  "CAIS"
    identify_organization("https://www.nist.gov/caisi")  ->  "CAISI"
    identify_organization("https://example.com/caisi")   ->  ValueError

    CAIS — некоммерческая исследовательская организация (рамка четырёх
    рисков, заявление 2023 года). CAISI — центр внутри NIST (добровольные
    соглашения с лабораториями, несекретные оценки возможностей). Миссии не
    пересекаются, акронимы почти совпадают.

    Третий пример — вся ловушка: акроним в пути ничего не доказывает.
    Незнакомый хост — ValueError, а не догадка по буквам.

    URL без схемы — тоже ValueError: у "safe.ai/ai-risk" хост не разобран,
    и подставлять его вручную значит гадать. Регистр хоста не важен.
    """
    raise NotImplementedError


def sb53_obligations(deployment):
    """Обязательства по California SB-53 для развёртывания. Отсортировано.

    sb53_obligations({})  ->  ["whistleblower_protection"]
    sb53_obligations({"agent_autonomy_hours": 12.0})
        ->  ["capability_threshold_disclosure", "incident_reporting",
             "whistleblower_protection"]
    sb53_obligations({"harmful_capability_labels": ["cbrn"]})
        ->  те же три

    Защита информантов безусловна: она не зависит ни от возможностей
    модели, ни от порогов. Это отдельный пункт билля, и путать его с
    пороговыми обязательствами нельзя — иначе получится, что сотрудник
    лаборатории ниже порога защиты не имеет.

    Порог включается либо автономией от SB53_AUTONOMY_HOURS часов, либо
    наличием любой метки опасных возможностей. Проверка признаков — та же,
    что у tag_risks, поэтому опечатка в признаке остаётся ValueError.
    """
    raise NotImplementedError


def incident_report_status(incident_at, now, deadline_hours=None):
    """Срок отчёта об инциденте: {"deadline_at", "hours_remaining", "overdue"}.

    Время — в часах по любой монотонной шкале, лишь бы одной и той же.

    incident_report_status(100.0, 110.0)  ->  deadline_at 124.0,
                                              hours_remaining 14.0,
                                              overdue False
    incident_report_status(100.0, 124.0)  ->  hours_remaining 0.0,
                                              overdue False
    incident_report_status(100.0, 130.0)  ->  hours_remaining -6.0,
                                              overdue True

    now приходит ПАРАМЕТРОМ, а не берётся из часов. Отчёт о просрочке,
    который меняется от момента запуска, нельзя ни проверить, ни приложить
    к делу; а инциденты разбирают именно постфактум.

    Ровно на границе просрочки нет: срок «в течение 24 часов» включает
    двадцать четвёртый час. Сдвиг этой границы на секунду — самый дешёвый
    способ превратить соблюдение в нарушение и обратно.

    now раньше инцидента — ValueError: отчёт не может быть готов раньше
    того, о чём он. Неположительное окно — тоже ValueError.
    """
    raise NotImplementedError

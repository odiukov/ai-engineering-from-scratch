"""
Агент-ревьюер: строитель и приёмщик — разные роли — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Гейт (Phase 14 · 38) отвечает на детерминированные вопросы: прогнали ли
acceptance, held ли scope. Ревьюер отвечает на качественные: ту ли задачу
решили, записаны ли допущения, можно ли с этого места продолжить. Оба нужны.

Соответствие настоящей системе:

    reviewer_view              <-  read-only доступ ревьюера к артефактам
    score_rubric               <-  reviewer_checklist.md, пять измерений по 0..2
    verdict_from_scores        <-  pass / soft_fail / hard_fail
    review_report              <-  review_report.json + порог уверенности
    re_review                  <-  повторный заход после правок строителя
    consistent_pairwise_winner <-  защита от position bias у LLM-судьи
    calibration_agreement      <-  калибровочный набор, без которого рубрику
                                   нельзя выкатывать

Ни одного вызова LLM: оценки здесь считаются по правилам, чтобы тест был
воспроизводим. В проде каждая функция score_* — это промпт, но интерфейс и
пороги остаются ровно те же.
"""

import copy
import fnmatch

# Пять измерений рубрики. Порядок фиксирован: он же порядок колонок в отчёте.
DIMENSIONS = (
    "problem_fit",
    "scope_discipline",
    "assumptions",
    "verification_quality",
    "handoff_readiness",
)

MAX_PER_DIMENSION = 2

# Пороги вердикта. Ниже 7 из 10 — правки, ниже 5 или любой ноль — стоп.
SOFT_FAIL_BELOW = 7
HARD_FAIL_BELOW = 5

# Ниже этого ревьюер отказывается выносить вердикт: отчёт с низким сигналом
# хуже отсутствия отчёта, потому что его прочитают и поверят.
CONFIDENCE_FLOOR = 0.6

# Уверенность по наличию улик. Числа условные, важна граница относительно
# CONFIDENCE_FLOOR.
CONFIDENCE_WITH_EVIDENCE = 0.9
CONFIDENCE_ASSUMED = 0.8
CONFIDENCE_BLIND = 0.3

# Согласие с историческим набором, ниже которого рубрику не выкатывают.
CALIBRATION_FLOOR = 0.8


def reviewer_view(artifacts):
    """Копия артефактов строителя для ревьюера: смотреть можно, править нельзя.

    view = reviewer_view({"diff": {"touched_files": ["app/a.py"]}})
    view["diff"]["touched_files"].append("app/b.py")   # исходник не изменится

    Ревьюер читает дифф, состояние, журнал и вердикт — и пишет только отчёт.
    Если он умеет править дифф, роли схлопываются и весь зазор между
    строителем и приёмщиком исчезает.

    Ловушка: dict(artifacts) копирует только верхний уровень, вложенные
    списки остались бы общими.
    """
    return copy.deepcopy(artifacts)


def score_rubric(inputs):
    """Оценить пять измерений. Вернуть {измерение: {score, confidence, reason,
    evidence}}.

    score_rubric(clean_inputs)["problem_fit"]["score"]  ->  2
    score_rubric(clean_inputs)["problem_fit"]["evidence"]
        ->  кортеж из тронутых файлов и добавленных тестов

    score — 0, 1 или 2. evidence — отпечаток ТВЁРДЫХ улик (какие файлы
    тронуты, какие команды прогнаны), а не пересказ строителя. Отпечаток
    нужен повторному заходу: если строитель дописал себе галочку
    "поведение покрыто", а файлы не тронул, отпечаток не сдвинется.

    Правила:
      problem_fit          2 — покрыты все требуемые поведения, 0 — ни одного;
      scope_discipline     2 — правок вне контракта нет, 0 — есть незаявленные;
      assumptions          2 — все допущения куда-то записаны, 0 — ни одно;
      verification_quality 2 — все acceptance прогнаны и зелены, 0 — ни одна
                               не прогнана либо в журнале есть пустой exit_code;
      handoff_readiness    2 — есть next_action и чистое состояние,
                               0 — next_action нет.
    """
    task = inputs.get("task", {})
    diff = inputs.get("diff", {})
    scope = inputs.get("scope", {})
    assumptions = inputs.get("assumptions", [])
    feedback = inputs.get("feedback", [])
    handoff = inputs.get("handoff", {})

    scores = {}

    # -- problem_fit -------------------------------------------------------
    required = set(task.get("required_behaviors", []))
    covered = required & set(diff.get("behaviors_covered", []))
    if not required:
        fit, reason = 2, "требуемые поведения не сформулированы, судить не по чему"
    elif covered == required:
        fit, reason = 2, "покрыты все требуемые поведения"
    elif covered:
        fit, reason = 1, f"покрыто {len(covered)} из {len(required)}"
    else:
        fit, reason = 0, "ни одно требуемое поведение не покрыто"
    scores["problem_fit"] = {
        "score": fit,
        "confidence": CONFIDENCE_WITH_EVIDENCE if required else CONFIDENCE_BLIND,
        "reason": reason,
        "evidence": (
            tuple(sorted(diff.get("touched_files", []))),
            tuple(sorted(diff.get("added_tests", []))),
        ),
    }

    # -- scope_discipline --------------------------------------------------
    allowed = scope.get("allowed_files", [])
    declared = set(scope.get("declared_growth", []))
    touched = diff.get("touched_files", [])
    off = [p for p in touched if not any(fnmatch.fnmatch(p, g) for g in allowed)]
    undeclared = [p for p in off if p not in declared]
    if not off:
        disc, reason = 2, "правок вне контракта нет"
    elif not undeclared:
        disc, reason = 1, "границы расширены, но расширение заявлено"
    else:
        disc, reason = 0, f"незаявленные правки: {sorted(undeclared)}"
    scores["scope_discipline"] = {
        "score": disc,
        "confidence": CONFIDENCE_WITH_EVIDENCE if allowed else CONFIDENCE_BLIND,
        "reason": reason,
        "evidence": (tuple(sorted(touched)), tuple(sorted(allowed)), tuple(sorted(declared))),
    }

    # -- assumptions -------------------------------------------------------
    recorded = [a for a in assumptions if a.get("recorded_in")]
    if not assumptions:
        asm, reason = 2, "допущений не заявлено"
    elif len(recorded) == len(assumptions):
        asm, reason = 2, "каждое допущение записано"
    elif recorded:
        asm, reason = 1, f"записано {len(recorded)} из {len(assumptions)}"
    else:
        asm, reason = 0, "ни одно допущение не записано"
    scores["assumptions"] = {
        "score": asm,
        "confidence": CONFIDENCE_ASSUMED,
        "reason": reason,
        # recorded_in бывает None, поэтому подменяем на "": иначе sorted
        # сравнил бы None со строкой и упал бы на одинаковых текстах
        "evidence": tuple(sorted((a["text"], a.get("recorded_in") or "") for a in assumptions)),
    }

    # -- verification_quality ---------------------------------------------
    acceptance = task.get("acceptance", [])
    last = {}
    for record in feedback:
        last[record["command"]] = record.get("exit_code")
    ran = [c for c in acceptance if c in last]
    green = [c for c in ran if last[c] == 0]
    if any(code is None for code in last.values()):
        ver, reason = 0, "в журнале есть прогон без exit_code"
    elif acceptance and len(green) == len(acceptance):
        ver, reason = 2, "все acceptance прогнаны и зелены"
    elif not ran:
        ver, reason = 0, "ни одна acceptance-команда не прогнана"
    else:
        ver, reason = 1, f"зелено {len(green)} из {len(acceptance)}"
    scores["verification_quality"] = {
        "score": ver,
        "confidence": CONFIDENCE_WITH_EVIDENCE if feedback else CONFIDENCE_BLIND,
        "reason": reason,
        "evidence": tuple(sorted(last.items())),
    }

    # -- handoff_readiness -------------------------------------------------
    next_action = handoff.get("next_action")
    dirty = list(handoff.get("clean_state", []))
    if not next_action:
        hnd, reason = 0, "нет next_action: это статус-репорт, а не хендофф"
    elif dirty:
        hnd, reason = 1, f"состояние грязное: {sorted(dirty)}"
    else:
        hnd, reason = 2, "следующая сессия стартует с одного конкретного шага"
    scores["handoff_readiness"] = {
        "score": hnd,
        "confidence": CONFIDENCE_WITH_EVIDENCE if handoff else CONFIDENCE_BLIND,
        "reason": reason,
        "evidence": (next_action, tuple(sorted(dirty))),
    }

    return scores


def verdict_from_scores(scores):
    """Свести оценки в вердикт: "pass", "soft_fail" или "hard_fail".

    verdict_from_scores(десятка)                    ->  "pass"
    verdict_from_scores(шесть без нулей)            ->  "soft_fail"
    verdict_from_scores(девять, но одно измерение 0) ->  "hard_fail"

    Ноль в любом измерении сильнее суммы: девять из десяти при нуле в
    problem_fit означает отличную работу над не той задачей.

    Отсутствующее измерение или оценка вне 0..MAX_PER_DIMENSION — ValueError:
    неполная рубрика молча занизила бы сумму и вердикт стал бы строже
    случайно, а не по существу.
    """
    total = 0
    for name in DIMENSIONS:
        if name not in scores:
            raise ValueError(f"missing dimension: {name}")
        value = scores[name]["score"]
        if not 0 <= value <= MAX_PER_DIMENSION:
            raise ValueError(f"score out of range for {name}: {value}")
        total += value
    if any(scores[name]["score"] == 0 for name in DIMENSIONS):
        return "hard_fail"
    if total < HARD_FAIL_BELOW:
        return "hard_fail"
    if total < SOFT_FAIL_BELOW:
        return "soft_fail"
    return "pass"


def review_report(inputs, confidence_floor=CONFIDENCE_FLOOR, now=0):
    """Отчёт ревьюера: оценки, сумма, вердикт, основания отказа.

    review_report(clean_inputs)["verdict"]  ->  "pass"
    review_report(без журнала прогонов)["verdict"]  ->  "needs_evidence"

    Если самое неуверенное измерение ниже порога, вердикт не выносится:
    verdict="needs_evidence", ship=False, в "blocked_by" перечислены
    измерения, по которым улик не хватило. Отчёт с низким сигналом прочитают
    и поверят — поэтому дешевле попросить улики, чем выдать цифру наугад.

    "grounds" — измерения с нулём. Именно они переезжают в повторный заход.
    """
    scores = score_rubric(inputs)
    total = sum(scores[name]["score"] for name in DIMENSIONS)
    weakest = min(scores[name]["confidence"] for name in DIMENSIONS)
    blocked_by = sorted(n for n in DIMENSIONS if scores[n]["confidence"] < confidence_floor)
    if blocked_by:
        verdict, ship = "needs_evidence", False
    else:
        verdict = verdict_from_scores(scores)
        ship = True
    return {
        "scores": scores,
        "total": total,
        "verdict": verdict,
        "ship": ship,
        "min_confidence": weakest,
        "blocked_by": blocked_by,
        "grounds": sorted(n for n in DIMENSIONS if scores[n]["score"] == 0),
        "sticky_grounds": [],
        "rejected_claims": [],
        "generated_at": now,
    }


def re_review(previous, inputs, confidence_floor=CONFIDENCE_FLOOR, now=0):
    """Повторный заход после правок строителя.

    Ключевое правило: то, что уже отклонили, нельзя одобрить по тому же
    основанию. Если измерение получило ноль в previous, а отпечаток улик по
    нему не сдвинулся — оценка возвращается в ноль, сколько бы строитель ни
    дописал себе в отчёт.

    inputs["resolved_claims"] — список измерений, которые строитель объявил
    починенными. Каждое такое заявление без сдвига улик попадает в
    "rejected_claims".

    Пример: строитель дописал "поведение покрыто" в behaviors_covered, но не
    тронул ни одного файла. score_rubric поверит записи, отпечаток — нет.
    """
    report = review_report(inputs, confidence_floor=confidence_floor, now=now)
    claims = set(inputs.get("resolved_claims", []))
    sticky = []
    for name in previous.get("grounds", []):
        if report["scores"][name]["evidence"] == previous["scores"][name]["evidence"]:
            # улики те же — основание не снято, что бы ни говорил новый счёт
            report["scores"][name]["score"] = 0
            report["scores"][name]["reason"] = "основание не снято: улики не изменились"
            sticky.append(name)
    report["sticky_grounds"] = sorted(sticky)
    report["rejected_claims"] = sorted(claims & set(sticky))
    report["total"] = sum(report["scores"][n]["score"] for n in DIMENSIONS)
    report["grounds"] = sorted(n for n in DIMENSIONS if report["scores"][n]["score"] == 0)
    if not report["blocked_by"]:
        report["verdict"] = verdict_from_scores(report["scores"])
    return report


def consistent_pairwise_winner(judge, a, b):
    """Сравнить два варианта в обоих порядках. Победитель — только если судья
    не передумал.

    judge(x, y) -> "first" или "second".

    consistent_pairwise_winner(lambda x, y: "first", "A", "B")  ->  None
        потому что "всегда первый" — это и есть position bias в чистом виде

    Судьи-LLM переворачивают решение примерно в 40% случаев при перестановке
    вариантов местами. Лечение — считать только согласованные победы, а
    несогласованные объявлять ничьёй.

    Ответ судьи вне {"first", "second"} — ValueError.
    """
    def pick(first, second):
        answer = judge(first, second)
        if answer not in ("first", "second"):
            raise ValueError(f"judge must answer 'first' or 'second', got {answer!r}")
        return first if answer == "first" else second

    forward = pick(a, b)
    backward = pick(b, a)
    return forward if forward == backward else None


def calibration_agreement(review_fn, cases, floor=CALIBRATION_FLOOR):
    """Согласие ревьюера с историческим набором закрытых задач.

    cases: [{"id": ..., "inputs": {...}, "verdict": "pass"}, ...]

    Вернуть {"agreement": доля, "ships": bool, "disagreements": [id, ...]}.

    Ниже floor рубрику не выкатывают: ревьюер, расходящийся с историей,
    сначала правится сам, а уже потом судит новую работу.

    Пустой набор — ValueError: доля от нуля случаев не считается, и главное —
    пустая калибровка ничего не доказывает, а флаг ships при этом был бы True.
    """
    if not cases:
        raise ValueError("empty calibration set proves nothing")
    disagreements = []
    for case in cases:
        if review_fn(case["inputs"])["verdict"] != case["verdict"]:
            disagreements.append(case["id"])
    agreement = (len(cases) - len(disagreements)) / len(cases)
    return {
        "agreement": agreement,
        "ships": agreement >= floor,
        "disagreements": disagreements,
    }

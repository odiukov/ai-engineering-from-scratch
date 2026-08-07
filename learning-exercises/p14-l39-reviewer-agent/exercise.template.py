"""
Агент-ревьюер: строитель и приёмщик — разные роли

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l39-reviewer-agent
Разбор:  /check-code p14-l39-reviewer-agent
"""

import copy
import fnmatch

DIMENSIONS = (
    "problem_fit",
    "scope_discipline",
    "assumptions",
    "verification_quality",
    "handoff_readiness",
)
MAX_PER_DIMENSION = 2
SOFT_FAIL_BELOW = 7
HARD_FAIL_BELOW = 5
CONFIDENCE_FLOOR = 0.6
CONFIDENCE_WITH_EVIDENCE = 0.9
CONFIDENCE_ASSUMED = 0.8
CONFIDENCE_BLIND = 0.3
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def calibration_agreement(review_fn, cases, floor=CALIBRATION_FLOOR):
    """Согласие ревьюера с историческим набором закрытых задач.

    cases: [{"id": ..., "inputs": {...}, "verdict": "pass"}, ...]

    Вернуть {"agreement": доля, "ships": bool, "disagreements": [id, ...]}.

    Ниже floor рубрику не выкатывают: ревьюер, расходящийся с историей,
    сначала правится сам, а уже потом судит новую работу.

    Пустой набор — ValueError: доля от нуля случаев не считается, и главное —
    пустая калибровка ничего не доказывает, а флаг ships при этом был бы True.
    """
    raise NotImplementedError

"""
Сборка полного LLM-пайплайна

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p10-l13-building-complete-llm-pipeline
Разбор:  /check-code p10-l13-building-complete-llm-pipeline
"""

import hashlib
import json


def stable_hash(payload):
    """Контентный адрес артефакта: sha256 от канонического JSON.

    stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})  ->  True
    len(stable_hash({}))  ->  64

    Ловушка: json.dumps по умолчанию сохраняет порядок ключей, и тогда два
    одинаковых по смыслу манифеста дадут разные хэши. Нужен sort_keys=True.

    Content-addressed storage — весь смысл артефактного хранилища из урока:
    имя `latest.pt` врёт, sha256 не врёт никогда.
    """
    raise NotImplementedError


def topological_order(deps):
    """Порядок запуска стадий: зависимости раньше зависимых.

    topological_order({"b": ["a"], "a": []})       ->  ["a", "b"]
    topological_order({"a": [], "b": [], "c": []}) ->  ["a", "b", "c"]

    При равных правах стадии идут по алфавиту — порядок обязан быть
    воспроизводимым, иначе два запуска одного манифеста дадут разные логи.

    Цикл в зависимостях -> ValueError. Ссылка на несуществующую стадию ->
    тоже ValueError: молча пропустить её значит потерять зависимость.
    """
    raise NotImplementedError


def descendants(deps, stage):
    """Все стадии, которые прямо или косвенно зависят от stage.

    descendants({"a": [], "b": ["a"], "c": ["b"]}, "a")  ->  ["b", "c"]
    descendants({"a": [], "b": ["a"], "c": ["b"]}, "c")  ->  []

    Сама стадия в ответ НЕ входит. Обход идёт по обратным рёбрам, поэтому
    считать надо транзитивно: c зависит от a через b.
    """
    raise NotImplementedError


def rollback_set(deps, failed):
    """Что придётся пересчитать при падении стадии failed: она сама и потомки.

    rollback_set({"a": [], "b": ["a"], "c": ["b"]}, "b")  ->  ["b", "c"]
    rollback_set({"a": [], "b": ["a"], "c": ["b"]}, "c")  ->  ["c"]

    Из урока: падение на 06 (SFT) обнуляет 06..12, падение на 11
    (квантизация) — только 11 и 12. План отката пишется ДО запуска, а не
    в четыре утра на выжатой команде.
    """
    raise NotImplementedError


def chain_violations(stages):
    """Разрывы цепочки хэшей: где ожидаемый вход не равен реальному выходу.

    Возвращает отсортированный список пар (стадия, зависимость).

    stages = {"a": {"deps": [], "inputs": {}, "output": "h1"},
              "b": {"deps": ["a"], "inputs": {"a": "h1"}, "output": "h2"}}
    chain_violations(stages)  ->  []

    Если у "b" записан inputs={"a": "СТАРЫЙ"}, ответ  ->  [("b", "a")].

    Ловушка: отсутствие записи об ожидаемом входе — тоже нарушение, а не
    «ну ладно». Именно так порча данных доезжает до конца пайплайна.
    """
    raise NotImplementedError


def gate_failures(gates, metrics):
    """Какие ship-гейты не прошли. Порог задаётся парой (оператор, число).

    gates = {"mmlu": (">=", 0.65), "cost_usd": ("<=", 50000)}
    gate_failures(gates, {"mmlu": 0.70, "cost_usd": 40000})  ->  []
    gate_failures(gates, {"mmlu": 0.60, "cost_usd": 40000})  ->  ["mmlu"]

    Ловушка: метрика, которой в отчёте нет, — это ПРОВАЛ гейта, а не
    пропуск. «Не измерили» и «прошли» — разные вещи.

    Оператор кроме ">=" и "<=" -> ValueError: молчаливое «ну наверное
    больше» в ship-решении недопустимо.
    """
    raise NotImplementedError


def estimate_cost_usd(params, tokens, peak_flops, mfu, usd_per_gpu_hour):
    """Оценка стоимости пред-обучения в долларах до запуска.

    Формула урока: FLOPs = 6 * params * tokens, дальше делим на реально
    достижимую производительность peak_flops * mfu и переводим в часы.

    estimate_cost_usd(1e9, 1e9, 1e12, 0.5, 2.0)  ->  примерно 6666.7
    estimate_cost_usd(7e9, 2e12, 989e12, 0.4, 2.5)  ->  примерно 147455.0

    mfu вне (0, 1] -> ValueError: MFU это доля пикового FLOPs, 40% на 70B
    и 55% на 7B — типичные значения, 1.5 не бывает.

    Дешёвая проверка на этапе `plan` экономит настоящие деньги на `run`.
    """
    raise NotImplementedError


def plan(manifest):
    """Полный `plan`: порядок стадий, цена, нарушения, вердикт SHIP/HOLD.

    Манифест — словарь с ключами "stages", "gates", "metrics",
    "budget_usd" и "pretrain" (аргументы для estimate_cost_usd).

    Возвращает словарь:
      {"order": [...], "cost_usd": float, "violations": [...],
       "failed_gates": [...], "decision": "SHIP" | "HOLD"}

    SHIP выдаётся только если цепочка хэшей цела, все гейты пройдены И
    смета уложилась в бюджет. Любое «почти» -> HOLD.

    Это и есть весь урок в одной функции: манифест на входе, решение
    «катим или держим» на выходе, без субъективных «выглядит нормально».
    """
    raise NotImplementedError

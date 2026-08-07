"""
Сборка полного LLM-пайплайна — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Пайплайн из урока — это словарь стадий. Одна стадия описывается так:

    "06_sft": {
        "deps": ["04_pretrain"],           # от кого зависит
        "inputs": {"04_pretrain": "ab12"}, # какой output_hash она ЖДЁТ
        "output": "cd34",                  # какой output_hash она ВЫДАЛА
    }

Правило урока: output_hash стадии N обязан совпасть с тем, что стадия N+1
записала у себя в inputs. Любое расхождение — остановка пайплайна.
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
    # separators без пробелов: пробелы не несут смысла, а хэш меняют
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def topological_order(deps):
    """Порядок запуска стадий: зависимости раньше зависимых.

    topological_order({"b": ["a"], "a": []})       ->  ["a", "b"]
    topological_order({"a": [], "b": [], "c": []}) ->  ["a", "b", "c"]

    При равных правах стадии идут по алфавиту — порядок обязан быть
    воспроизводимым, иначе два запуска одного манифеста дадут разные логи.

    Цикл в зависимостях -> ValueError. Ссылка на несуществующую стадию ->
    тоже ValueError: молча пропустить её значит потерять зависимость.
    """
    # алгоритм Кана: берём стадии с нулевой входящей степенью, снимаем рёбра
    pending = {name: set(d) for name, d in deps.items()}
    for name, ds in pending.items():
        unknown = ds - set(deps)
        if unknown:
            raise ValueError(f"стадия {name} ссылается на {sorted(unknown)}")

    order = []
    while pending:
        # sorted даёт детерминированный порядок среди равноправных стадий
        ready = sorted(n for n, ds in pending.items() if not ds)
        if not ready:
            raise ValueError(f"цикл в зависимостях: {sorted(pending)}")
        for name in ready:
            order.append(name)
            del pending[name]
        for ds in pending.values():
            ds.difference_update(ready)
    return order


def descendants(deps, stage):
    """Все стадии, которые прямо или косвенно зависят от stage.

    descendants({"a": [], "b": ["a"], "c": ["b"]}, "a")  ->  ["b", "c"]
    descendants({"a": [], "b": ["a"], "c": ["b"]}, "c")  ->  []

    Сама стадия в ответ НЕ входит. Обход идёт по обратным рёбрам, поэтому
    считать надо транзитивно: c зависит от a через b.
    """
    # обратный индекс: кто на кого ссылается
    children = {name: [] for name in deps}
    for name, ds in deps.items():
        for d in ds:
            if d in children:
                children[d].append(name)

    seen = set()
    stack = list(children.get(stage, []))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children.get(node, []))
    return sorted(seen)


def rollback_set(deps, failed):
    """Что придётся пересчитать при падении стадии failed: она сама и потомки.

    rollback_set({"a": [], "b": ["a"], "c": ["b"]}, "b")  ->  ["b", "c"]
    rollback_set({"a": [], "b": ["a"], "c": ["b"]}, "c")  ->  ["c"]

    Из урока: падение на 06 (SFT) обнуляет 06..12, падение на 11
    (квантизация) — только 11 и 12. План отката пишется ДО запуска, а не
    в четыре утра на выжатой команде.
    """
    return sorted([failed] + descendants(deps, failed))


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
    bad = []
    for name in stages:
        stage = stages[name]
        for dep in stage.get("deps", []):
            expected = stage.get("inputs", {}).get(dep)
            actual = stages.get(dep, {}).get("output")
            if expected is None or expected != actual:
                bad.append((name, dep))
    return sorted(bad)


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
    failed = []
    for name in gates:
        op, threshold = gates[name]
        if op not in (">=", "<="):
            raise ValueError(f"неизвестный оператор гейта: {op!r}")
        if name not in metrics:
            failed.append(name)
            continue
        value = metrics[name]
        ok = value >= threshold if op == ">=" else value <= threshold
        if not ok:
            failed.append(name)
    return sorted(failed)


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
    if not 0 < mfu <= 1:
        raise ValueError(f"mfu должен быть в (0, 1], получено {mfu}")
    flops = 6.0 * params * tokens
    gpu_seconds = flops / (peak_flops * mfu)
    return gpu_seconds / 3600.0 * usd_per_gpu_hour


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
    stages = manifest["stages"]
    deps = {name: list(s.get("deps", [])) for name, s in stages.items()}

    order = topological_order(deps)
    violations = chain_violations(stages)
    failed = gate_failures(manifest.get("gates", {}), manifest.get("metrics", {}))
    cost = estimate_cost_usd(**manifest["pretrain"])

    budget = manifest.get("budget_usd", float("inf"))
    over_budget = cost > budget
    ok = not violations and not failed and not over_budget
    return {
        "order": order,
        "cost_usd": cost,
        "violations": violations,
        "failed_gates": failed,
        "decision": "SHIP" if ok else "HOLD",
    }

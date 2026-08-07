"""
Ограниченное самоулучшение: четыре примитива-предохранителя

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l08-bounded-self-improvement
Разбор:  /check-code p15-l08-bounded-self-improvement
"""

import hashlib

GATE_ORDER = ("frozen", "invariant", "anchor", "multi", "regression")


def anchor_digest(text):
    """Отпечаток alignment anchor: первые 16 hex-символов sha256 от текста.

    anchor_digest("")     ->  "e3b0c44298fc1c14"
    anchor_digest("abc")  ->  "ba7816bf8f01cfea"

    Якорь — это неизменяемая формулировка цели, лежащая вне поверхности
    редактирования петли. Сравнивать сами тексты неудобно (они длинные),
    поэтому сравнивают отпечатки.

    Ловушка: hashlib работает с байтами, а не со строками. Кодируй в utf-8
    явно, иначе получишь TypeError на первом же вызове.
    """
    raise NotImplementedError


def gate_frozen(edited_paths, frozen_paths):
    """Примитив «петля не правит собственный проверяющий код».

    gate_frozen(["src/agent.py"], ["eval/checker.py"])   ->  True
    gate_frozen(["eval/checker.py"], ["eval/checker.py"])  ->  False

    Это самый прямой из четырёх предохранителей: набор файлов, к которым
    петля не имеет доступа на запись — сам evaluator, конституция, манифест
    прав. Проверку делает внешний код, который петля редактировать не может.

    Проверять надо ЛЮБОЕ пересечение, а не только полное совпадение: одна
    строчка в проверяющем файле обесценивает все остальные гейты.
    """
    raise NotImplementedError


def gate_invariant(active_manifest, approved_manifest):
    """Примитив «инвариант»: активный манифест инструментов — подмножество одобренного.

    gate_invariant({"trim"}, {"trim", "title"})          ->  True
    gate_invariant({"trim", "raw_eval"}, {"trim"})       ->  False
    gate_invariant(set(), {"trim"})                      ->  True

    Петля вправе перестать пользоваться инструментом, но не вправе
    добавить себе новый. Пустое множество — законное подмножество любого.
    """
    raise NotImplementedError


def gate_anchor(objective, approved_digest):
    """Примитив «якорь»: формулировка цели совпадает с одобренной побайтово.

    gate_anchor("clean the text", anchor_digest("clean the text"))  ->  True
    gate_anchor("clean the text  # relax", anchor_digest("clean the text"))  ->  False

    Даже приписка в комментарии меняет отпечаток. Это и требуется: типичный
    дрейф цели выглядит именно как безобидное уточнение в конце строки.
    """
    raise NotImplementedError


def gate_multi(scores, minimums):
    """Примитив «многокритериальность»: КАЖДАЯ ось должна дотянуть до своего минимума.

    gate_multi({"perf": 0.8, "safety": 1.0}, {"perf": 0.5, "safety": 1.0})  ->  True
    gate_multi({"perf": 0.9, "safety": 0.5}, {"perf": 0.5, "safety": 1.0})  ->  False

    Ось, для которой нет минимума, не ограничена. Ось, для которой минимум
    есть, но нет оценки, считается проваленной — иначе «забыл посчитать
    safety» становится способом её пройти.
    """
    raise NotImplementedError


def gate_regression(history, scores, tol=0.0):
    """Примитив «детектор регрессии»: ни одна ось не просела ниже (лучшее - tol).

    history — список прошлых словарей оценок.

    gate_regression([{"perf": 0.9}], {"perf": 0.8}, tol=0.2)  ->  True
    gate_regression([{"perf": 0.9}], {"perf": 0.5}, tol=0.2)  ->  False
    gate_regression([], {"perf": 0.1})                        ->  True

    Сравнивать надо с ИСТОРИЧЕСКИМ МАКСИМУМОМ, а не с предыдущим циклом.
    Иначе просадка по чуть-чуть на каждом цикле проходит все проверки и
    растворяется в скользящем среднем — ровно тот тихий отказ, ради
    которого примитив и придуман.
    """
    raise NotImplementedError


def review_edit(edit, policy, history=()):
    """Стек гейтов: прогнать правку через все примитивы в порядке GATE_ORDER.

    edit — dict с ключами "files", "manifest", "objective", "scores".
    policy — dict с ключами "frozen", "approved_manifest", "approved_digest",
             "minimums", "tol".

    Вернуть кортеж (accepted, failed) где failed — список имён гейтов,
    которые правка НЕ прошла, в порядке GATE_ORDER.

    Правка принимается, только если failed пустой: любой один отказ
    останавливает петлю. Проверяются все гейты, а не до первого отказа, —
    отчёт «провалено 3 из 5» полезнее, чем «провалено что-то одно».
    """
    raise NotImplementedError


def bounded_loop(proposals, policy, max_cycles):
    """Ограниченная петля: принимать правки, пока не кончится потолок итераций.

    proposals — последовательность правок (в том же формате, что у
    review_edit). Петля берёт их по порядку и не смотрит дальше max_cycles.

    Вернуть dict:
      "accepted"  — сколько правок принято
      "rejected"  — {имя гейта: сколько раз он завалил правку}
      "history"   — список оценок принятых правок
      "reason"    — "ceiling", если израсходован max_cycles,
                    "exhausted", если кончились предложения

    Свойство, ради которого всё: даже когда предложения продолжают
    улучшать метрику, петля останавливается на max_cycles. Потолок — это
    единственный предохранитель, который не зависит от того, насколько
    честно посчитаны метрики.
    """
    raise NotImplementedError

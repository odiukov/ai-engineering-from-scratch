"""
Блоки памяти и sleep-time compute

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l08-memory-blocks-sleep-time-compute
Разбор:  /check-code p14-l08-memory-blocks-sleep-time-compute
"""


def make_block(label, description, limit=300, value=""):
    """Создать блок памяти. Вернуть словарь.

    make_block("human", "facts about the user", limit=180)
        ->  {"label": "human", "value": "", "limit": 180,
             "description": "facts about the user", "version": 1,
             "history": ()}

    description — не украшение: именно по нему модель решает, в какой блок
    писать очередной факт. Пустое описание превращает набор блоков в один
    большой блок с лишними шагами.

    limit меньше длины стартового значения — ValueError: блок, который уже
    переполнен в момент создания, дальше не починить.
    """
    raise NotImplementedError


def block_append(block, text):
    """Дописать текст в блок. Вернуть НОВЫЙ блок с version + 1.

    b = make_block("human", "facts about the user", limit=20)
    block_append(b, "name=ava")["value"]    ->  "name=ava"
    block_append(b, "name=ava")["version"]  ->  2
    block_append(b, "x" * 50)               ->  ValueError

    Куски склеиваются через пробел, как секции core memory в MemGPT.

    Переполнение — ValueError, а не молчаливое обрезание. Это block bloat:
    вызывающий обязан сначала позвать summarize_block и только потом писать.
    Обрезать на месте нельзя — потеряется хвост, и никто этого не заметит.

    Прежнее значение уезжает в history: диффы блоков — единственный способ
    отладить «почему агент забыл X».
    """
    raise NotImplementedError


def block_replace(block, old, new):
    """Заменить подстроку в блоке. Вернуть НОВЫЙ блок с version + 1.

    b = make_block("human", "facts", limit=50, value="city=Berlin")
    block_replace(b, "Berlin", "Lisbon")["value"]  ->  "city=Lisbon"
    block_replace(b, "Paris", "Lisbon")            ->  ValueError

    Как и в append: отсутствие old — ValueError. Агент считает, что поправил
    факт; тихий отказ оставит в промпте старый город навсегда.

    Замена тоже может переполнить блок (new длиннее old) — проверяй лимит.
    """
    raise NotImplementedError


def near_limit(block, threshold=0.8):
    """Пора ли ужимать блок: длина достигла threshold от лимита.

    b = make_block("human", "facts", limit=10, value="12345678")
    near_limit(b)             ->  True    (8 >= 0.8 * 10)
    near_limit(b, 0.9)        ->  False   (8 <  0.9 * 10)

    Порог — компромисс: слишком низкий гоняет summarize на каждый ход,
    слишком высокий даёт block_append упасть с ValueError на ровном месте.

    Сравнение нестрогое (>=): ровно на пороге ужимать уже надо, иначе
    следующая же запись переполнит блок.
    """
    raise NotImplementedError


def summarize_block(block, target_len):
    """Ужать блок до target_len по границам предложений. НОВЫЙ блок, version + 1.

    Предложения разделены точкой. Берём их с начала, пока влезают; хвост
    отбрасываем. Результат оканчивается точкой.

    b = make_block("t", "task", limit=100,
                   value="Plan curriculum. Audience senior. Cite arXiv.")
    summarize_block(b, 35)["value"]  ->  "Plan curriculum. Audience senior."
    summarize_block(b, 20)["value"]  ->  "Plan curriculum."

    Ужимаем ПО ПРЕДЛОЖЕНИЯМ, а не по символам: обрезанное на полуслове
    "Audience seni" модель прочитает как факт и будет на него ссылаться.

    Если даже первое предложение не влезает — возвращаем срез по символам,
    другого выхода нет. Пустое значение остаётся пустым.
    """
    raise NotImplementedError


def dedup_archival(records, threshold=0.9):
    """Схлопнуть почти одинаковые записи. Вернуть НОВЫЙ список.

    Похожесть — коэффициент Жаккара по словам в нижнем регистре. Из группы
    похожих выживает та, что записана РАНЬШЕ: у неё меньший rid, на неё уже
    могли сослаться.

    recs = [{"rid": "a001", "text": "ava lives in berlin", "valid": True},
            {"rid": "a002", "text": "ava lives in berlin now", "valid": True}]
    dedup_archival(recs, threshold=0.7)  ->  только a001

    Это работа sleep-time агента, и только его: дедуп на критическом пути
    добавил бы к каждому ответу задержку ради пользы, которую видно лишь
    через сотни записей.

    Уже инвалидированные записи (valid=False) в сравнении не участвуют и
    остаются в списке как есть — они история, а не мусор.

    Ловушка: функция обязана быть идемпотентной. Второй прогон по своему же
    результату ничего больше не выбрасывает.
    """
    raise NotImplementedError


def invalidate_contradicted(records, claim):
    """Пометить противоречащие записи невалидными. Вернуть (НОВЫЙ список, ids).

    Противоречит та запись, чей текст содержит claim (без учёта регистра).

    recs = [{"rid": "a001", "text": "ava lives in Berlin", "valid": True}]
    invalidate_contradicted(recs, "ava lives in berlin")
        ->  ([{"rid": "a001", "text": "ava lives in Berlin", "valid": False}],
             ["a001"])

    Записи НЕ удаляются, а помечаются: это soft delete. Удалённый факт
    невозможно предъявить аудиту и невозможно откатить, если инвалидация
    оказалась ошибочной. Ровно этим Mem0g отвечает на вопрос «что было
    правдой в марте».

    Уже невалидные записи второй раз не трогаем и в ids не возвращаем —
    иначе трасса sleep-time прохода будет расти на пустом месте.
    """
    raise NotImplementedError


def sleep_time_pass(blocks, records, contradictions=(), threshold=0.8,
                    dedup_threshold=0.9):
    """Один проход sleep-time агента. Вернуть (blocks, records, trace).

    blocks — словарь label -> блок. Проход делает три вещи, в этом порядке:
      1. ужимает каждый блок, для которого near_limit истинно (до половины
         лимита);
      2. инвалидирует записи по каждому утверждению из contradictions;
      3. схлопывает дубликаты в archival.

    sleep_time_pass({}, [], ())  ->  ({}, [], [])

    trace — список строк вида "consolidate human v2", "invalidate a003",
    "dedup 2 -> 1". По ней оператор видит, что именно агент сделал во сне.

    Главное свойство: блок, которому до лимита далеко, проход не трогает —
    ни значение, ни version. Sleep-time агент не должен «переписывать всё на
    всякий случай»: каждая правка блока стоит примерно один вызов модели, а
    молчаливый перезапис persona-блока — это уже инцидент.

    Порядок важен: инвалидация до дедупа, иначе дедуп схлопнет свежий факт
    в старый, который мы ровно сейчас собирались признать протухшим.
    """
    raise NotImplementedError

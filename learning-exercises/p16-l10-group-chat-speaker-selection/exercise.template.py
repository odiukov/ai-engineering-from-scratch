"""
Групповой чат и выбор говорящего

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l10-group-chat-speaker-selection
Разбор:  /check-code p16-l10-group-chat-speaker-selection
"""

TERMINATION_TOKEN = "TERMINATE"
DEFAULT_MAX_ROUNDS = 10


def keyword_score(text, keywords):
    """Сколько ключевых слов роли встретилось в тексте. Регистр не важен.

    keyword_score("Please REVIEW the code", ["review", "test"])  ->  1
    keyword_score("nothing relevant here", ["review"])           ->  0

    Считаем именно СЛОВА-совпадения, а не вхождения: слово, встретившееся
    трижды, даёт 1, а не 3. Иначе один многословный агент перетягивал бы
    выбор на себя простым повтором.

    Это грубая замена LLM-селектора: настоящий читает пул и отвечает именем.
    """
    raise NotImplementedError


def round_robin_selector(pool, names):
    """Следующий говорящий по кругу: тот, кто идёт за последним.

    round_robin_selector([], ["a", "b"])              ->  'a'
    round_robin_selector([("a", "hi")], ["a", "b"])   ->  'b'
    round_robin_selector([("b", "hi")], ["a", "b"])   ->  'a'

    Детерминированно и дёшево, но контекст игнорируется полностью: очередь
    дойдёт до юриста, даже если обсуждают отступы в коде.

    Ловушка: последним мог говорить агент, которого нет в names (его уже
    выгнали из чата). Это ошибка конфигурации, а не повод молча начать
    круг заново.
    """
    raise NotImplementedError


def relevance_selector(pool, specialties, candidates=None):
    """Следующий говорящий по релевантности последнего сообщения.

    specialties — dict {имя агента: список его ключевых слов}.
    candidates — кого вообще рассматриваем; None означает всех.

    relevance_selector([("m", "fix the code")],
                       {"coder": ["code"], "lawyer": ["law"]})   ->  'coder'
    relevance_selector([], {"coder": ["code"], "lawyer": ["law"]})  ->  'coder'

    Ничья решается порядком в specialties (или в candidates) — первый
    подходящий. Без этого правила селектор был бы недетерминированным.

    Осторожно: сам по себе этот селектор НЕ запрещает повтор. Если агент
    сам произносит свои ключевые слова, он же и получит следующий ход — и
    так до конца лимита. Чат выродится в монолог.
    """
    raise NotImplementedError


def auto_selector(pool, specialties, allow_repeat=False):
    """Релевантность плюс запрет говорить дважды подряд.

    auto_selector([("coder", "code code")], {"coder": ["code"], "rev": ["review"]})
        ->  'rev'
    auto_selector([("coder", "code code")], {"coder": ["code"], "rev": ["review"]}, True)
        ->  'coder'

    allow_repeat=True — это и есть hot speaker из урока: тот же агент
    забирает ход снова и снова.

    Ловушка: если в команде ровно один агент, запрет повтора не должен
    оставить селектор без кандидатов. Один агент имеет право говорить
    подряд — выбора всё равно нет.
    """
    raise NotImplementedError


def is_terminated(pool, token=TERMINATION_TOKEN, max_rounds=DEFAULT_MAX_ROUNDS):
    """Пора ли останавливать чат: сработал стоп-токен или упёрлись в лимит.

    is_terminated([("a", "done TERMINATE")])   ->  True
    is_terminated([("a", "still working")])    ->  False
    is_terminated([("a", "x")] * 10)           ->  True   (лимит по умолчанию)

    Два условия, а не одно: токен — это «мы закончили», лимит — «мы
    сдались». Без второго LLM-селектор способен крутить чат бесконечно.
    """
    raise NotImplementedError


def run_groupchat(policies, selector, max_rounds=DEFAULT_MAX_ROUNDS, token=TERMINATION_TOKEN):
    """Прогон группового чата. Возвращает пул сообщений — пары (кто, текст).

    policies — dict {имя агента: функция(pool) -> текст реплики}.
    selector — функция(pool) -> имя следующего говорящего.

    Работа менеджера ровно в этом цикле: спросить селектор, вызвать агента,
    дописать реплику в общий пул, проверить условие остановки.

    Ловушка: селектор может вернуть имя, которого нет в policies (LLM
    выдумал агента). Молча проглотить — значит получить KeyError в
    случайном месте; лучше ValueError сразу и с именем.
    """
    raise NotImplementedError


def speaker_counts(pool):
    """Сколько реплик у каждого агента, в порядке первого появления.

    speaker_counts([("a", "x"), ("b", "y"), ("a", "z")])  ->  {'a': 2, 'b': 1}
    speaker_counts([])                                     ->  {}

    Это метрика speaker balance из чек-листа урока.
    """
    raise NotImplementedError


def dominance(pool):
    """Доля реплик самого говорливого агента. 1.0 — это монолог.

    dominance([("a", "x"), ("b", "y")])  ->  0.5
    dominance([("a", "x"), ("a", "y")])  ->  1.0

    Пустой чат — это не «монолог никого», а отсутствие данных: ValueError.
    """
    raise NotImplementedError

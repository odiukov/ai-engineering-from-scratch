"""
Coreference resolution

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p05-l24-coreference-resolution
Разбор:  /check-code p05-l24-coreference-resolution
"""

import re


def extract_mentions(text):
    """Найти в тексте все mention-ы и вернуть их со span-офсетами.

    Mention — кусок текста, который на кого-то ссылается. Три типа:
      * "ne"      — имя собственное: Mary, Tim Cook, Apple
      * "nominal" — определённое описание: the company, the doctor
      * "pronoun" — местоимение: he, she, it, they

    Каждый mention — dict:
      {"text": ..., "start": int, "end": int, "type": ..., "gender": ..., "number": ...}
    где gender из {"m", "f", "n", "u"} ("u" = unknown, подходит ко всему),
    number из {"sg", "pl"}, а start/end — ОФСЕТЫ В СИМВОЛАХ исходного text,
    причём end эксклюзивный: text[start:end] обязано равняться text mention-а.

    extract_mentions("Mary called John.")
      ->  [{"text": "Mary", "start": 0, "end": 4, "type": "ne",
            "gender": "f", "number": "sg"},
           {"text": "John", "start": 12, "end": 16, "type": "ne",
            "gender": "m", "number": "sg"}]

    extract_mentions("She left.")
      ->  [{"text": "She", "start": 0, "end": 3, "type": "pronoun",
            "gender": "f", "number": "sg"}]

    Словари, на которых работает эта игрушка (больше ничего знать не надо):

      PRONOUNS = {"he": ("m","sg"), "him": ("m","sg"), "his": ("m","sg"),
                  "she": ("f","sg"), "her": ("f","sg"), "hers": ("f","sg"),
                  "it": ("n","sg"), "its": ("n","sg"),
                  "they": ("u","pl"), "them": ("u","pl"), "their": ("u","pl")}

      DETERMINERS = {"the", "a", "an"}

      NOMINAL_HEADS = {"company": ("n","sg"), "firm": ("n","sg"),
                       "device": ("n","sg"), "ceo": ("u","sg"),
                       "doctor": ("u","sg"), "engineer": ("u","sg"),
                       "engineers": ("u","pl"), "products": ("n","pl")}

      FEMALE_FIRST = {"mary", "alice", "sarah", "emma", "jane"}
      MALE_FIRST   = {"john", "james", "david", "tim", "steve", "peter"}

      STOPWORDS = {"the","a","an","when","while","after","before","if","but",
                   "and","then","there","this","that","in","on","at","for",
                   "to","of"}   # с заглавной, но не имя: начало предложения

    Ловушки:
      * end эксклюзивный. Проверка text[start:end] == mention["text"] должна
        держаться на КАЖДОМ mention-е, включая многословные имена.
      * "Tim Cook" — одно имя из двух токенов. А "John. Steve" — два разных
        имени, между ними точка. Соседние заглавные слова склеиваются только
        если между ними ровно один пробел.
      * первое слово предложения тоже с заглавной — для того и STOPWORDS.

    Зачем это в AI: без coreference NER-пайплайн теряет 60-80% упоминаний
    сущности — все "the company" и "they" проходят мимо.
    """
    raise NotImplementedError


def agreement_score(mention, candidate):
    """Насколько mention и кандидат-антецедент согласуются по числу и роду.

    Возвращает float("-inf"), если они несовместимы в принципе, иначе
    1.0 за совпадение числа плюс ещё 1.0, если род совпал ТОЧНО (не по "u").

    agreement_score({"gender":"f","number":"sg"}, {"gender":"f","number":"sg"})  ->  2.0
    agreement_score({"gender":"n","number":"sg"}, {"gender":"u","number":"sg"})  ->  1.0
    agreement_score({"gender":"f","number":"sg"}, {"gender":"m","number":"sg"})  ->  -inf
    agreement_score({"gender":"u","number":"pl"}, {"gender":"u","number":"sg"})  ->  -inf

    Ловушка: "u" — это wildcard, а не четвёртый род. "u" против "m" — это
    совместимо (просто без бонуса), а вот "m" против "f" — нет.

    Зачем это в AI: это ровно тот hard constraint, который нейросетевой
    ranker выучивает из данных, а нам приходится писать руками. И это же
    место, где такие системы ломаются на небинарных референтах.
    """
    raise NotImplementedError


def recency_score(mention, candidate):
    """Насколько кандидат близок к mention-у: чем ближе, тем больше.

    Формула: 1 / (1 + расстояние в символах между их start-ами.)

    recency_score({"start": 10}, {"start": 0})  ->  0.0909...   (1/11)
    recency_score({"start": 10}, {"start": 9})  ->  0.5         (1/2)

    Кандидат ОБЯЗАН начинаться строго раньше mention-а — иначе ValueError.
    Это не придирка: антецедент по определению стоит слева, а случай
    "местоимение раньше референта" (катафора) обрабатывается отдельно.

    Ловушка: результат всегда лежит в (0, 1], поэтому близость никогда не
    перевесит бонус за точное согласование из agreement_score. Так и задумано:
    согласование — жёсткое правило, близость — только тай-брейк.
    """
    raise NotImplementedError


def resolve_pronouns(mentions):
    """Для каждого местоимения выбрать лучший антецедент. Mention-ranking.

    Возвращает список пар (индекс местоимения, индекс антецедента или None),
    в порядке появления местоимений.

    Кандидатами считаются только предшествующие mention-ы типа "ne" и
    "nominal": местоимение, указывающее на местоимение, ничего не проясняет.
    Score кандидата = agreement_score + recency_score; кандидаты со score
    -inf выбывают.

    resolve_pronouns(extract_mentions("Mary called John. She was late."))
      ->  [(2, 0)]      # She -> Mary, потому что John не проходит по роду

    resolve_pronouns(extract_mentions("When she walked in, Mary smiled."))
      ->  [(0, None)]   # катафора: слева от she нет ни одного кандидата

    Ловушка: если совместимых кандидатов не осталось, ставь None, а не
    ближайший попавшийся. Ложная ссылка хуже отсутствующей — она уедет в
    knowledge graph и её оттуда никто не выковыряет.
    """
    raise NotImplementedError


def build_clusters(n_mentions, links):
    """Собрать кластеры из попарных ссылок транзитивным замыканием.

    Кластер — множество mention-ов, указывающих на одну сущность. Если a
    ссылается на b, а c ссылается на b, то a, b и c — один кластер.

    build_clusters(4, [(2, 0), (3, 2)])  ->  [[0, 2, 3], [1]]
    build_clusters(3, [])                ->  [[0], [1], [2]]

    Пары со вторым элементом None игнорируются. Кластеры возвращаются
    отсортированными по возрастанию индекса внутри и по первому элементу
    снаружи. Одиночки (singletons) тоже возвращаются — выкидывать их или
    нет, решает вызывающий код.

    Индекс вне диапазона [0, n_mentions) — ValueError.

    Ловушка: наивное "склеить пары в списки" даёт неверный ответ, когда
    цепочка длиннее двух звеньев. Нужен именно транзитив: union-find или
    обход графа.
    """
    raise NotImplementedError


def resolve_document(text):
    """Полный проход: текст -> кластеры текстов mention-ов. Singletons отброшены.

    resolve_document("Mary called John. She was late. She apologized.")
      ->  [["Mary", "She", "She"]]

    resolve_document("Nothing to see here.")
      ->  []

    Это тот же контракт, что у `doc._.coref_clusters` в spaCy: на выходе
    только кластеры длиннее одного mention-а, потому что кластер из одного
    элемента ничего не связывает.

    Собери из уже написанных функций, не пиши логику заново.
    """
    raise NotImplementedError


def muc_f1(pred_clusters, gold_clusters):
    """MUC precision, recall и F1 между предсказанными и золотыми кластерами.

    MUC считает не mention-ы, а СВЯЗИ. Кластеру из k элементов нужно k-1
    связей, чтобы быть связным. Recall: сколько золотых связей уцелело, если
    разрезать каждый золотой кластер по границам предсказанных. Precision —
    то же самое с переставленными аргументами.

    muc_f1([[0, 1, 2]], [[0, 1, 2]])        ->  (1.0, 1.0, 1.0)
    muc_f1([[0], [1], [2]], [[0, 1, 2]])    ->  (0.0, 0.0, 0.0)
    muc_f1([[0, 1, 2, 3]], [[0, 1], [2, 3]])  ->  (0.666..., 1.0, 0.8)

    Ловушки:
      * знаменатель precision — сумма (|P| - 1) по предсказанным кластерам.
        На «singleton explosion» (каждый mention сам себе кластер) он равен
        нулю. Деление на ноль тут не ошибка входных данных, а ответ 0.0.
      * F1 при p + r == 0 — тоже 0.0, а не ZeroDivisionError.

    Зачем это в AI: одной метрики для кластеризации не хватает, поэтому в
    CoNLL F1 усредняют MUC, B-cubed и CEAF. MUC — самая строгая к singleton-ам.
    """
    raise NotImplementedError

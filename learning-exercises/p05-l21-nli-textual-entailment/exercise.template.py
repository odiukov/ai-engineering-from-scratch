"""
NLI и textual entailment

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p05-l21-nli-textual-entailment
Разбор:  /check-code p05-l21-nli-textual-entailment
"""

import math
import re


def tokenize(text):
    """Режет текст на слова: нижний регистр, только буквы и цифры.

    tokenize("The cat is on the mat.")  ->  ['the', 'cat', 'is', 'on', 'the', 'mat']
    tokenize("Nobody's HOME!")          ->  ['nobody', 's', 'home']
    tokenize("")                        ->  []

    Ловушка: text.split() оставит точку приклеенной к слову, и "mat." никогда
    не совпадёт с "mat" из другого предложения. Резать надо по символам, а не
    по пробелам.

    Зачем это в AI: любой NLI-модели тоже сначала нужен токенизатор. У BERT он
    subword и обучаемый, у нас — одна регулярка, но роль та же: превратить
    строку в сравнимые единицы.
    """
    raise NotImplementedError


def has_negation(tokens):
    """Есть ли в списке токенов слово-отрицание.

    Список отрицаний ровно такой:
    {"not", "no", "never", "nobody", "nothing", "none", "neither", "nor", "without"}

    has_negation(["there", "is", "a", "cat"])   ->  False
    has_negation(["there", "is", "no", "cat"])  ->  True
    has_negation(["nobody", "came"])            ->  True

    Ловушка: проверять вхождение подстроки в текст нельзя. "not" сидит внутри
    слов "note", "nothing", "cannot", и проверка `"not" in text` объявит
    отрицанием фразу "the note is on the table". Сравнивай токены целиком.

    Зачем это в AI: отрицание — единственный признак, который переворачивает
    entailment в contradiction, не трогая ни одного другого слова. Именно на
    нём модели и учатся жульничать (см. hypothesis_only_label).
    """
    raise NotImplementedError


def lexical_overlap(premise, hypothesis):
    """Доля слов гипотезы, которые встречаются в premise. Число от 0 до 1.

    lexical_overlap("A cat is sleeping on the couch.", "There is a cat.")
        ->  0.75      (3 слова из 4: is, a, cat; "there" в premise нет)
    lexical_overlap("A cat is sleeping.", "The dog chased a ball.")
        ->  0.2       (из пяти слов гипотезы нашлось одно: "a")
    lexical_overlap("A cat is sleeping.", "")
        ->  0.0

    Ловушка №1: делить надо на длину ГИПОТЕЗЫ, а не premise. Мера намеренно
    несимметричная: длинный premise, покрывающий короткую гипотезу, — это
    хорошо, а не плохо.
    Ловушка №2: пустая гипотеза даёт 0.0, а не ZeroDivisionError.
    Ловушка №3: слова premise стоит сложить в set — иначе на длинных текстах
    получится квадрат.

    Зачем это в AI: это и есть та самая "lexical overlap heuristic" из урока.
    Она вытягивает SNLI и с треском проваливает HANS — но показывает форму
    задачи: два текста на входе, одно число близости.
    """
    raise NotImplementedError


def softmax(scores):
    """Превращает словарь {метка: score} в распределение вероятностей.

    softmax({"a": 0.0, "b": 0.0})   ->  {'a': 0.5, 'b': 0.5}
    softmax({"a": 1.0, "b": 0.0})   ->  {'a': 0.731…, 'b': 0.268…}
    softmax({})                     ->  ValueError

    Формула: exp(s_i) / sum(exp(s_j)). Сумма результата всегда 1, порядок
    score-ов сохраняется: у кого score больше, у того и вероятность больше.

    Ловушка: math.exp(1000) — это OverflowError. Вычти максимум из всех
    score-ов перед экспонентой: exp(s - m) / sum(exp(s_j - m)). Результат
    математически тот же (множитель exp(-m) сокращается), а переполнения нет.

    Зачем это в AI: последний слой NLI-модели — ровно этот softmax по трём
    числам {entailment, contradiction, neutral}. Всё, что до него, — способ
    получить эти три числа.
    """
    raise NotImplementedError


def nli_scores(premise, hypothesis):
    """Игрушечный NLI: вероятности entailment / contradiction / neutral.

    Правило (лексическое перекрытие + рассогласование отрицаний):

        support  = 6 * lexical_overlap(premise, hypothesis) - 3
        mismatch = 1, если отрицание есть ровно в одном из текстов, иначе 0

        entailment    = support - 4 * mismatch
        contradiction = support - 4 * (1 - mismatch)
        neutral       = 0

    и всё это через softmax.

    nli_scores("A cat is sleeping on the couch.", "There is a cat.")
        ->  entailment ≈ 0.81   (перекрытие 0.75, отрицаний нет)
    nli_scores("A cat is sleeping on the couch.", "There is no cat on the couch.")
        ->  contradiction ≈ 0.77   (перекрытие есть, но отрицание одно на двоих)
    nli_scores("A cat is sleeping.", "The dog chased the ball.")
        ->  neutral ≈ 0.95     (перекрытия нет, спорить не о чем)

    Читается так: перекрытие даёт "поддержку", а отрицание решает, кому эта
    поддержка достанется — entailment или contradiction. neutral — это ноль,
    планка, которую надо перебить.

    Ловушка №1: mismatch — это ИСКЛЮЧАЮЩЕЕ ИЛИ. Отрицание с обеих сторон
    ("no cat" против "not a cat") друг друга гасит, mismatch = 0.
    Ловушка №2: перекрытие ровно 0.5 — точка безразличия, support = 0, и
    entailment сравнивается с neutral один в один.

    Зачем это в AI: контракт тот же, что у
    `pipeline("text-classification", model="facebook/bart-large-mnli")` на паре
    {"text": premise, "text_pair": hypothesis}: на входе два текста, на выходе
    три метки со score-ами, которые в сумме дают 1. Меняется только то, чем
    считаются три числа: у нас регулярка, там 400M параметров.
    """
    raise NotImplementedError


def hypothesis_only_label(hypothesis):
    """Baseline, который НЕ ВИДИТ premise: метка по одной гипотезе.

    Правило: есть отрицание -> "contradiction", иначе -> "neutral".

    hypothesis_only_label("Nobody is in the room.")  ->  'contradiction'
    hypothesis_only_label("There is no cat.")        ->  'contradiction'
    hypothesis_only_label("A man plays guitar.")     ->  'neutral'

    Обрати внимание на сигнатуру: premise сюда не передаётся вообще. Это не
    забывчивость, а вся суть baseline-а.

    Зачем это в AI: на SNLI такой baseline даёт около 60% при случайных 33% —
    потому что аннотаторы, придумывая contradiction, почти всегда писали
    "not"/"nobody"/"never". Метка утекла в гипотезу. Прежде чем радоваться
    точности своей NLI-модели, прогони hypothesis-only: разрыв между ними и
    есть то, чему модель научилась на самом деле.
    """
    raise NotImplementedError


def zero_shot_classify(text, candidate_labels,
                       hypothesis_template="This example is about {label}."):
    """Классификация без обучения: метки превращаются в гипотезы.

    Каждая метка подставляется в шаблон, получается гипотеза; text играет роль
    premise; берётся вероятность entailment. Полученные числа нормируются на
    свою сумму, чтобы получилось распределение по меткам.

    Вернуть список пар (метка, score), отсортированный по убыванию score.

    zero_shot_classify("The finance ministry cut interest rates today.",
                       ["finance", "sports"])
        ->  [('finance', 0.749…), ('sports', 0.250…)]
    zero_shot_classify("Interest rates and the finance ministry dominated "
                       "the news.", ["finance", "sports"], "{label}")
        ->  [('finance', 0.951…), ('sports', 0.048…)]
        (тот же текст, другой шаблон — и разрыв между метками другой)

    Ловушка №1: сортировка по убыванию, а не по возрастанию, и метка идёт
    первой в паре.
    Ловушка №2: при равных score порядок должен остаться входным. Питоновский
    sorted стабилен — просто не сортируй дважды и не выворачивай список
    через reversed().
    Ловушка №3: шаблон подставляется через .format(label=...), сам текст метки
    при этом не трогается.

    Зачем это в AI: это контракт `pipeline("zero-shot-classification")` —
    на входе текст и candidate_labels, на выходе метки, отсортированные по
    score, и score-ы в сумме дают 1. Шаблон "This example is about {label}."
    там ровно такой же по умолчанию, и он же — главный рычаг качества: смена
    шаблона двигает точность на десяток пунктов.
    """
    raise NotImplementedError


def is_faithful(answer, context, threshold=0.5):
    """Проверка ответа RAG на выдумки: доля claim-ов, выводимых из context.

    Ответ режется на атомарные claim-ы по границам предложений. Для каждого
    claim-а считается nli_scores(context, claim); claim засчитывается, если
    самая вероятная метка — "entailment". Функция возвращает True, когда
    доля засчитанных claim-ов >= threshold.

    ctx = "The Eiffel Tower is in Paris. It was completed in 1889."
    is_faithful("The Eiffel Tower is in Paris.", ctx, 1.0)        ->  True
    is_faithful("The Eiffel Tower is in Paris. It is made of "
                "chocolate and floats.", ctx, 1.0)                ->  False
                                                     (доля упала до 0.5)
    is_faithful("", ctx)                                          ->  ValueError

    Ловушка №1: не режь по answer.split("."). Точка при этом исчезает, а в
    конце появляется пустой кусок, и доля вместо 1.0 станет 2/3. Режь так,
    чтобы знак конца предложения оставался в claim-е, и выкидывай куски без
    единого слова.
    Ловушка №2: порядок аргументов у nli_scores — (premise, hypothesis).
    Premise здесь context, гипотеза — claim, а не наоборот.
    Ловушка №3: сравнение с порогом нестрогое (>=), иначе полностью
    подтверждённый ответ не пройдёт проверку с threshold=1.0.

    Зачем это в AI: это ядро метрики faithfulness из RAGAS. Разбить ответ на
    атомарные утверждения, каждое проверить NLI-моделью против найденного
    контекста, вернуть долю подтверждённых. Одна невыводимая фраза в ответе
    сразу роняет долю — именно так галлюцинацию и ловят.
    """
    raise NotImplementedError

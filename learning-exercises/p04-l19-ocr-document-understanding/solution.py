"""
OCR и понимание документов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def ctc_collapse(indices, blank=0):
    """Схлопывание CTC: склеить соседние повторы, потом выбросить blank.

    ctc_collapse([1, 1, 1, 0, 0, 2, 2, 3, 3, 0, 3, 3, 4])  ->  [1, 2, 3, 3, 4]
    ctc_collapse([0, 0, 0])                                ->  []

    Порядок операций принципиален: сначала повторы, потом blank. Именно
    поэтому blank между двумя одинаковыми буквами спасает двойное "ll" в
    "hello" — он разрывает серию повторов, и обе буквы доживают до выхода.
    Если сначала выбросить blank, а потом схлопнуть, останется одна "l".
    """
    out = []
    prev = None
    for idx in indices:
        # два условия: не повтор предыдущего И не blank.
        # prev обновляем ВСЕГДА, в том числе на blank — это и есть разрыв серии
        if idx != prev and idx != blank:
            out.append(idx)
        prev = idx
    return out


def greedy_ctc_decode(log_probs, blank=0):
    """Жадное CTC-декодирование: argmax на каждом шаге, потом схлопывание.

    log_probs — таблица T строк по C чисел: log-вероятность каждого символа
    на каждом шаге времени.

    greedy_ctc_decode([[0.0, -9.0], [-9.0, 0.0], [-9.0, 0.0]])  ->  [1]

    Жадный декодер берёт лучший символ независимо на каждом шаге. Beam search
    честнее (он суммирует вероятности всех выравниваний), но на практике
    разница по CER около одного процента, поэтому в проде живёт жадный.
    """
    # argmax по строке: index максимума. max(range(...), key=...) вместо
    # sort — нам нужен один элемент, а не порядок, это O(C) вместо O(C log C)
    best = [max(range(len(row)), key=lambda c: row[c]) for row in log_probs]
    return ctc_collapse(best, blank)


def decode_text(log_probs, vocab, blank=0):
    """То же декодирование, но сразу строкой: индексы отображаются в vocab.

    vocab — список символов, где vocab[blank] это служебный пустой символ.

    decode_text([[0, -9, -9], [-9, 0, -9], [-9, 0, -9], [-9, -9, 0]],
                ["_", "a", "b"])  ->  "ab"

    Здесь видно, зачем blank сидит в vocab: модель предсказывает C классов,
    и один из них — "я ничего не говорю на этом шаге".
    """
    return "".join(vocab[i] for i in greedy_ctc_decode(log_probs, blank))


def levenshtein(a, b):
    """Расстояние Левенштейна: минимум вставок, удалений и замен.

    levenshtein("kitten", "sitting")  ->  3
    levenshtein("abc", "abc")         ->  0
    levenshtein("", "abc")            ->  3

    Работает и на строках, и на списках слов — важно только сравнение
    элементов на равенство.

    Это основа CER и WER, то есть основа всей оценки качества OCR.
    """
    # держим только две строки таблицы, а не всю матрицу:
    # памяти O(len(b)) вместо O(len(a) * len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,           # удаление
                cur[j - 1] + 1,        # вставка
                prev[j - 1] + (ca != cb),  # замена (или совпадение)
            ))
        prev = cur
    return prev[-1]


def cer(reference, hypothesis):
    """Character Error Rate: расстояние Левенштейна, делённое на длину эталона.

    cer("hello", "helo")   ->  0.2    (одно удаление на пять символов)
    cer("hello", "hello")  ->  0.0

    Ловушка: делим на длину ЭТАЛОНА, не гипотезы. Иначе модель, выдающая
    длинную кашу, получит подозрительно хороший балл.

    Второе: CER может быть больше единицы. Если модель на слово из трёх
    букв выдала сорок символов, ошибка честно равна 13.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def wer(reference, hypothesis):
    """Word Error Rate: то же самое, но единица сравнения — слово.

    wer("the cat sat", "the cat sit")  ->  примерно 0.333
    wer("the cat sat", "the cat sat")  ->  0.0

    Слова режем по пробелам. Одна опечатка внутри слова стоит целого слова —
    поэтому WER всегда пессимистичнее CER на тех же данных.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def reading_order(boxes, line_tol=10.0):
    """Порядок чтения: сгруппировать боксы в строки, вернуть индексы по порядку.

    boxes — список (x1, y1, x2, y2). Возвращается список индексов:
    сверху вниз по строкам, внутри строки слева направо.

    reading_order([(50, 0, 60, 10), (0, 2, 10, 12)])  ->  [1, 0]
    reading_order([(0, 100, 10, 110), (0, 0, 10, 10)])  ->  [1, 0]

    Две ловушки. Первая: сортировать просто по y — неверно, соседние слова
    одной строки почти никогда не выровнены пиксель в пиксель, отсюда
    line_tol. Вторая: строку определяет центр бокса, а не его верх — иначе
    высокая заглавная буква уедет в предыдущую строку.
    """
    # 1) все боксы по вертикали, ключ — центр, а не верхняя грань
    by_y = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)

    # 2) жадная нарезка на строки: пока центр не ушёл от начала строки дальше
    #    line_tol, считаем что это всё ещё та же строка
    lines = []
    for i in by_y:
        cy = (boxes[i][1] + boxes[i][3]) / 2
        if lines and cy - lines[-1][0] <= line_tol:
            lines[-1][1].append(i)
        else:
            lines.append((cy, [i]))

    # 3) внутри строки — слева направо по левому краю
    order = []
    for _, idxs in lines:
        order.extend(sorted(idxs, key=lambda i: boxes[i][0]))
    return order


def field_f1(predicted, gold):
    """F1 по структурным полям документа: сравнение двух словарей.

    Поле засчитывается, только если совпали и ключ, и значение.
    Возвращает кортеж (precision, recall, f1).

    field_f1({"total": "42.50"}, {"total": "42.50"})  ->  (1.0, 1.0, 1.0)
    field_f1({"total": "42.50", "date": "x"},
             {"total": "42.50"})                      ->  (0.5, 1.0, ...)

    Ловушка: precision делится на число предсказанных полей, recall — на
    число эталонных. Перепутать местами легко, а разница драматична:
    модель, выдающая одно верное поле из ста нужных, получит precision 1.0
    и recall 0.01.
    """
    tp = sum(1 for k, v in predicted.items() if k in gold and gold[k] == v)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return (precision, recall, 0.0)
    return (precision, recall, 2 * precision * recall / (precision + recall))

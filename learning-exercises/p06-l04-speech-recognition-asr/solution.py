"""
Распознавание речи: CTC, RNN-T, attention — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def collapse_ctc(ids, blank=0):
    """Свернуть выход CTC в последовательность токенов: склеить повторы, убрать blank.

    Два правила, и порядок между ними важен: СНАЧАЛА схлопываются подряд
    идущие одинаковые токены, ПОТОМ выбрасываются blank.

    collapse_ctc([1, 1, 0, 0, 1, 2, 2, 0, 3])  ->  [1, 1, 2, 3]
    collapse_ctc([1, 1])                       ->  [1]
    collapse_ctc([1, 0, 1])                    ->  [1, 1]
    collapse_ctc([0, 0, 0])                    ->  []

    Ловушка: blank РАЗДЕЛЯЕТ повторы, а не удаляется до склейки. Если сначала
    выкинуть blank, то [1, 0, 1] превратится в [1, 1] -> [1], и удвоенные
    буквы («сумма», «hello») перестанут распознаваться вообще. Ровно ради
    этого blank в алфавите и заведён.

    Зачем в AI: это весь inference CTC. wav2vec 2.0 и MMS выдают по вектору
    вероятностей на кадр, argmax по кадрам — и вот эта функция.
    """
    out = []
    prev = None  # None, а не -1: blank вполне может иметь индекс 0
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def ctc_greedy_decode(frame_probs, blank=0, vocab=None):
    """Жадное декодирование CTC: argmax по каждому кадру, затем collapse_ctc.

    frame_probs — список кадров, каждый кадр это вектор вероятностей по
    словарю (индекс blank входит в него же). Если передан vocab (строка или
    список символов), результат склеивается в строку, иначе возвращаются id.

    ctc_greedy_decode([[0.1, 0.9], [0.1, 0.9], [0.8, 0.2]])       ->  [1]
    ctc_greedy_decode([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])       ->  [1, 1]
    ctc_greedy_decode([[0.1, 0.9], [0.8, 0.2]], vocab="_a")       ->  "a"

    При ничьей внутри кадра берётся наименьший индекс — так декодирование
    остаётся детерминированным.

    Ловушка: argmax берётся ПОКАДРОВО и независимо. У CTC нет внутренней
    языковой модели: каждый кадр не знает, что предсказали соседние. Отсюда
    «фонетически похожая чушь» на выходе и необходимость внешней LM.
    """
    # max(range(...), key=...) при равенстве возвращает первый максимум —
    # именно поэтому ничья уходит наименьшему индексу
    best = [max(range(len(p)), key=lambda i: p[i]) for p in frame_probs]
    ids = collapse_ctc(best, blank)
    if vocab is None:
        return ids
    return "".join(vocab[i] for i in ids)


def ctc_beam_decode(frame_probs, beam=8, blank=0):
    """Лучевой поиск по CTC: держим beam лучших гипотез и возвращаем список id.

    На каждом кадре каждую гипотезу продолжаем всеми токенами, складывая
    log-вероятности, потом оставляем beam лучших по сумме. Blank и повтор
    последнего токена гипотезу не удлиняют.

    ctc_beam_decode([[0.1, 0.9], [0.1, 0.9], [0.8, 0.2]], beam=4)  ->  [1]

    Вероятности берутся через log(max(p, 1e-10)): log(0) это -inf, и одна
    нулевая вероятность отравила бы всю гипотезу навсегда.

    Ловушка: при beam=1 результат обязан совпасть с жадным декодированием.
    Если не совпадает — где-то перепутано сложение log-вероятностей.

    Честное ограничение этого скелета: он не умеет выдавать удвоенный токен
    ([1, 1]), потому что не различает «повтор через blank» и «повтор подряд».
    Настоящий prefix beam search хранит для каждой гипотезы две вероятности
    (закончилась blank / закончилась токеном) и умеет. Здесь важна идея:
    жадность смотрит на один кадр, луч — на всю последовательность.
    """
    beams = [([], 0.0)]  # (последовательность токенов, сумма log p)
    for p in frame_probs:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                if t == blank or (seq and seq[-1] == t):
                    new = seq
                else:
                    new = seq + [t]
                candidates.append((new, lp + lpt))
        # сортируем по убыванию log-вероятности и подрезаем до ширины луча
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]


def count_ctc_alignments(target, n_frames, blank=0):
    """Сколько выравниваний длины n_frames схлопываются ровно в target.

    Это та самая сумма, которую CTC-loss берёт по всем выравниваниям, только
    без вероятностей — просто счёт путей.

    count_ctc_alignments([1, 2], 2)  ->  1    (единственный путь [1, 2])
    count_ctc_alignments([1, 2], 3)  ->  5
    count_ctc_alignments([1], 2)     ->  3    ([1,1], [1,blank], [blank,1])
    count_ctc_alignments([1, 1], 2)  ->  0    (двойной токен требует blank)
    count_ctc_alignments([1, 2], 1)  ->  0    (кадров меньше, чем токенов)

    Как считать: разложи target в «расширенную» последовательность
    blank, y1, blank, y2, ..., blank длины 2*U+1 и пройди по кадрам динамикой.
    Из позиции s можно шагнуть в s (стоим), s+1 (следующий элемент) и в s+2 —
    но только если s+2 не blank и не равен элементу в s (иначе два одинаковых
    токена склеятся в один).

    Зачем в AI: CTC не требует разметки «какой кадр какой букве соответствует»
    именно потому, что суммирует по всем вариантам. Эта функция показывает,
    насколько их много: уже на 10 кадрах и 3 токенах счёт идёт на сотни.
    """
    ext = [blank]
    for t in target:
        ext.append(t)
        ext.append(blank)
    n = len(ext)

    # alpha[s] — сколько путей длины (номер кадра + 1) заканчиваются в ext[s]
    alpha = [0] * n
    alpha[0] = 1
    if n > 1:
        alpha[1] = 1

    for _ in range(1, n_frames):
        nxt = [0] * n
        for s in range(n):
            total = alpha[s]
            if s > 0:
                total += alpha[s - 1]
            # прыжок через blank разрешён только между РАЗНЫМИ токенами
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                total += alpha[s - 2]
            nxt[s] = total
        alpha = nxt

    if n == 1:
        return alpha[0]
    return alpha[n - 1] + alpha[n - 2]


def normalize_text(text):
    """Нормализация текста перед подсчётом WER: нижний регистр, без пунктуации.

    Знаки препинания удаляются (не заменяются пробелом), подряд идущие
    пробелы схлопываются в один, края обрезаются.

    normalize_text("Hello, WORLD!")     ->  "hello world"
    normalize_text("  Don't   stop. ")  ->  "dont stop"
    normalize_text("")                  ->  ""

    Ловушка: нормализовать надо ОБЕ строки — и эталон, и гипотезу. Whisper
    возвращает текст с пунктуацией и заглавными, LibriSpeech хранит эталон
    капсом без знаков. Сравнение «как есть» даёт WER около 100% на идеальном
    распознавании.

    Функция обязана быть идемпотентной: применить её дважды — то же самое,
    что применить один раз.
    """
    kept = [c for c in text.lower() if c.isalnum() or c.isspace()]
    # split() без аргумента сам схлопывает любые пробельные и режет края
    return " ".join("".join(kept).split())


def edit_distance(ref, hyp):
    """Расстояние Левенштейна между двумя последовательностями (списками токенов).

    Минимальное число замен, удалений и вставок, превращающих ref в hyp.

    edit_distance(["a", "b"], ["a", "b"])       ->  0
    edit_distance(["a", "b", "c"], ["a", "x"])  ->  2   (замена + удаление)
    edit_distance([], ["a", "b"])               ->  2   (две вставки)

    Считается динамикой по таблице (len(ref)+1) x (len(hyp)+1): первая строка
    и первый столбец заполняются индексами (превратить в пустую строку можно
    только удалениями), дальше минимум из трёх переходов.

    Сложность O(len(ref) * len(hyp)) по времени.

    Зачем в AI: числитель WER. Он же — CER, если токены это символы, и он же
    лежит в основе всех метрик выравнивания последовательностей.
    """
    n, m = len(ref), len(hyp)
    # держим только две строки таблицы: полная нужна лишь для backtrace
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def edit_counts(ref, hyp):
    """Разложить расстояние на три числа: замены, удаления, вставки.

    Возвращает словарь с ключами "substitutions", "deletions", "insertions".
    Удаление — слово было в эталоне и пропало. Вставка — модель придумала
    лишнее слово.

    edit_counts(["a", "b"], ["a", "b"])
        ->  {"substitutions": 0, "deletions": 0, "insertions": 0}
    edit_counts(["a", "b", "c"], ["a", "x", "c"])
        ->  {"substitutions": 1, "deletions": 0, "insertions": 0}
    edit_counts(["a", "b"], ["a"])
        ->  {"substitutions": 0, "deletions": 1, "insertions": 0}

    Сумма трёх чисел обязана совпасть с edit_distance(ref, hyp) — на этом
    свойстве функцию и проверяют.

    Зачем в AI: WER 15% из одних вставок и WER 15% из одних удалений — это
    две разные поломки. Много вставок — модель галлюцинирует на тишине
    (нужен VAD). Много удалений — режется звук или слишком агрессивный
    порог. Одна цифра WER этого не показывает.
    """
    n, m = len(ref), len(hyp)
    # здесь нужна вся таблица: по ней пойдём назад и посчитаем типы правок
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        # порядок проверок задаёт разбор ничьих; сумма от него не зависит
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    subs += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
            continue
        ins += 1
        j -= 1
    return {"substitutions": subs, "deletions": dels, "insertions": ins}


def wer(reference, hypothesis, normalize=True):
    """Word Error Rate: (S + D + I) / N, где N — число слов в эталоне.

    Строки режутся на слова по пробелам. При normalize=True обе строки
    сначала прогоняются через normalize_text.

    wer("turn on the light", "turn on the light")   ->  0.0
    wer("Turn on the light!", "turn on the light")  ->  0.0   (нормализация)
    wer("a b c d", "a x c")                         ->  0.5   (замена + удаление)
    wer("", "hello")                                ->  1.0   (без деления на ноль)

    Ловушка: WER бывает БОЛЬШЕ 1.0. Если модель нагаллюцинировала двадцать
    слов на трёхсловной фразе, вставок больше, чем слов в эталоне. Метрика,
    зажатая в [0, 1], посчитана неправильно.

    Вторая ловушка: пустой эталон. Делить на ноль нельзя, знаменатель берут
    как max(1, N).

    Зачем в AI: единственное число, которым меряют ASR. 1.4% у Parakeet-TDT
    на LibriSpeech test-clean, выше 20% — система непригодна.
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    ref, hyp = reference.split(), hypothesis.split()
    return edit_distance(ref, hyp) / max(1, len(ref))

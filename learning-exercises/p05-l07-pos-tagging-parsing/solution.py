"""
POS-теггинг и синтаксический разбор — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def ptb_to_ud(tag):
    """Перевод тега Penn Treebank в грубый тег Universal Dependencies.

    ptb_to_ud("NNS")  ->  "NOUN"
    ptb_to_ud("NNP")  ->  "PROPN"
    ptb_to_ud("VBD")  ->  "VERB"
    ptb_to_ud("ZZZ")  ->  "X"      (неизвестный тег)

    В PTB 36 тегов и они английские, в UD 17 и они общие для всех языков.
    Отображение одностороннее: из UD обратно в PTB не восстановить, число и
    время потеряны.

    Ловушка: NNP и NNPS — это PROPN, а не NOUN. Проверь порядок сравнений,
    иначе префикс "NN" съест имена собственные.
    """
    # порядок важен: сначала самые длинные префиксы
    if tag.startswith("NNP"):
        return "PROPN"
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("RB"):
        return "ADV"
    if tag in ("DT", "PDT", "WDT"):
        return "DET"
    if tag == "IN":
        return "ADP"
    if tag in ("PRP", "PRP$", "WP", "WP$"):
        return "PRON"
    if tag == "CC":
        return "CCONJ"
    if tag == "CD":
        return "NUM"
    if tag in (".", ",", ":", "``", "''", "-LRB-", "-RRB-"):
        return "PUNCT"
    return "X"


def train_mft(train_examples):
    """Обучение baseline «самый частый тег»: для слова берём его частый тег.

    Вход — список пар (tokens, tags). Ответ — пара (word_best, default_tag),
    где word_best это dict нижний_регистр_слова -> тег, а default_tag — самый
    частый тег корпуса, им размечаются слова, которых в обучении не было.

    train_mft([(["The", "cat"], ["DET", "NOUN"])])  ->  ({"the": "DET", "cat": "NOUN"}, "DET")
    train_mft([])                                   ->  ({}, None)

    Ничьи разрешай детерминированно: при равных счётчиках бери тег, который
    меньше по алфавиту. Иначе тесты будут краснеть через раз в зависимости от
    порядка обхода словаря.

    На Brown этот baseline даёт ~85% — пол, ниже которого не должна падать ни
    одна серьёзная модель.
    """
    counts = {}
    tag_totals = {}
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word = token.lower()
            counts.setdefault(word, {})
            counts[word][tag] = counts[word].get(tag, 0) + 1
            tag_totals[tag] = tag_totals.get(tag, 0) + 1

    # ключ (-count, tag): сначала по убыванию частоты, потом по алфавиту
    word_best = {w: min(c, key=lambda t: (-c[t], t)) for w, c in counts.items()}
    default_tag = None
    if tag_totals:
        default_tag = min(tag_totals, key=lambda t: (-tag_totals[t], t))
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    """Разметка baseline'ом: смотрим слово в таблице, иначе default_tag.

    predict_mft(["The", "cat"], {"the": "DET", "cat": "NOUN"}, "NOUN")  ->  ["DET", "NOUN"]
    predict_mft(["zzz"], {"the": "DET"}, "NOUN")                        ->  ["NOUN"]

    Регистр: таблица обучалась на нижнем регистре, значит и здесь слово надо
    привести к нижнему. Забыть про это — классический способ получить 40%
    точности на тексте, где первое слово предложения с большой буквы.
    """
    return [word_best.get(t.lower(), default_tag) for t in tokens]


def tag_accuracy(predicted, gold):
    """Доля совпавших тегов. Обе последовательности одной длины.

    tag_accuracy(["DET", "NOUN"], ["DET", "NOUN"])  ->  1.0
    tag_accuracy(["DET", "VERB"], ["DET", "NOUN"])  ->  0.5
    tag_accuracy([], [])                            ->  0.0

    При разной длине бросай ValueError: молча обрезать по zip нельзя, так
    рождаются отчёты с точностью 97% на модели, которая теряет половину слов.
    """
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must have the same length")
    if not gold:
        return 0.0
    hits = sum(1 for p, g in zip(predicted, gold) if p == g)
    return hits / len(gold)


def count_hmm(train_examples):
    """Счётчики для биграммной HMM: переходы тег->тег и эмиссии тег->слово.

    Ответ — кортеж (transitions, emissions, tags, vocab):
      transitions[prev_tag][tag]  сколько раз tag шёл после prev_tag,
                                  начало предложения — псевдотег "<BOS>",
                                  конец — "<EOS>";
      emissions[tag][word]        сколько раз tag породил это слово (в нижнем
                                  регистре);
      tags                        отсортированный список настоящих тегов
                                  (без "<BOS>" и "<EOS>");
      vocab                       отсортированный список слов.

    count_hmm([(["The", "cat"], ["DET", "NOUN"])])
      ->  ({"<BOS>": {"DET": 1}, "DET": {"NOUN": 1}, "NOUN": {"<EOS>": 1}},
           {"DET": {"the": 1}, "NOUN": {"cat": 1}},
           ["DET", "NOUN"],
           ["cat", "the"])

    tags и vocab возвращаем списками, а не множествами: порядок тегов задаёт
    порядок столбцов решётки Витерби, а от множества воспроизводимого порядка
    не дождёшься.
    """
    transitions = {}
    emissions = {}
    tags = set()
    vocab = set()

    for tokens, tag_seq in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, tag_seq):
            word = token.lower()
            transitions.setdefault(prev, {})
            transitions[prev][tag] = transitions[prev].get(tag, 0) + 1
            emissions.setdefault(tag, {})
            emissions[tag][word] = emissions[tag].get(word, 0) + 1
            tags.add(tag)
            vocab.add(word)
            prev = tag
        transitions.setdefault(prev, {})
        transitions[prev]["<EOS>"] = transitions[prev].get("<EOS>", 0) + 1

    return transitions, emissions, sorted(tags), sorted(vocab)


def laplace_logprob(counts, key, n_outcomes, alpha=0.01):
    """Логарифм сглаженной по Лапласу вероятности: (c + alpha) / (N + alpha*K).

    counts — словарь исход -> счётчик, n_outcomes (K) — сколько исходов вообще
    возможно, включая ни разу не встретившиеся.

    laplace_logprob({"a": 3, "b": 1}, "a", 4, alpha=1.0)  ->  log(4/8) = -0.693...
    laplace_logprob({"a": 3, "b": 1}, "c", 4, alpha=1.0)  ->  log(1/8) = -2.079...

    Зачем логарифм: вероятности предложения из 30 слов перемножаются в 1e-60 и
    схлопываются в ноль двойной точности. Сумма логарифмов — то же самое, но
    без потери разрядов.

    Сглаживание обязательно: без него одно незнакомое слово даёт log(0) и
    весь путь получает -inf.
    """
    total = sum(counts.values())
    return math.log((counts.get(key, 0) + alpha) / (total + alpha * n_outcomes))


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    """Декодирование Витерби: самая вероятная последовательность тегов.

    Динамика по решётке: V[i][j] — лучший логарифм вероятности пути, который
    заканчивается тегом tags[j] на позиции i. Переход считается по
    transitions, эмиссия — по emissions, обе через laplace_logprob.

    Число исходов для сглаживания: у переходов len(tags) + 1 (все теги плюс
    "<EOS>"), у эмиссий len(vocab) + 1 (весь словарь плюс слот на незнакомое
    слово).

    viterbi([], ...)  ->  []

    Возвращает список тегов той же длины, что tokens.

    Сложность O(n * |tags|^2), и это ровно то, ради чего динамика: перебор
    всех |tags|^n путей на предложении из 20 слов и 12 тегов — это 10^21.

    Хвостовой переход в "<EOS>" в счёте пути здесь не участвует, как и в коде
    урока: он влияет только на знаменатель сглаживания.
    """
    n = len(tokens)
    if n == 0:
        return []

    tag_list = list(tags)
    n_tags = len(tag_list)
    n_trans = n_tags + 1  # теги + "<EOS>"
    n_emit = len(vocab) + 1  # словарь + незнакомое слово

    V = [[0.0] * n_tags for _ in range(n)]
    back = [[0] * n_tags for _ in range(n)]

    for j, tag in enumerate(tag_list):
        tr = laplace_logprob(transitions.get("<BOS>", {}), tag, n_trans, alpha)
        em = laplace_logprob(emissions.get(tag, {}), tokens[0].lower(), n_emit, alpha)
        V[0][j] = tr + em

    for i in range(1, n):
        word = tokens[i].lower()
        for j, tag in enumerate(tag_list):
            em = laplace_logprob(emissions.get(tag, {}), word, n_emit, alpha)
            best_k, best_score = 0, float("-inf")
            for k, prev_tag in enumerate(tag_list):
                score = V[i - 1][k] + laplace_logprob(
                    transitions.get(prev_tag, {}), tag, n_trans, alpha
                )
                if score > best_score:
                    best_score, best_k = score, k
            V[i][j] = best_score + em
            back[i][j] = best_k

    # обратный проход по указателям: путь собирается с конца
    last = max(range(n_tags), key=lambda j: V[n - 1][j])
    path = [last]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tag_list[j] for j in reversed(path)]


def extract_svo(tokens, arcs):
    """Тройки (подлежащее, сказуемое, дополнение) из дерева зависимостей.

    arcs — список рёбер (head_index, dep_index, relation); у корня head_index
    равен -1. Берём каждый узел, у которого есть и ребёнок с отношением
    "nsubj", и ребёнок с "dobj", и выдаём тройку слов. Порядок ответа — по
    индексу сказуемого.

    tokens = ["cats", "eat", "fish"]
    arcs   = [(-1, 1, "ROOT"), (1, 0, "nsubj"), (1, 2, "dobj")]
    extract_svo(tokens, arcs)  ->  [("cats", "eat", "fish")]

    Глагол без дополнения тройку не даёт: extract_svo(["cats", "sleep"],
    [(-1, 1, "ROOT"), (1, 0, "nsubj")])  ->  []

    Это в двух строках то, ради чего в проде дёргают spaCy: `token.dep_` и
    `token.head` дают ровно такие рёбра, а извлечение фактов из текста
    начинается с этих троек.
    """
    subj = {}
    obj = {}
    for head, dep, rel in arcs:
        if rel == "nsubj":
            subj.setdefault(head, dep)
        elif rel == "dobj":
            obj.setdefault(head, dep)

    triples = []
    for verb in sorted(set(subj) & set(obj)):
        triples.append((tokens[subj[verb]], tokens[verb], tokens[obj[verb]]))
    return triples

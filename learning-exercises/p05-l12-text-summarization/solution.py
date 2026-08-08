"""
Суммаризация текста — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import re
from collections import Counter


def sentence_split(text):
    """Разбить текст на предложения по .!? — без выбрасывания знаков.

    sentence_split("Hi there. How are you?")  ->  ["Hi there.", "How are you?"]
    sentence_split("   ")                     ->  []

    Пустые куски выкидываются: иначе пустая строка станет «предложением» и
    попадёт в экстрактивное резюме.

    Знаки препинания остаются на месте намеренно: экстрактивная
    суммаризация возвращает предложения ДОСЛОВНО, а обрезанная точка — уже
    не дословно.
    """
    # lookbehind: режем ПОСЛЕ знака конца предложения, сам знак остаётся
    # в левом куске
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part]


def similarity(first, second):
    """Похожесть двух предложений по TextRank: пересечение слов, нормированное
    на логарифмы длин.

    similarity("the cat sat", "the cat ran")  ->  положительное число
    similarity("a b", "c d")                  ->  0.0

    Нормировка логарифмом нужна, чтобы длинные предложения не побеждали
    просто потому, что у них больше слов: то же одно совпадение в длинной
    паре весит меньше.

    Пересечение считаем как multiset через Counter, регистр не важен. Но в
    знаменателе стоят полные длины предложений в токенах, включая повторы,
    как в формуле TextRank, а не размеры словарей различных слов.
    """
    left = Counter(first.lower().split())
    right = Counter(second.lower().split())
    # (left & right) — покомпонентный минимум, то есть честное пересечение
    # с учётом повторов
    intersection = sum((left & right).values())
    denominator = math.log(sum(left.values()) + 1) + math.log(sum(right.values()) + 1)
    if denominator == 0:
        return 0.0
    return intersection / denominator


def textrank_scores(sentences, damping=0.85, iterations=50, epsilon=1e-4):
    """PageRank по графу похожести предложений. Вернуть список весов.

    Узлы — предложения, рёбра — similarity. Вес предложения тем больше, чем
    сильнее оно связано со всем остальным текстом.

    len(textrank_scores(sentences)) == len(sentences)
    textrank_scores([])  ->  []

    damping=0.85 и epsilon — обычные значения PageRank. Итерации
    прерываются досрочно, как только веса перестали меняться.

    Ловушка: предложение без единого соседа делит на сумму 0. Сумма
    подстраховывается крошечным числом.
    """
    n = len(sentences)
    if n == 0:
        return []
    # матрица похожести, диагональ нулевая: предложение не голосует за себя
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        converged = max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon
        scores = new_scores
        if converged:
            break
    return scores


def textrank_summary(text, top_k=3):
    """Экстрактивное резюме: top_k предложений с наибольшим весом TextRank,
    выданных в исходном порядке.

    Если предложений не больше top_k, возвращается весь текст как есть.

    Ключевое свойство: каждое предложение ответа ДОСЛОВНО встречается в
    исходном тексте. Экстрактивная суммаризация не умеет галлюцинировать
    по построению — за это её и держат в регулируемых доменах.

    Порядок восстанавливается по индексу, а не по весу: иначе резюме
    читается задом наперёд.
    """
    sentences = sentence_split(text)
    if len(sentences) <= top_k:
        return sentences
    scores = textrank_scores(sentences)
    # сначала отбираем по весу, потом сортируем по позиции — два разных
    # критерия, их нельзя объединить в одну сортировку
    chosen = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:top_k]
    chosen.sort()
    return [sentences[i] for i in chosen]


def lcs_length(first, second):
    """Длина наибольшей общей ПОДПОСЛЕДОВАТЕЛЬНОСТИ двух списков токенов.

    lcs_length(["a", "b", "c"], ["a", "b", "c"])  ->  3
    lcs_length(["a", "x", "b"], ["a", "y", "b"])  ->  2

    Подпоследовательность, а не подстрока: между совпадениями разрешены
    любые пропуски, поэтому вставленное слово не обнуляет счёт.

    Классическая динамика O(n*m). Хранить всю таблицу не нужно — хватает
    двух строк.
    """
    if not first or not second:
        return 0
    previous = [0] * (len(second) + 1)
    for a in first:
        current = [0]
        for j, b in enumerate(second):
            if a == b:
                current.append(previous[j] + 1)
            else:
                current.append(max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


def rouge_n(candidate, reference, n=1):
    """ROUGE-N: (precision, recall, fmeasure) по совпадению n-грамм.

    На вход списки токенов. Совпадения обрезаются по числу вхождений в
    референсе, как в BLEU.

    rouge_n(t, t)               ->  (1.0, 1.0, 1.0)
    rouge_n(["a"], ["b"])       ->  (0.0, 0.0, 0.0)

    ROUGE называют recall-oriented: главный вопрос «сколько референса
    покрыто», а не «сколько лишнего написано». Поэтому короткое резюме
    получает высокий precision и низкий recall.

    Ловушка: перестановка слов не трогает ROUGE-1, но убивает ROUGE-2 —
    биграммы держат порядок.
    """
    cand_grams = Counter(
        tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1)
    )
    ref_grams = Counter(
        tuple(reference[i : i + n]) for i in range(len(reference) - n + 1)
    )
    cand_total = sum(cand_grams.values())
    ref_total = sum(ref_grams.values())
    overlap = sum(min(count, ref_grams[gram]) for gram, count in cand_grams.items())
    precision = overlap / cand_total if cand_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def rouge_l(candidate, reference):
    """ROUGE-L: (precision, recall, fmeasure) по длине LCS.

    rouge_l(t, t)  ->  (1.0, 1.0, 1.0)

    В отличие от ROUGE-2, вставленное посреди фразы слово почти ничего не
    стоит: подпоследовательность спокойно его перешагивает. Поэтому
    ROUGE-L мягче к перефразированию и жёстче к перестановкам.
    """
    lcs = lcs_length(candidate, reference)
    precision = lcs / len(candidate) if candidate else 0.0
    recall = lcs / len(reference) if reference else 0.0
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def hallucinated_entities(source, summary):
    """Сущности, которые есть в резюме, но которых нет в источнике.

    Наивный NER без библиотек: сущность — это токен с заглавной первой
    буквой или токен с цифрой (числа тоже «дрейфуют»: 25 000 превращается
    в 25 миллионов). Первое слово предложения пропускается, иначе каждое
    «The» станет сущностью.

    hallucinated_entities("... John Smith ...", "... John Brown ...")
        ->  ["Brown"]

    Ответ отсортирован — прогон обязан быть воспроизводимым.

    Это ручная версия entity-level factuality из урока: настоящий пайплайн
    берёт сущности из spaCy, но вопрос тот же — что модель дописала от себя.
    Пустой ответ означает «новых сущностей нет», а не «резюме правдиво».
    """

    def extract(text):
        found = []
        for sentence in sentence_split(text):
            # первое слово предложения всегда с большой буквы — оно не улика
            for token in sentence.split()[1:]:
                clean = token.strip(".,!?;:\"'()[]")
                if not clean:
                    continue
                if clean[0].isupper() or any(ch.isdigit() for ch in clean):
                    found.append(clean)
        return found

    known = set(extract(source))
    return sorted({token for token in extract(summary) if token not in known})

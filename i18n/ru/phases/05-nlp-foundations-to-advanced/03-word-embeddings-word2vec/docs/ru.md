<!-- i18n:manual -->
# Word embeddings — Word2Vec с нуля

> Слово — это его окружение. Обучите на этой мысли мелкую сеть, и геометрия появится сама.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## The Problem

TF-IDF знает, что `dog` и `puppy` — разные слова. Он не знает, что они значат почти одно и то же. Классификатор, обученный на `dog`, не обобщится на отзыв про `puppy`. Можно заклеить дыру списком синонимов, но он развалится на редких терминах, отраслевом жаргоне и любом языке, который вы не предусмотрели.

Вам нужно представление, где `dog` и `puppy` оказываются рядом в пространстве. Где `king - man + woman` попадает близко к `queen`. Где модель, обученная на `dog`, бесплатно переносит часть сигнала на `puppy`.

Word2Vec дал нам это пространство. Двухслойная нейросеть, обучение на триллионах tokens, публикация 2013 года. Архитектура почти неприлично простая. Результат перекроил NLP на десятилетие.

> 🎒 **На пальцах.** В TF-IDF слова `dog` и `puppy` — это две разные оси, между ними ровно ноль связи: их косинусная близость равна 0.0, как у `dog` и `банана`. После Word2Vec `dog` и `puppy` получают близость порядка 0.8, а `dog` и `банан` — около 0.1. Ничего кроме сырого текста мы модели не сообщали.

## The Concept

**Distributional hypothesis** (Firth, 1957): «слово узнаётся по компании, которую оно водит». Если два слова встречаются в похожих контекстах, они, скорее всего, значат похожее.

> 🎒 **На пальцах.** Представьте, что вы не знаете слова «мурлыкать», но встретили его в 200 предложениях, и рядом стоят «кошка», «на коленях», «тёплый», «уснул». Значения вам никто не сказал — но соседи выдали его с потрохами. Word2Vec ровно этим и занимается, только на миллиардах предложений.

Word2Vec существует в двух вариантах, оба эксплуатируют эту идею.

- **Skip-gram.** По центральному слову предсказываем соседей. `cat -> (the, sat, on)` при окне размера 2.
- **CBOW (continuous bag of words).** По соседям предсказываем центральное слово. `(the, sat, on) -> cat`.

Skip-gram обучается медленнее, но лучше справляется с редкими словами. Он и стал вариантом по умолчанию.

> 🎒 **На пальцах.** Разница в количестве обучающих примеров. Для окна 2 одно центральное слово в skip-gram даёт до 4 отдельных примеров (по одному на соседа), а CBOW из тех же соседей делает 1 пример. Поэтому редкое слово, встретившееся в corpus 10 раз, в skip-gram получит около 40 обновлений вместо 10 — отсюда и преимущество на редких словах.

Сеть имеет один скрытый слой без нелинейности. Вход — one-hot вектор по словарю. Выход — softmax по словарю. После обучения выходной слой выбрасывается. Веса скрытого слоя и есть embeddings.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

> 🎒 **На пальцах.** Умножение one-hot вектора на матрицу `W` — это не умножение, а выбор строки: везде нули, кроме одной единицы, поэтому от всей матрицы остаётся ровно одна строка. При словаре 10 000 и dim 100 матрица `W` содержит 1 000 000 чисел, а на каждое слово мы достаём из неё 100. Эта строка и есть embedding слова, всё остальное — леса, которые потом снесут.

Хитрость: softmax по 100 000 слов недопустимо дорог. Word2Vec использует **negative sampling**, чтобы превратить задачу в бинарную классификацию. Предсказываем «встречалось ли это контекстное слово рядом с этим центральным, да или нет». На каждую обучающую пару берём горстку negative (не встречавшихся рядом) слов вместо подсчёта softmax по всему словарю.

> 🎒 **На пальцах.** Считаем экономию. Полный softmax по словарю 100 000 требует 100 000 скалярных произведений на каждую пару. Negative sampling с k=5 требует 6: одно положительное и пять отрицательных. Это примерно в 16 000 раз меньше работы на шаг — разница между «обучаемся год» и «обучаемся к вечеру».

```figure
word-vector-arithmetic
```

## Build It

### Step 1: training pairs from a corpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Каждая пара (центральное, контекстное) внутри окна — это положительный обучающий пример.

> 🎒 **На пальцах.** Посчитайте пары для `["the","cat","sat","on","mat"]` при окне 2. У `the` соседей 2, у `cat` — 3, у `sat` — 4 (окно целиком помещается), у `on` — 3, у `mat` — 2. Итого 14 пар из 5 слов. Слова в середине предложения дают больше примеров, чем слова по краям, — это нормально и в больших corpus не имеет значения.

### Step 2: embedding tables

Две матрицы. `W` — таблица embeddings центральных слов (та, которую вы оставите). `W'` — таблица контекстных слов (обычно выбрасывается, иногда усредняется с `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Инициализация мелкими случайными числами. Размер словаря 10 000 и dim 100 — реалистично; для учёбы хватит 50 слов на 16 измерений, чтобы увидеть геометрию.

> 🎒 **На пальцах.** На учебном масштабе 50 × 16 = 800 чисел в каждой таблице, 1600 всего — это влезет в один экран. На реалистичном 10 000 × 100 = 1 000 000 чисел в каждой, 2 миллиона параметров. При этом сохранить вы собираетесь только половину: `W'` после обучения идёт в мусор. Половина работы делается ради того, чтобы её выбросить.

### Step 3: negative sampling objective

Для каждой положительной пары `(center, context)` берём `k` случайных слов из словаря как negatives. Обучаем модель так, чтобы скалярное произведение `W[center] · W'[context]` было большим для положительных пар и малым для отрицательных.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Формула-магия: логистическая потеря на положительной паре (хотим sigmoid близко к 1) плюс логистическая потеря на отрицательных парах (хотим sigmoid близко к 0). Градиенты текут в обе таблицы. Полный вывод есть в оригинальной статье; пройдите его один раз с карандашом, если хотите, чтобы он улёгся.

> 🎒 **На пальцах.** Посмотрите на `(pos_score - 1)` в градиенте. Если модель уже уверена и `pos_score` = 0.99, множитель равен −0.01 — шаг почти нулевой, трогать нечего. Если модель ошибается и `pos_score` = 0.1, множитель равен −0.9, шаг в 90 раз больше. Обучение само распределяет усилия туда, где ещё плохо. И `np.clip(x, -20, 20)` в sigmoid — не украшение: без него `np.exp(1000)` даст переполнение и весь массив станет `nan`.

### Step 4: train on a toy corpus

```python
def sample_negatives(rng, vocab_size, k_neg, forbidden):
    if vocab_size <= len(forbidden):
        return []
    negs = []
    while len(negs) < k_neg:
        draw = rng.integers(0, vocab_size, size=k_neg)
        negs.extend(int(n) for n in draw if n not in forbidden)
    return negs[:k_neg]


def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = sample_negatives(rng, vocab_size, k_neg, {c_idx, ctx_idx})
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

`sample_negatives` досэмплирует, пока не наберёт ровно `k_neg` негативов. Соблазнительный короткий путь — насэмплировать один раз и потом выкинуть центральное и контекстное слово, — но тогда одни пары молча обучаются против четырёх негативов, другие против трёх, а иногда и вовсе ни против одного: `k_neg`, который вы передали, перестаёт быть тем `k_neg`, который вы получили.

После достаточного числа эпох на большом corpus слова, делящие контексты, получают похожие центральные embeddings. На игрушечном corpus эффект едва заметен. На миллиардах tokens — заметен драматически.

> 🎒 **На пальцах.** Посчитайте объём работы для одного предложения из 5 слов: 14 пар × 100 эпох = 1400 обновлений, и на каждое приходится 1 положительный и ровно 5 отрицательных примеров (за это и отвечает `sample_negatives`), то есть 8400 скалярных произведений. Звучит много — но это меньше, чем один шаг обучения трансформера. Word2Vec на corpus в миллион слов спокойно обучается на ноутбуке.

### Step 5: the analogy trick

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

На предобученных 300-мерных векторах Google News:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Не потому, что модель знает, что такое монархия. А потому, что вектор `(king - man)` кодирует нечто вроде «королевскости», и прибавление его к `woman` приземляет результат в область «королевская особа женского пола».

> 🎒 **На пальцах.** Обратите внимание на числа: `queen` — 0.71, `monarch` — 0.62, `princess` — 0.59. Это косинусные близости, то есть косинус угла между векторами. 0.71 — это примерно 44 градуса: не совпадение, а «в ту сторону». И заметьте, что сами `man`, `king`, `woman` из ответа исключены через `exclude`, иначе они бы заняли весь топ — арифметика приземляет результат совсем рядом с ними.

## Use It

Писать Word2Vec с нуля — это учёба. В продакшене NLP берут `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

> 🎒 **На пальцах.** Разберите параметры по смыслу: `sg=1` включает skip-gram (`sg=0` дал бы CBOW), `negative=5` — те самые 5 negatives на пару, `window=5` даёт до 10 соседей на слово вместо 4, `min_count=1` оставляет в словаре даже слова, встреченные один раз. На настоящих данных `min_count=5` — гораздо разумнее: слова-однодневки просто не наберут обновлений, чтобы обучиться.

Для реальной работы вы почти никогда не обучаете Word2Vec сами. Вы скачиваете предобученные векторы.

- **GloVe** — стэнфордский подход через факторизацию матрицы совстречаемости. Чекпойнты на 50d, 100d, 200d, 300d. Хорошее общее покрытие. GloVe отдельно разбирается в уроке 04.
- **fastText** — расширение Word2Vec от Facebook, которое встраивает символьные n-граммы. Справляется со словами вне словаря, собирая их из subword-кусков. Урок 04.
- **Pretrained Word2Vec on Google News** — 300d, словарь на 3 миллиона слов, опубликован в 2013 году. Скачивают до сих пор ежедневно.

> 🎒 **На пальцах.** Прикиньте размер Google News-векторов: 3 000 000 слов × 300 чисел = 900 миллионов чисел, по 4 байта каждое, то есть примерно 3.6 гигабайта. Отсюда и совет качать 100d вместо 300d, если качество терпит: файл сразу худеет втрое.

### When Word2Vec still wins in 2026

- Лёгкий доменный поиск. Обучите на медицинских абстрактах за час на ноутбуке — получите специализированные векторы, которых нет ни в одной общей модели.
- Инженерия признаков через аналогии. `gender_vector = mean(man - woman pairs)`. Вычтите его из других слов, чтобы получить гендерно-нейтральную ось. До сих пор применяется в исследованиях справедливости.
- Интерпретируемость. 100 измерений достаточно мало, чтобы нарисовать через PCA или t-SNE и правда увидеть, как складываются кластеры.
- Везде, где инференс идёт на устройстве без GPU. Достать Word2Vec-вектор — это выборка одной строки из таблицы.

> 🎒 **На пальцах.** Сравните стоимость инференса. Word2Vec: чтение 300 чисел по индексу, наносекунды, ноль умножений. BERT: прогон 110 миллионов параметров, десятки миллисекунд, желательно GPU. Если задача — «найти похожие товары по названию» на телефоне, разница решает всё.

### Where Word2Vec fails

Стена многозначности. У `bank` один вектор. `river bank` и `financial bank` делят его на двоих. У `table` (таблица против мебели) он тоже один. Классификатор дальше по конвейеру не может различить значения по такому вектору.

Контекстные embeddings (ELMo, BERT, любой трансформер после них) решили это, выдавая для каждого вхождения слова свой вектор в зависимости от окружения. Это и есть скачок от Word2Vec к BERT: от статических представлений к контекстным. Трансформерную половину разбирает Phase 7.

> 🎒 **На пальцах.** Один вектор на слово — значит, обучение усредняет все значения в одну точку. Если `bank` в corpus в 80% случаев финансовый и в 20% речной, вектор осядет где-то на 80% пути к финансовому смыслу. Предложение про рыбалку на берегу получит вектор, который в основном про деньги. Контекстный embedding вместо одной точки выдаёт разную для каждого вхождения.

Второй способ сломаться — слова вне словаря. Word2Vec никогда не видел `Zoomer-approved`, если этого не было в обучающих данных. Запасного варианта нет. fastText чинит это композицией из subword-кусков (урок 04).

> 🎒 **На пальцах.** Разница подходов на одном слове. Word2Vec ищет `Zoomer-approved` в словаре из 3 миллионов записей, не находит и выбрасывает исключение — вектора нет вообще. fastText разбирает слово на символьные n-граммы вроде `Zoo`, `oom`, `ome`, `app`, суммирует их векторы и выдаёт хоть какое-то осмысленное представление. Ноль информации против частичной — большая разница.

## Ship It

Сохраните как `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Exercises

1. **Easy.** Прогоните цикл обучения на крошечном corpus (20 предложений про кошек и собак). После 200 эпох проверьте, что `nearest(vocab, W, W[vocab["cat"]])` возвращает `dog` в топ-3. Если нет — увеличьте число эпох или словарь.
2. **Medium.** Добавьте подвыборку частых слов. Слова с частотой выше `10^-5` выбрасываются из обучающих пар с вероятностью, пропорциональной их частоте. Измерьте эффект на похожести редких слов.
3. **Hard.** Обучите модель на corpus 20 Newsgroups. Постройте две оси смещения: `he - she` и `doctor - nurse`. Спроецируйте названия профессий на обе оси. Опишите, у каких профессий разрыв смещения самый большой. Именно такие пробы используют исследователи справедливости.

> 🎒 **На пальцах.** Подсказка к первому заданию: 20 предложений — это примерно 120 слов, при окне 2 около 400 пар, а за 200 эпох — около 80 000 обновлений. Этого хватает только если `cat` и `dog` реально стоят в похожих контекстах: пишите предложения вроде «the cat ran» и «the dog ran», а не про разное. Если `dog` не появился в топ-3, проблема почти всегда в corpus, а не в числе эпох.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | «Слово как вектор» | Плотное низкоразмерное (обычно 100-300) представление, выученное из контекста. |
| Skip-gram | «Фишка Word2Vec» | Предсказание контекстных слов по центральному. Медленнее CBOW, лучше на редких словах. |
| Negative sampling | «Ускоритель обучения» | Замена softmax по всему словарю на бинарную классификацию против `k` случайных слов. |
| Static embedding | «Один вектор на слово» | Один и тот же вектор независимо от контекста. Ломается на многозначности. |
| Contextual embedding | «Вектор с оглядкой на контекст» | Свой вектор для каждого вхождения, зависящий от соседних слов. Это то, что выдают трансформеры. |
| OOV | «Слова нет в словаре» | Слово не встречалось при обучении. Word2Vec не может дать для него вектор. |

## Further Reading

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — статья про negative sampling. Короткая и читаемая.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — самый ясный вывод градиентов, если математика оригинальной статьи кажется плотной.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — настройки обучения, которые действительно работают в продакшене.

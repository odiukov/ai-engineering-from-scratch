<!-- i18n:manual -->
# GloVe, FastText и subword-эмбеддинги

> Word2Vec обучал по одному эмбеддингу на слово. GloVe разложил матрицу совместной встречаемости. FastText встроил кусочки слов. BPE стал мостом к трансформерам.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## The Problem

Word2Vec оставил два открытых вопроса.

Первый. Параллельно существовало целое направление, которое раскладывало матрицу совместной встречаемости напрямую (LSA, HAL), а не делало пошаговые skip-gram обновления. Итеративный подход Word2Vec действительно лучше — или разница просто в том, как два метода обращались со счётчиками? **GloVe** ответил: матричное разложение с грамотно подобранной функцией потерь не уступает Word2Vec, а иногда и выигрывает, и обучается дешевле.

Второй. Ни один из методов не знал, что делать со словами, которых он никогда не видел. `Zoomer-approved`, `dogecoin`, любое имя собственное, придуманное на прошлой неделе, каждая словоформа редкого корня. **FastText** это починил, обучая эмбеддинги символьных n-gram: слово — это сумма его кусочков, включая морфемы, поэтому даже OOV-слово получает осмысленный вектор.

Третий. Когда пришли трансформеры, вопрос сменился снова. Словарь на уровне слов упирается примерно в миллион записей; живой язык открытее. **Byte-pair encoding (BPE)** и его родственники решили это, выучив словарь частых subword-единиц, который покрывает вообще всё. Каждый современный токенизатор каждой современной LLM — subword-токенизатор.

Этот урок проходит все три подхода и объясняет, за каким тянуться в какой ситуации.

> 🎒 **На пальцах.** Представьте словарь иностранного языка в кармане. Word2Vec — это словарь, где каждое слово выписано целиком: увидели незнакомое — и всё, тупик. FastText — это словарь корней и приставок: «дог» + «коин» уже что-то значат по отдельности. BPE — набор слогов, которыми можно собрать любое слово вообще. Три способа не остаться без ответа.

> 🎒 **На пальцах.** Почему словарь слов упирается в миллион: в него нужно вписать все формы, все опечатки, все имена. А байтов в BPE-словаре стартово всего 256 — и этого достаточно, чтобы записать любой текст на любом языке, просто более длинной цепочкой.

## The Concept

**GloVe (Global Vectors).** Строим матрицу совместной встречаемости слов `X`, где `X[i][j]` — как часто слово `j` попадается в контексте слова `i`. Обучаем векторы так, чтобы `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Взвешиваем функцию потерь, чтобы частые пары не задавили всё остальное. Готово.

> 🎒 **На пальцах.** Это как таблица «кто с кем сидел за столом» за весь год. GloVe не бегает по каждому обеду заново — он один раз смотрит на итоговую таблицу и подбирает векторы так, чтобы скалярное произведение предсказывало логарифм числа совместных обедов. Пара, встретившаяся 100 раз, даёт цель log(100) ≈ 4.6; пара, встретившаяся 1 раз, — цель log(1) = 0.

**FastText.** Слово — это сумма его символьных n-gram плюс само слово. `where` превращается в `<wh, whe, her, ere, re>, <where>`. Вектор слова — сумма векторов этих кусочков. Обучается как Word2Vec. Выигрыш: невиданное слово (`whereupon`) собирается из знакомых n-gram.

> 🎒 **На пальцах.** Угловые скобки `<` и `>` — это метки начала и конца слова. Благодаря им кусочек `<wh` (начало слова) отличается от `wh` в середине. Поэтому `where` и `whereupon` делят начало `<wh` и `<whe`, но `where` при этом не путается с `nowhere`.

**BPE (Byte-Pair Encoding).** Начинаем со словаря отдельных байтов (или символов). Считаем каждую соседнюю пару в корпусе. Сливаем самую частую пару в новый токен. Повторяем `k` раз. Результат: словарь из `k + 256` токенов, где частые последовательности (`ing`, `tion`, `the`) — цельные токены, а редкие слова разбиваются на знакомые кусочки. Любое предложение как-нибудь да токенизируется.

> 🎒 **На пальцах.** Похоже на стенографию. Сначала вы пишете по буквам. Замечаете, что «ing» встречается всё время, и придумываете для него один значок. Потом «tion». Через 30 000 таких значков вы пишете обычный текст втрое короче, а незнакомое слово всё равно можете записать по буквам — просто длиннее.

```figure
n5-subword-merge
```

## Build It

### GloVe: factorize the co-occurrence matrix

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            # В точной производной f(x) * diff^2 есть множитель 2.
            # Здесь он спрятан внутрь `lr`, поэтому `coef` — это градиент
            # с точностью до этой константы, а не сам градиент.
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Две движущиеся детали, которые стоит назвать. Весовая функция `f(x) = (x/x_max)^alpha` уменьшает вклад очень частых пар (вроде `(the, and)`), чтобы они не задавили функцию потерь. Итоговый эмбеддинг — сумма таблиц `W` (центр) и `W_tilde` (контекст). Складывать обе — опубликованный трюк, который обычно работает лучше, чем брать одну.

> 🎒 **На пальцах.** Подставьте числа в весовую функцию при `x_max=100` и `alpha=0.75`. Пара, встретившаяся 10 раз: (10/100)^0.75 ≈ 0.18. Пара, встретившаяся 100 раз и чаще: вес ровно 1.0. То есть редкая пара влияет на обучение впятеро слабее частой, но не исчезает совсем.

> 🎒 **На пальцах.** Обратите внимание на строчку `pair_counts[(center, indexed[j])] += 1.0 / distance`. Соседнее слово (расстояние 1) добавляет 1.0, а слово на краю окна из пяти (расстояние 5) — только 0.2. Логика бытовая: чем дальше слово, тем меньше оно про вас говорит.

### FastText: subword-aware embeddings

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>', '<where', 'where>'}
```

Пятнадцать кусочков на слово из пяти букв: обёрнутая форма целиком плюс все 3-, 4-, 5- и 6-граммы строки `<where>`. Обратите внимание, что `where` (5-грамма внутри скобок) и `<where>` (слово целиком) — это две разные записи, и так задумано: FastText хочет отдельную ячейку под само слово вдобавок к ячейкам под его части.

Каждое слово представлено набором своих n-gram (обычно от 3 до 6 символов). Эмбеддинг слова — сумма эмбеддингов его n-gram. Для skip-gram обучения подставьте это туда, где Word2Vec использовал один вектор.

> 🎒 **На пальцах.** Разберём `<where>` вручную. Триграммы, то есть кусочки по 3 символа: `<wh`, `whe`, `her`, `ere`, `re>` — ровно пять, потому что в строке из 7 символов ровно 7 − 3 + 1 = 5 позиций для окна длины 3. Дальше то же самое для длин 4, 5 и 6: 7 − 4 + 1 = 4, 7 − 5 + 1 = 3 и 7 − 6 + 1 = 2, то есть ещё девять кусочков. Пять плюс девять — четырнадцать, и сверху отдельная запись на само `<where>` целиком: итого пятнадцать. Шестёрки `<where` и `where>` при подсчёте руками теряются чаще всего — если у вас вышло тринадцать, вы забыли ровно их.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

Для невиданного слова вы всё равно получаете вектор, если хотя бы часть его n-gram известна. `whereupon` делит с `where` кусочки `<wh`, `her`, `ere` и `<where`, поэтому оба слова оказываются рядом.

> 🎒 **На пальцах.** Именно поэтому FastText так хорош для русского. `кот`, `кота`, `коту`, `котом` для Word2Vec — четыре разных, никак не связанных слова. Для FastText они делят кусочек `кот` и получают почти одинаковые векторы бесплатно.

### BPE: learned subword vocabulary

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        # Самая частая пара, ничьи разрешаются лексикографически.
        # `most_common(1)` разрешил бы ничью по порядку вставки, и тот же
        # корпус в другом порядке выучил бы другой словарь.
        best = min(pair_freq.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Первая итерация сливает самую частую соседнюю пару. На маленьких корпусах ничьи по частоте — обычное дело, поэтому правило разрешения ничьих должно быть явным: иначе список слияний зависит от того, в каком порядке вам случилось прочитать корпус, и словарь перестаёт быть воспроизводимым. После достаточного числа итераций частые подстроки (`low`, `est`, `tion`) становятся отдельными токенами, а редкие слова разбиваются аккуратно.

> 🎒 **На пальцах.** Посчитаем первое слияние на этом корпусе руками. Пара `e`+`s` встречается в `newest` (6 раз) и в `widest` (3 раза) — итого 9. Пара `l`+`o` встречается в `low` (5) и `lower` (2) — итого 7. Девять больше семи, так что `lo` в первом раунде проигрывает. Но вот сюрприз: с частотой 9 идут сразу три пары — `e`+`s`, `s`+`t` и `t`+`</w>`, потому что все они живут в тех же самых `newest` и `widest` и ни разу нигде больше. Максимума мало, нужна ничья: ключ `(-kv[1], kv[0])` сначала берёт самую частую, а среди равных — лексикографически меньшую пару, то есть `('e', 's')`. Дальше к `es` приклеится `t`, потом `</w>` — так и рождается токен `est</w>`, который вы видите в ответе для слова `lowest`.

> 🎒 **На пальцах.** Заметьте: слова `lowest` в корпусе не было вообще. Но BPE собрал его из `low` и `est</w>` — двух кусочков, выученных на других словах. Ни ошибки, ни OOV, ни специального токена «неизвестно».

Настоящие токенизаторы GPT / BERT / T5 выучивают от 30 000 до 100 000 слияний. Результат: любой текст токенизируется в последовательность известных ID ограниченной длины, никакого OOV.

## Use It

На практике вы почти никогда не обучаете это сами. Вы загружаете готовые чекпоинты.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Для subword-токенизации в стиле BPE в эпоху трансформеров:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Префикс `Ġ` отмечает границы слов (соглашение GPT-2). Каждый современный токенизатор — это вариант BPE, WordPiece (BERT) или SentencePiece (T5, LLaMA).

> 🎒 **На пальцах.** Посмотрите на разбор: `unbelievably tokenized` — 22 символа и 6 токенов, то есть примерно 3.7 символа на токен. Именно поэтому счета за API считают в токенах, а не в словах: два «слова» здесь стоят как шесть. И заметьте, что `Ġtoken` начинается с пробела — токенизатор кодирует пробел внутрь токена, а не отдельно.

> 🎒 **На пальцах.** Строчка `ft.get_word_vector("zoomerapproved")` не упадёт, хотя такого слова нет ни в одном словаре. FastText соберёт вектор из кусочков `zoom`, `oomer`, `appro` и так далее. Word2Vec на этом месте выбросил бы KeyError.

### When to pick which

| Situation | Pick |
|-----------|------|
| Pretrained general-purpose word vectors, no OOV tolerance needed | GloVe 300d |
| Pretrained general-purpose word vectors, must handle misspellings / neologisms / morphologically rich languages | FastText |
| Anything going into a transformer (training or inference) | Тот токенизатор, с которым модель вышла. Никогда не меняйте. |
| Training your own language model from scratch | Сначала обучите BPE или SentencePiece токенизатор на своём корпусе |
| Production text classification with a linear model | По-прежнему TF-IDF. Урок 02. |

> 🎒 **На пальцах.** Строка про трансформеры — самая важная в таблице. Токенизатор и веса модели — это как замок и ключ: ID токена 15496 значит «Hello» только для того словаря, на котором модель училась. Подставьте чужой токенизатор — модель не упадёт с ошибкой, она просто начнёт выдавать бессмыслицу. Это самый тихий и самый частый баг в продакшене.

## Ship It

Сохраните как `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

> 🎒 **На пальцах.** Обратите внимание на последний абзац скилла: это не подсказки, а запреты. Хороший промпт-скилл описывает не только что делать, но и от чего отказываться — например, не советовать обучать свой токенизатор тому, кто просто дообучает готовую LLM. Запреты экономят больше времени, чем советы.

## Exercises

1. **Easy.** Запустите `char_ngrams("playing")` и `char_ngrams("played")`. Посчитайте коэффициент Жаккара для двух наборов n-gram. Вы увидите заметное пересечение (`pla`, `lay`, `play`) — именно поэтому FastText хорошо переносится между морфологическими вариантами.
2. **Medium.** Расширьте `learn_bpe`, чтобы он отслеживал рост словаря. Постройте график числа токенов на символ корпуса в зависимости от числа слияний. Вы увидите быстрое сжатие в начале, выходящее на асимптоту около 2-3 символов на токен.
3. **Hard.** Обучите BPE с 1000 слияний на полном собрании сочинений Шекспира. Сравните токенизацию частых слов и редких имён собственных. Измерьте среднее число токенов на слово до и после. Опишите, что вас удивило.

> 🎒 **На пальцах.** Подсказка к первому заданию. Коэффициент Жаккара — это размер пересечения, делённый на размер объединения. Если у `playing` и `played` нашлось 6 общих кусочков, а всего разных кусочков на двоих 26, то ответ 6/26 ≈ 0.23. Не пугайтесь маленького числа: важно не абсолютное значение, а то, что для пары случайных слов оно будет близко к нулю.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Co-occurrence matrix | Таблица частот «слово-слово» | `X[i][j]` = как часто слово `j` встречается в окне вокруг слова `i`. |
| Subword | Кусок слова | Символьная n-gram (FastText) или выученный токен (BPE/WordPiece/SentencePiece). |
| BPE | Byte-pair encoding | Итеративное слияние самых частых соседних пар, пока словарь не дорастёт до нужного размера. |
| OOV | Out of vocabulary | Слово, которого модель никогда не видела. Word2Vec/GloVe ломаются. FastText и BPE справляются. |
| Byte-level BPE | BPE поверх сырых байтов | Схема GPT-2. Словарь начинается с 256 байтов, поэтому OOV не бывает в принципе. |

## Further Reading

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — статья про GloVe, семь страниц, до сих пор лучший вывод функции потерь.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — статья, которая принесла BPE в современный NLP.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — чем BPE, WordPiece и SentencePiece реально отличаются на практике.

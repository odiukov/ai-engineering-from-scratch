<!-- i18n:manual -->
# Генерация текста до трансформеров — языковые модели на n-gram

> Если слово удивляет модель, модель плохая. Perplexity превращает удивление в число. Сглаживание не даёт этому числу уйти в бесконечность.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## The Problem

До трансформеров, до RNN, до word embeddings языковая модель предсказывала следующее слово простым счётом: сколько раз оно шло после предыдущих `n-1` слов. Посчитали: «the cat» → «sat» 47 раз, «the cat» → «jumped» 12 раз, «the cat» → «refrigerator» 0 раз. Нормализовали и получили распределение вероятностей.

Это и есть n-gram языковая модель. На ней работали все распознаватели речи, все проверки орфографии и все фразовые системы машинного перевода с 1980 по 2015 год. Она работает и сейчас, когда нужна дешёвая языковая модель прямо на устройстве.

> 🎒 **На пальцах.** Это записная книжка бармена: «после „the cat" гость 47 раз сказал „sat" и 12 раз „jumped"». Всего 59 наблюдений, значит вероятность «sat» = 47/59 ≈ 0.80, а «jumped» = 12/59 ≈ 0.20. Никакой магии — только счёт и деление.

Интересная задача — что делать с n-gram, которых не было в обучении. Модель на сырых счётчиках даёт нулевую вероятность всему, чего она не видела, и это катастрофа: предложения длинные, и почти в каждом длинном предложении есть хотя бы одна невиданная последовательность. Пятьдесят лет исследований сглаживания решили эту проблему. Kneser-Ney smoothing — итог этой работы, а современный deep learning унаследовал её эмпирическую традицию.

## The Concept

![N-gram model: count, smooth, generate](../assets/ngram.svg)

### The prediction game

Ещё до всей этой машинерии один эксперимент определил, что такое языковая модель. Закройте следующую букву английского предложения. Попросите человека угадывать её по одной букве за попытку, пока он не попадёт. Запишите номер попытки. Повторите несколько сотен раз.

Номера попыток — не забавная статистика. Это перекодировка текста без потерь: отдайте последовательность номеров второму такому же угадывающему, и он восстановит каждую букву, потому что в каждой позиции он точно знает, какие буквы называются первыми. Сообщение, которое можно перекодировать меньшим числом символов, несёт меньше информации на символ, поэтому статистика номеров попыток задаёт потолок энтропии английского языка.

> 🎒 **На пальцах.** Это «Поле чудес», только вы называете буквы по одной, пока не попадёте, и записываете номер попытки. Если после «сегодня хорошая пого» вы угадали «д» с первой попытки — записали 1. Последовательность единиц и двоек и есть сжатый текст: угадывать легко, значит информации мало.

Шеннон провёл этот эксперимент в 1951 году и получил число, которое до сих пор управляет всей областью. Алфавит из 27 символов (26 букв плюс пробел) мог бы нести `log2(27) ≈ 4.75` бита на букву. Люди, у которых было 100 букв контекста, укладывались в 0.6–1.3 бита на букву. Английский примерно на три четверти состоит из вынужденных ходов. Структуру, которую модель обязана выучить, измерили раньше, чем появилась хоть одна модель.

> 🎒 **На пальцах.** 4.75 бита — это цена буквы, если угадывать вслепую. Живые люди тратили 0.6–1.3 бита, то есть в четыре-семь раз меньше. Разница и есть то, что модель может выучить: три четверти текста предсказуемы заранее.

Каждая языковая модель с тех пор — механический игрок в эту игру, и каждое число оценки в этом уроке есть счёт в той же игре:

- **Cross-entropy loss** — среднее число бит, которое модели нужно на символ. Обучение языковой модели буквально минимизирует её счёт в игре в угадайку.
- **Perplexity** — это `2^bits` (или `e^nats`): коэффициент ветвления, который остаётся перед моделью после всех её догадок. Равномерное угадывание из 27 символов даёт perplexity 27; игрок с 1 битом на букву имеет perplexity 2.
- **Context length is the player's memory.** Триграммная модель играет с памятью в два токена. Трансформер играет в ту же игру со 100K токенов. Правила не менялись, просто игрок стал лучше.

> 🎒 **На пальцах.** Perplexity отвечает на вопрос «сколько вариантов у модели реально осталось». Игрок с 1 битом на букву имеет perplexity 2^1 = 2: как будто на каждом шаге выбор из двух букв, а не из 27. Слепое угадывание — perplexity 27. Меньше — лучше, всегда.

Одно переключение единиц, за которым стоит следить: игра считает биты на букву (`log2`), а формулы n-gram ниже считают наты на словесный токен (натуральный логарифм) — и поскольку perplexity `e^H` в натах равна `2^H` в битах, это одно и то же измерение в разных единицах.

```figure
prediction-game
```

**N-gram probability:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Фиксируем `n` (обычно 3 для триграмм, 4 для 4-грамм). Считаем по счётчикам:

```text
P(w | context) = count(context, w) / count(context)
```

> 🎒 **На пальцах.** Формула читается так: сколько раз контекст встретился вместе с этим словом, делённое на то, сколько раз контекст встретился вообще. Для «the cat» → «sat»: 47 / 59 ≈ 0.80. Это вся математика n-gram-модели, дальше только сглаживание.

**The zero-count problem.** Любая n-gram, которой не было в обучении, получает вероятность ноль. Исследование 2007 года на корпусе Brown показало, что даже у 4-граммной модели 30% отложенных 4-грамм не встречались в обучении. Без сглаживания вы не сможете оценить модель ни на одном настоящем тексте.

> 🎒 **На пальцах.** 30% — это почти каждая третья четвёрка слов в новом тексте, которую модель никогда не видела. Представьте словарь, где треть строк пустая. Модель без сглаживания на таком тексте просто уходит в ноль и перестаёт что-либо значить.

**Smoothing approaches, in order of sophistication:**

1. **Laplace (add-one).** Прибавить 1 к каждому счётчику. Просто, но ужасно на редких событиях.
2. **Good-Turing.** Перераспределить вероятностную массу от частых событий к невиданным, опираясь на частоты частот.
3. **Interpolation.** Смешать оценки n-gram, (n-1)-gram и так далее с настраиваемыми весами.
4. **Backoff.** Если счётчик n-gram нулевой, откатиться к (n-1)-gram. Katz backoff нормализует такой откат.
5. **Absolute discounting.** Вычесть фиксированную скидку `D` из всех счётчиков и раздать вычтенное невиданным.
6. **Kneser-Ney.** Absolute discounting плюс умный выбор модели младшего порядка: вместо сырой частоты берётся *continuation probability* (в скольких контекстах встречается слово).

> 🎒 **На пальцах.** Шесть способов — это шесть уровней щедрости. Laplace просто раздаёт всем по единице. Absolute discounting отнимает у каждого фиксированные D = 0.75 и складывает отнятое в общий котёл. Kneser-Ney делает то же самое, но умнее делит котёл между теми, кто ничего не получил.

Идея Kneser-Ney глубока. «San Francisco» — частая биграмма. Униграмма «Francisco» встречается в основном после «San». Наивный absolute discounting даёт «Francisco» высокую униграммную вероятность (потому что счётчик большой). Kneser-Ney замечает, что «Francisco» встречается только в одном контексте, и соответственно снижает его continuation probability. Результат: новая биграмма, заканчивающаяся на «Francisco», получает подобающе низкую вероятность.

> 🎒 **На пальцах.** Пусть «Francisco» встретилось в тексте 500 раз, но 499 из них — после «San». Разных соседей слева у него один, а не 500. Kneser-Ney считает не «сколько раз слово встретилось», а «со сколькими разными соседями», и «Francisco» падает почти в самый низ списка.

**Evaluation: perplexity.** Экспонента от средней отрицательной логарифмической правдоподобности на слово на отложенном тестовом наборе. Меньше — лучше. Perplexity 100 означает, что модель растеряна ровно настолько, насколько была бы при равномерном выборе из 100 слов.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

> 🎒 **На пальцах.** Формула берёт логарифм вероятности каждого слова, усредняет и возвращает обратно через `exp`. Если модель на каждом слове давала вероятность 1/100, средний логарифм равен log(0.01), а exp(−log(0.01)) = 100. Perplexity 100 = «выбор вслепую из ста слов».

```figure
ngram-backoff
```

## Build It

### Step 1: trigram counts

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

На вход подаётся список токенизированных предложений. На выходе — счётчики n-gram и счётчики контекстов. `<s>` и `</s>` — границы предложения.

> 🎒 **На пальцах.** Паддинг `<s>` нужен, чтобы у первого слова тоже был контекст. Для триграммы (n = 3) в начало добавляются два токена `<s>`, поэтому предложение из 5 слов даёт 2 + 5 + 1 = 8 позиций и 8 − 3 + 1 = 6 триграмм. Без паддинга первые слова просто нечем предсказывать.

### Step 2: Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Прибавляем 1 к каждому счётчику. Сглаживает, но отдаёт слишком много массы невиданным событиям, попутно портя и редкие известные.

> 🎒 **На пальцах.** Словарь из 10 000 слов, контекст встретился 5 раз, нужное слово после него — ни разу. Laplace выдаёт (0 + 1) / (5 + 10000) ≈ 0.0001 вместо нуля — хорошо. Но слово, которое встречалось 4 раза из 5, получает (4 + 1) / 10005 ≈ 0.0005 вместо честных 4/5 = 0.8. Вот почему Laplace годится только для учебников.

### Step 3: Kneser-Ney (bigram, interpolated)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(context, w):
        prev = context[-1]
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Три подвижные части. `continuation_prob` отвечает на вопрос «в скольких разных контекстах встречается это слово?» (это и есть находка Kneser-Ney). `lambda_prev` — масса, освобождённая скидкой, она задаёт вес отката. Итоговая вероятность — уменьшенный основной член плюс взвешенный continuation-член. Ни одного сырого униграммного счётчика внутри этой функции нет, и это принципиально: у Kneser-Ney модель младшего порядка — continuation probability, а не частота.

Возвращаемая `prob` имеет ту же форму `(context, word)`, что `raw_probability` и `laplace_probability` из Шагов 1-2: `context` — кортеж из предыдущих `n-1` токенов, здесь кортеж из одного элемента. Именно единая сигнатура у всех трёх моделей и позволяет Шагам 4 и 5 принимать любую из них.

> 🎒 **На пальцах.** Смотрите на `discount=0.75`. Если биграмма встретилась 4 раза при контексте, встреченном 10 раз, основной член даёт (4 − 0.75) / 10 = 0.325 вместо 0.4. Отнятые 0.075 уходят в `lambda_prev` и раздаются словам пропорционально continuation probability.

### Step 4: generating text with sampling

```python
import random


def generate(prob_fn, vocab, prefix, n=2, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        context = tuple(tokens[-(n - 1):])
        candidates = [(w, prob_fn(context, w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Сэмплирование пропорционально вероятности. Каждый seed даёт свой результат. Чтобы получить вывод в духе beam search, берите argmax на каждом шаге (жадно) и добавьте небольшую ручку случайности (temperature).

> 🎒 **На пальцах.** Сэмплирование — это рулетка с секторами разной ширины. Слово с вероятностью 0.8 занимает 80% круга, слово с 0.2 — оставшиеся 20%. `rng.random() * total` бросает шарик, цикл идёт по секторам и останавливается там, где шарик упал. Тот же `seed=0` — тот же бросок, тот же текст.

### Step 5: perplexity

```python
import math


def perplexity(prob_fn, sentences, n=2):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(n - 1, len(padded)):
            context = tuple(padded[i - n + 1:i])
            p = prob_fn(context, padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

`n` должно совпадать с порядком модели, стоящей за `prob_fn`: паддинг и ширина контекста обязаны быть одинаковыми на обучении и на тесте. Любая из трёх моделей Шагов 1-3 подставляется напрямую — это и есть Упражнение 2:

```python
ngrams, contexts = train_ngram(train, n=3)
vocab_size = len({w for s in train for w in s} | {"<s>", "</s>"})

kn = kneser_ney_bigram_model(train)
laplace = lambda ctx, w: laplace_probability(ngrams, contexts, vocab_size, ctx, w)

print(perplexity(kn, test, n=2))
print(perplexity(laplace, test, n=3))
```

Сравнивать можно только модели, посчитанные на одних и тех же токенах: биграммная и триграммная модели здесь видят один и тот же тестовый набор токенов, а вот сравнение через разные токенизации не значит вообще ничего.

Меньше — лучше. На корпусе Brown хорошо настроенная 4-граммная KN-модель выходит на perplexity около 140. Трансформерная языковая модель на том же тесте даёт 15–30. Разрыв примерно десятикратный. Из-за этого разрыва область и пошла дальше.

> 🎒 **На пальцах.** Переведите числа обратно в игру: n-gram на каждом слове мечется между 140 вариантами, трансформер — между двадцатью. 140 / 15 ≈ 9, в тексте округляют до 10x. Ради этой девятки и построили всю Phase 6.

## Use It

- **Classical NLP teaching.** Самое ясное знакомство со сглаживанием, MLE и perplexity, какое вообще можно получить.
- **KenLM.** Продакшн-библиотека для n-gram. Используется как rescorer в системах распознавания речи и перевода, где важна низкая задержка.
- **On-device autocomplete.** Триграммные модели в клавиатурах. До сих пор.
- **Baselines.** Всегда считайте perplexity n-gram-модели, прежде чем объявлять свою нейросетевую LM хорошей. Если ваш трансформер не обгоняет KN с большим отрывом, что-то сломано.

> 🎒 **На пальцах.** Baseline — это карандашная отметка роста на дверном косяке. Если ваш трансформер даёт perplexity 130 против 140 у KN, вы потратили GPU-недели ради 7% — почти наверняка ошибка в данных, токенизации или обучении, а не «трансформеры не работают».

## Ship It

Сохраните как `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Exercises

1. **Easy.** Обучите триграммную LM на корпусе из 1000 предложений Шекспира. Сгенерируйте 20 предложений. Локально они будут правдоподобны, глобально — бессвязны. Это каноническая демонстрация.
2. **Medium.** Реализуйте perplexity для своей KN-модели на отложенной части Шекспира. Сравните с Laplace. Вы должны увидеть, что KN снижает perplexity на 30–50%.
3. **Hard.** Постройте триграммный корректор опечаток: по неверно написанному слову и его контексту генерируйте варианты исправления и ранжируйте их по вероятности в контексте под вашей LM. Оцените на корпусе опечаток Birkbeck (открытый).

> 🎒 **На пальцах.** Подсказка ко второму заданию: готовый сниппет лежит в Шаге 5. Главное — не перепутать `n`: KN-модель из Шага 3 биграммная, поэтому `perplexity(kn, test, n=2)`, а Laplace обучен на триграммах, поэтому `n=3`. Передадите не то `n` — и `prob_fn` получит контекст не той ширины, а число на выходе окажется мусором. И считайте perplexity на одних и тех же предложениях с одинаковой токенизацией, иначе числа несравнимы вообще. Ожидайте примерно такую картину: Laplace ≈ 300, Kneser-Ney ≈ 180. Это падение на 40% — как раз обещанные 30–50%.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| N-gram | Последовательность слов | Последовательность из `n` подряд идущих токенов. |
| Smoothing | Чтобы не было нулей | Перераспределение вероятностной массы так, чтобы невиданные события получили ненулевую вероятность. |
| Perplexity | Метрика качества LM | `exp(-average log-prob)` на отложенных данных. Меньше — лучше. |
| Backoff | Откат к более короткому контексту | Если счётчик триграммы ноль, берём биграмму. Katz backoff формализует это правило. |
| Kneser-Ney | Лучшее сглаживание для n-gram | Absolute discounting плюс continuation probability для модели младшего порядка. |
| Continuation probability | Специфика KN | `P(w)`, взвешенная числом контекстов, в которых встречается `w`, а не сырым счётчиком. |
| Entropy of text | Информация на символ | Среднее число бит, нужное для кодирования следующего символа при данном контексте. Оценка Шеннона 1951 года для печатного английского с контекстом до 100 букв: 0.6–1.3 бита на букву, измерена до появления любых моделей. |

## Further Reading

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — эксперимент с угадыванием, который задал цель, оптимизируемую каждой языковой моделью до сих пор.
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — канонический разбор n-gram-моделей и сглаживания.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — статья, окончательно закрепившая Kneser-Ney как лучшее сглаживание для n-gram.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — оригинальная статья про KN.
- [KenLM](https://kheafield.com/code/kenlm/) — быстрая продакшн-библиотека n-gram LM, в 2026 году всё ещё используется там, где важна задержка.

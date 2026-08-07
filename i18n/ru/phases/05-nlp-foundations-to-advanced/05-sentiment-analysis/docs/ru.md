<!-- i18n:manual -->
# Анализ тональности

> Каноническая задача NLP. Почти всё, что нужно знать про классическую классификацию текста, всплывает именно здесь.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 minutes

## The Problem

«The food was not great.» Позитив или негатив?

Тональность кажется простой. Автор отзыва сказал, что ему что-то понравилось или не понравилось. Разметьте предложение. Канонической задачей NLP она стала потому, что за каждым лёгким случаем прячется тяжёлый. Отрицание переворачивает смысл. Сарказм выворачивает его наизнанку. «Not bad at all» — позитив, хотя оба значимых слова негативные. Эмодзи несут больше сигнала, чем весь окружающий текст. Доменная лексика решает (`tight` в рецензии на альбом и `tight` в обзоре одежды — разные вещи).

Тональность — рабочая лаборатория классического NLP. Если вы понимаете, почему у каждого наивного бейзлайна есть свой конкретный режим отказа, вы понимаете, зачем изобрели каждую следующую модель. В этом уроке мы строим Naive Bayes с нуля, добавляем логистическую регрессию и называем ловушки, из-за которых продакшен-тональность становится задачей уровня комплаенса.

> 🎒 **На пальцах.** Возьмите фразу «not great». Мешок слов видит два токена: `not` и `great`. Слово `great` в обучении почти всегда встречалось в позитивных отзывах, поэтому модель уверенно ставит «позитив» — и ошибается. Одно слово из двух перевесило смысл всей фразы. Весь урок про то, как это чинят.

## The Concept

Классический анализ тональности — рецепт из двух шагов.

1. **Represent.** Превратить текст в вектор признаков. BoW, TF-IDF или n-grams.
2. **Classify.** Обучить линейную модель (Naive Bayes, логистическая регрессия, SVM) на размеченных примерах.

Naive Bayes — самая тупая модель, которая работает. Считаем, что все признаки независимы при известной метке. Оцениваем `P(word | positive)` и `P(word | negative)` по счётчикам. На инференсе перемножаем вероятности. Предположение о независимости смехотворно неверно, и тем не менее результаты поразительно хороши. Причина: на разреженных текстовых признаках и умеренных данных классификатору важно, в какую сторону клонит каждое слово, а не насколько сильно.

> 🎒 **На пальцах.** «Независимость» здесь значит: модель считает, что слово `not` появилось в отзыве само по себе, безо всякой связи с соседним `great`. Это очевидная неправда. Но чтобы ответить «позитив или негатив», достаточно суммы голосов: если 8 слов из 10 клонят в плюс, ответ «плюс» — даже если модель неправильно оценила силу каждого голоса.

Логистическая регрессия чинит предположение о независимости. Она учит вес для каждого признака, в том числе отрицательные. Биграмма `not good` получает отрицательный вес. Naive Bayes так не умеет для биграмм, которых он не размечал.

> 🎒 **На пальцах.** Разница как между подсчётом голосов и подсчётом голосов с весами. Naive Bayes считает, сколько раз слово встречалось в каждом классе. Логистическая регрессия подбирает вес каждому слову так, чтобы весь набор вместе давал правильный ответ, — и может дать признаку `not_good` вес −2.3, то есть прямой минус, а не просто «редко встречался в позитиве».

```figure
sentiment-logits
```

## Build It

### Step 1: a real mini-dataset

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Маленький намеренно. В реальной работе используют десятки тысяч примеров (IMDb, SST-2, Yelp polarity). Математика та же самая.

> 🎒 **На пальцах.** Пять позитивных и пять негативных отзывов — значит априорная вероятность каждого класса ровно 5/10 = 0.5. Модель стартует с честной монетки, и всё, что её сдвинет, — это слова. На настоящем корпусе IMDb (25 000 отзывов) априорные вероятности тоже 0.5, потому что датасет специально сбалансировали.

### Step 2: multinomial Naive Bayes from scratch

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    if alpha <= 0:
        raise ValueError("alpha must be > 0; alpha=0 leaves zero probabilities and predict_nb logs them")
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts[w] for w in vocab) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Аддитивное сглаживание (alpha=1.0) — это сглаживание Лапласа. Без него слово, не встреченное в классе, имеет вероятность ноль, и логарифм улетает в бесконечность — поэтому `train_nb` отказывается работать с `alpha=0` сразу, а не даёт `predict_nb` умереть на `math.log(0)` уже во время инференса. На практике часто берут `alpha=0.01`. `alpha=1.0` — учебное значение по умолчанию.

Приглядитесь к знаменателю. Он суммирует счётчики только по `vocab`, а не берёт `sum(counts.values())`. Эти два выражения совпадают лишь тогда, когда словарь покрывает каждый обучающий token. Как только вы отсечёте редкие слова порогом по частоте, `counts` всё равно продолжит хранить выброшенные tokens, и полная сумма сделает так, что `class_word_probs[cls]` в сумме даст меньше единицы. Распределение, которое не распределение, — это баг, которого вы не увидите в цифре accuracy, только в калибровке.

> 🎒 **На пальцах.** Смотрите, что чинит alpha. Слово `brilliant` не встречается ни в одном негативном отзыве, значит `P(brilliant | negative) = 0/24 = 0`. Умножьте на это ноль — и весь отзыв получает вероятность негатива ноль, что бы в нём ещё ни стояло. С alpha=1 вместо нуля выходит 1/(24 + размер словаря): маленькое число, но не приговор. А `alpha=0` теперь и не примут: `train_nb` бросит `ValueError` на входе, потому что «сглаживание без сглаживания» — это не настройка, а отложенная поломка в `predict_nb`.

> 🎒 **На пальцах.** Почему в `predict_nb` складывают логарифмы, а не перемножают вероятности. Отзыв из 20 слов, каждое с вероятностью около 0.001, даёт произведение 10 в минус шестидесятой. Такое число float просто округлит до нуля, и оба класса получат одинаковый счёт. Логарифмы превращают умножение в сложение: 20 × log(0.001) ≈ −138. Число как число, сравнивать можно.

### Step 3: logistic regression from scratch

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2-регуляризация здесь важна. Текстовые признаки разрежены; без L2 модель запоминает обучающие примеры. Начните с `0.01` и подбирайте.

> 🎒 **На пальцах.** Обратите внимание на `np.clip(x, -20, 20)` внутри сигмоиды. Без него `np.exp(-x)` при большом отрицательном x переполняется и выдаёт inf. А с обрезкой на 20 мы теряем ровно ничего: sigmoid(20) = 0.999999998, дальше округлять просто некуда.

> 🎒 **На пальцах.** Что делает L2 на пальцах. Слово `heartwarming` встретилось в обучении один раз, в позитивном отзыве. Без штрафа модель радостно поставит ему огромный вес — этого хватит, чтобы угадать тот единственный пример. С `l2=0.01` каждый шаг подтягивает все веса обратно к нулю, и редкое слово не может получить решающий голос по одному наблюдению.

### Step 4: handling negation (the failure mode)

Возьмите «not good» и «not bad». BoW-классификатор видит `{not, good}` и `{not, bad}` и учится по тому, что чаще попадалось в обучении. Биграммный классификатор видит `not_good` и `not_bad` и учит их как разные признаки. Обычно этого достаточно.

Более грубый способ, который работает, когда биграмм нет: **negation scoping**. Приписываем префикс `NOT_` всем токенам после отрицания и до ближайшего знака препинания.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Теперь `good` и `NOT_good` — разные признаки. Классификатор может дать им противоположные веса. Три строчки препроцессинга — измеримый прирост accuracy на бенчмарках тональности.

> 🎒 **На пальцах.** Проследите по выводу, где кончается зона отрицания. `not` включает режим, `good`, `at`, `all` получают префикс, точка выключает режим, и `but funny` остаётся нетронутым. Ровно так и надо: похвала после точки не имеет отношения к предыдущему отрицанию. Если бы правило не смотрело на пунктуацию, `funny` превратилось бы в `NOT_funny` и модель выучила бы чепуху.

### Step 5: evaluation metrics that matter

Одна accuracy вводит в заблуждение, если классы несбалансированы. В настоящих корпусах тональности обычно 70-80% позитива или 70-80% негатива; классификатор, всегда отвечающий большинством, получает 80% accuracy и не стоит ничего. Показывайте всё из списка ниже:

- **Per-class precision and recall.** По паре на класс. Усредните их макро-усреднением, чтобы получить одно число, уважающее баланс классов.
- **Macro-F1 (primary metric for imbalanced data).** Среднее F1 по классам с равными весами. Используйте вместо accuracy, когда классы несбалансированы.
- **Weighted-F1 (alternative).** То же, что макро, но с весами по частоте классов. Показывайте рядом с macro-F1, когда сам дисбаланс имеет бизнес-смысл.
- **Confusion matrix.** Сырые счётчики. Всегда смотрите на неё, прежде чем верить любому скалярному числу; она показывает, какую пару классов модель путает.
- **Per-class error samples.** Достаньте по 5 неверных предсказаний на класс. Прочитайте их. Ничто не заменяет чтения настоящих ошибок.

> 🎒 **На пальцах.** Посчитаем ту самую ловушку. Корпус: 80 позитивных отзывов и 20 негативных. Модель «всегда позитив» даёт accuracy 80/100 = 0.80. Но для позитивного класса F1 = 2 × 0.8 × 1.0 / 1.8 ≈ 0.89, а для негативного precision и recall равны нулю, значит F1 = 0. Macro-F1 = (0.89 + 0) / 2 ≈ 0.44. Восемьдесят процентов превратились в сорок четыре — вот почему метрику надо выбирать до того, как показывать результат.

Для сильно несбалансированных данных (соотношение хуже 95-5) показывайте **AUROC** и **AUPRC** вместо accuracy. AUPRC чувствительнее к меньшему классу, а именно он вас обычно и интересует (спам, фрод, редкая тональность).

**Common bug to avoid.** Показывать micro-F1 вместо macro-F1 на несбалансированных данных — значит получить красивое число, которое целиком определяется большинством. Macro-F1 заставляет вас увидеть качество на меньшем классе.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

> 🎒 **На пальцах.** Прогоните `evaluate` в голове. Пусть в тесте 10 позитивных отзывов, модель назвала позитивными 8 и угадала 6. Тогда tp=6, fp=2, fn=4. Precision = 6/8 = 0.75 («из того, что я назвал позитивом, три четверти — правда»). Recall = 6/10 = 0.6 («из всего позитива я нашёл шестьдесят процентов»). F1 = 2 × 0.75 × 0.6 / 1.35 ≈ 0.67. Четыре счётчика, три деления, никакой магии.

## Use It

scikit-learn делает то же самое в шесть строк и без ошибок.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Три вещи, на которые стоит посмотреть. `stop_words=None` сохраняет отрицания. `ngram_range=(1, 2)` добавляет биграммы, поэтому `not_good` становится признаком. `sublinear_tf=True` гасит эффект повторов. Эти три флага — разница между бейзлайном с 75% accuracy и бейзлайном с 85% на SST-2.

> 🎒 **На пальцах.** Особенно важен `stop_words=None`. Списки стоп-слов по умолчанию выбрасывают `not`, `no`, `never` — то есть ровно те слова, которые в задаче тональности решают всё. Выбросить `not` из «not good» — значит оставить модели просто `good`. Десять процентов accuracy теряются на одном флаге, который вы поставили «чтобы почище».

> 🎒 **На пальцах.** Что делает `sublinear_tf=True`: заменяет счётчик tf на 1 + log(tf). Слово, встреченное 10 раз, получает вес 1 + log(10) ≈ 3.3, а не 10. Отзыв, где кто-то написал «awful awful awful awful», перестаёт весить вчетверо больше обычного «awful».

### When to reach for a transformer

- Детекция сарказма. Классические модели здесь не работают. Точка.
- Длинные отзывы, где тональность меняется по ходу текста.
- Aspect-based тональность. «Camera was great but battery was terrible.» Нужно привязать тональность к аспектам. Только трансформеры или модели со структурированным выходом.
- Не-английские языки с малыми ресурсами. Многоязычный BERT даёт zero-shot бейзлайн бесплатно.

Если вам нужно что-то из перечисленного, перепрыгивайте в фазу 7 (глубокое погружение в трансформеры). Иначе Naive Bayes или логистическая регрессия на TF-IDF плюс биграммы плюс обработка отрицаний — ваш продакшен-бейзлайн 2026 года.

### The reproducibility trap (again)

Переобучать модели тональности — рутина. Переоценивать их — нет. Цифры accuracy в статьях получены на конкретных разбиениях, конкретном препроцессинге, конкретных токенизаторах. Если вы сравниваете свою новую модель с бейзлайном, не воспроизведя идентичный пайплайн, дельты будут вводить в заблуждение. Всегда пересчитывайте бейзлайн в своём пайплайне, а не берите число из статьи.

> 🎒 **На пальцах.** Живой пример: статья заявляет 88.1% на SST-2, вы получили 89.4% и радуетесь плюс 1.3. А потом выясняется, что в статье тестовый сплит из 1821 предложения, а у вас случайные 20% всего корпуса. Сравнивали не модели, а сплиты. Единственное лекарство — прогнать чужой бейзлайн у себя.

## Ship It

Сохраните как `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

> 🎒 **На пальцах.** Обратите внимание, что промпт заканчивается тремя отказами, а не тремя советами. «Отказывайся выбрасывать стоп-слова», «отказывайся показывать одну accuracy на дисбалансе» — это ровно те две ошибки, которые вы только что разобрали в уроке. Скилл нужен не для того, чтобы напомнить хорошее, а чтобы заблокировать плохое.

## Exercises

1. **Easy.** Добавьте `apply_negation` как шаг препроцессинга в пайплайн scikit-learn и измерьте дельту F1 на небольшом датасете тональности.
2. **Medium.** Реализуйте логистическую регрессию со взвешиванием классов (передайте `class_weight="balanced"` в scikit-learn или выведите градиент сами). Измерьте эффект на синтетическом дисбалансе 90-10.
3. **Hard.** Постройте детектор сарказма, обучив второй классификатор на остатках модели тональности. Опишите постановку эксперимента. Предупредите читателя, если ваша accuracy ниже случайной (на двухклассовом сарказме случайный уровень около 50%, и большинство первых попыток попадают именно туда).

> 🎒 **На пальцах.** Подсказка ко второму заданию. `class_weight="balanced"` — это не магия, а одна формула: вес класса = число всех примеров / (число классов × число примеров этого класса). На дисбалансе 90-10 из 1000 объектов меньший класс получит вес 1000 / (2 × 100) = 5.0, а больший — 1000 / (2 × 900) ≈ 0.56. То есть каждая ошибка на редком классе штрафуется примерно в 9 раз сильнее. Ожидайте: accuracy слегка упадёт, macro-F1 вырастет.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Polarity | Позитив или негатив | Бинарная метка; иногда расширяется до нейтральной или дробной (5 звёзд). |
| Aspect-based sentiment | Тональность по аспектам | Привязка тональности к конкретным сущностям или свойствам, упомянутым в тексте. |
| Negation scoping | Переворот соседних токенов | Приписать префикс `NOT_` токенам после «not» и до знака препинания. |
| Laplace smoothing | Прибавление единицы к счётчикам | Не даёт признакам получить нулевую вероятность в Naive Bayes. |
| L2 regularization | Ужимание весов | Добавляет `lambda * sum(w^2)` к функции потерь. Обязательно для разреженных текстовых признаков. |

## Further Reading

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — основополагающий обзор. Длинный, но первые четыре раздела покрывают всю классику.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — статья, показавшая, что биграммы плюс Naive Bayes на коротких текстах трудно обыграть.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — справочник по `CountVectorizer`, `TfidfVectorizer` и каждому параметру, который вам придётся крутить.

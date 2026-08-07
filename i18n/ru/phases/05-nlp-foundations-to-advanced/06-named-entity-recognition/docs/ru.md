<!-- i18n:manual -->
# Распознавание именованных сущностей

> Вытащите имена. Звучит просто, пока не столкнётесь с размытыми границами, вложенными сущностями и доменным жаргоном.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## The Problem

«Apple sued Google over its iPhone search deal in the US.» Пять сущностей: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (может быть), US (GPE). Хорошая NER-система вытащит их все с правильными типами. Плохая пропустит iPhone, спутает Apple-фрукт с Apple-компанией и разметит «US» как PERSON.

NER — рабочая лошадь под каждым пайплайном структурированного извлечения. Разбор резюме, сканирование логов на комплаенс, обезличивание медкарт, понимание поисковых запросов, заземление ответов чат-бота, извлечение условий из договоров. Вы его почти не видите — и всегда от него зависите.

Этот урок проходит классический путь (правила, HMM, CRF) и выходит на современный (BiLSTM-CRF, потом трансформеры). Каждый шаг решает конкретное ограничение предыдущего. Сама эта закономерность и есть урок.

> 🎒 **На пальцах.** Разберите пример на пальцах: почему «US» может уехать в PERSON. Модель видит слово из двух заглавных букв. Такую же форму имеют инициалы человека. Без контекста «in the ___» это просто две буквы. Именно из-за таких случаев NER не решается словарём — нужна модель, которая смотрит на соседей.

## The Concept

**BIO tagging** (или BILOU) превращает извлечение сущностей в задачу разметки последовательности. Размечаем каждый токен как `B-TYPE` (начало сущности), `I-TYPE` (внутри сущности) или `O` (вне любой сущности).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Многотокенные сущности сцепляются: `New B-GPE`, `York I-GPE`, `City I-GPE`. Модель, понимающая BIO, умеет извлекать span любой длины.

> 🎒 **На пальцах.** Посчитайте по табличке выше: 12 токенов, из них ровно 3 помечены как `B-` и 9 как `O`. То есть на типичном предложении почти все метки — это скучное `O`. Отсюда важное следствие: модель, которая всегда отвечает `O`, получит 75% accuracy по токенам и не найдёт ни одной сущности. Поэтому NER меряют по сущностям, а не по токенам.

> 🎒 **На пальцах.** Зачем нужна отдельная метка `I-`, а не просто `B-` на каждом слове. «New York City» — это одна сущность из трёх слов. Если пометить все три как `B-GPE`, получится три отдельных города вместо одного. `B` говорит «здесь новая сущность начинается», `I` — «это продолжение предыдущей».

Прогрессия архитектур:

- **Rule-based.** Регулярки плюс поиск по газетиру. Высокая precision на известных сущностях, нулевое покрытие новых.
- **HMM.** Скрытая марковская модель. Вероятность эмиссии токена при данном теге, вероятность перехода тег-в-тег. Декодирование Витерби. Обучается на размеченных данных.
- **CRF.** Conditional Random Field. Как HMM, но дискриминативная, поэтому можно мешать любые признаки (форма слова, регистр, соседние слова). До сих пор классическая рабочая лошадь продакшена в 2026 году для развёртываний с малыми ресурсами.
- **BiLSTM-CRF.** Признаки нейросетевые, а не написанные руками. LSTM читает предложение в обе стороны, слой CRF сверху следит за согласованностью последовательности тегов.
- **Transformer-based.** Дообучаем BERT с головой для классификации токенов. Лучшая точность. Больше всего вычислений.

```figure
ner-bio-tagging
```

## Build It

### Step 1: BIO tagging helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        if any(labels[i] != "O" for i in range(start, end)):
            raise ValueError(
                f"span ({start}, {end}, {label}) overlaps an existing span; "
                "BIO gives each token exactly one label and cannot nest"
            )
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must be the same length")
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

> 🎒 **На пальцах.** Проследите за `bio_to_spans` по этому примеру. Семь токенов на входе, три span на выходе. Span `(0, 1, 'ORG')` читается как «с токена 0 по токен 1, не включая, тип ORG» — то есть ровно слово `Apple`. Это питоновские срезы: `tokens[0:1]`. Хранить границы, а не сами слова, удобно тем, что вы всегда можете вернуться к исходному тексту.

Проверка на пересечение в `spans_to_bio` — самая честная часть этого хелпера. BIO хранит по одной метке на token, поэтому «Bank of America Tower» физически нельзя разметить одновременно как ORG и как FACILITY. Без проверки второй span молча затирает первый, и разметка теряется где-то между загрузчиком данных и обучающим набором. Лучше упасть ровно там, где формат кончился, — смотрите «Nested entities» в разделе *Where it falls apart*. Заодно обратите внимание на `len(tokens) != len(labels)` в `bio_to_spans`: без этой строки параметр `tokens` вообще ни на что не влиял бы, а рассинхрон длин уехал бы в спаны с границами за пределами предложения.

### Step 2: hand-crafted features

Для классического (не нейросетевого) NER всё решают признаки. Полезные:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` возвращает `xXxxxx`. `word_shape("USA-2024")` возвращает `XXX-dddd`. Паттерны регистра — сильный сигнал для имён собственных.

> 🎒 **На пальцах.** Форма слова — это отпечаток без самого слова. `xXxxxx` для `iPhone` — строчная, потом заглавная в середине: так выглядят названия продуктов и почти ничто другое. `XXX-dddd` подходит и под `USA-2024`, и под `IBM-1401`. Модель, никогда не видевшая `IBM-1401`, всё равно узнает форму и поставит правильный тег.

> 🎒 **На пальцах.** Признак `suffix_3` — просто три последние буквы в нижнем регистре. Для `Washington` это `ton`, для `Arlington` тоже `ton`, для `Lexington` снова `ton`. Одна дешёвая строка кода даёт модели половину морфологии английских топонимов.

### Step 3: a simple rule-based + dictionary baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Заметьте, чего этот цикл не может сделать в принципе: он умеет выдавать только теги `B-`. Поиск идёт по одному token за раз, поэтому сущность из двух слов возвращается двумя отдельными односложными сущностями (`New` — `B-GPE`, `York` — `B-GPE`), а многословная запись газетира не сматчится вообще никогда. Дешёвое лечение — склеивать подряд идущие срабатывания одного типа в цепочку `B-` / `I-`; настоящее — искать по газетиру фразами, выбирая самое длинное совпадение (longest match).

Продакшен-газетиры содержат миллионы записей, собранных из Википедии и DBpedia. Покрытие хорошее. Разрешение неоднозначностей (`Apple` компания или фрукт) — ужасное. Поэтому статистические модели и победили.

> 🎒 **На пальцах.** В этом газетире 7 организаций, 6 стран и 5 продуктов — восемнадцать записей всего. Подайте на вход «Apple pie is delicious» — и `rule_based_ner` уверенно скажет `B-ORG` про яблочный пирог. Словарь не умеет смотреть на соседнее слово `pie`. Дальше миллион записей ничего не изменит: он просто ошибётся миллионом способов.

### Step 4: the CRF step (sketch, not full impl)

Полный CRF с нуля в 50 строк не даёт озарения без базы по теории вероятностей. Возьмите `sklearn-crfsuite`:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` и `c2` — это L1- и L2-регуляризация. `all_possible_transitions=True` позволяет модели выучить, что некорректные последовательности (например, `I-ORG` сразу после `O`) маловероятны — так CRF обеспечивает согласованность BIO, а вам не приходится писать это ограничение руками.

> 🎒 **На пальцах.** Вот в чём главное отличие CRF от обычного классификатора. Обычная модель размечает каждый токен отдельно и легко выдаст последовательность `O`, `I-ORG`, `O` — бессмыслицу, ведь сущность не может начаться с `I-`. CRF считает вес всей последовательности целиком, а переход `O` → `I-ORG` в обучении не встречался ни разу, поэтому получает почти минус бесконечность. Ошибка становится невозможной по построению.

> 🎒 **На пальцах.** Параметры `c1=0.1` и `c2=0.1` при десятке признаков на токен и, скажем, 15 000 предложений держат число реально работающих признаков в разумных рамках. L1 (`c1`) обнуляет бесполезные признаки целиком, L2 (`c2`) просто ужимает все. Первое даёт модель поменьше, второе — постабильнее.

### Step 5: what a BiLSTM-CRF adds

Признаки становятся выученными. На входе: эмбеддинги токенов (GloVe или fastText). LSTM читает слева направо и справа налево. Склеенные скрытые состояния идут в выходной слой CRF. CRF по-прежнему следит за согласованностью тегов; LSTM заменяет рукописные признаки выученными.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Для слоя CRF используйте `torchcrf.CRF` (pip install pytorch-crf). Выигрыш над CRF с рукописными признаками измерим, но меньше, чем вы ожидаете, — если только у вас нет десятков тысяч размеченных предложений.

> 🎒 **На пальцах.** Обратите внимание на `nn.Linear(hidden_dim * 2, n_labels)`. Двойка здесь потому, что LSTM двунаправленная: при `hidden_dim=128` на вход линейному слою приходит 256 чисел — 128 от прохода слева направо и 128 справа налево. Каждый токен видит и то, что было до него, и то, что после. Именно это и отличает `Apple sued` от `Apple pie`.

## Use It

spaCy даёт продакшен-уровень NER из коробки.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Заметьте: `iPhone` размечен как `ORG`, а не `PRODUCT` — у маленькой модели spaCy слабое покрытие сущностей-продуктов. Большая модель (`en_core_web_lg`) справляется лучше. Трансформерная (`en_core_web_trf`) — ещё лучше.

> 🎒 **На пальцах.** Считаем по выводу: из пяти сущностей исходного предложения модель нашла четыре, из них три с правильным типом. Это 60% верных сущностей на одной строке кода. Полезно — и одновременно ровно та причина, по которой перед продакшеном каждую модель прогоняют на своих текстах, а не верят цифрам из README.

Hugging Face для NER на базе BERT:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` склеивает подряд идущие токены B-X и I-X в один span. Без него вы получаете метки на уровне токенов и склеиваете сами.

> 🎒 **На пальцах.** Почему склейка вообще нужна. BERT режет слова BPE-токенизатором: `Washington` может стать `Wash` + `##ington`, то есть два токена и две метки `B-GPE` + `I-GPE`. Без агрегации вы получите на выходе куски слов. С `aggregation_strategy="simple"` они снова собираются в человеческое слово.

### LLM-based NER (the 2026 option)

Zero-shot и few-shot NER на LLM теперь конкурентны с дообученными моделями во многих доменах и заметно лучше там, где размеченных данных мало.

- **Zero-shot prompting.** Дайте LLM список типов сущностей и пример схемы. Попросите JSON. Работает из коробки; точность на новых доменах средняя.
- **ZeroTuneBio-style prompting.** Разложите задачу на шаги: извлечение кандидатов → объяснение смысла → решение → перепроверка. Многоступенчатый промпт (не одношаговый) существенно поднимает точность на биомедицинском NER. Тот же приём работает в юридическом, финансовом и научном доменах.
- **Dynamic prompting with RAG.** Для каждого вызова доставайте наиболее похожие размеченные примеры из небольшого затравочного набора и собирайте few-shot промпт на лету. На бенчмарках 2026 года это поднимает F1 биомедицинского NER у GPT-4 на 11-12% относительно статического промпта.
- **Per-entity-type decomposition.** На длинных документах один вызов, извлекающий сразу все типы, теряет recall по мере роста длины. Делайте отдельный проход на каждый тип сущности. Дороже по инференсу, заметно точнее. Это стандартный приём для клинических записей и юридических договоров.

Рекомендация для продакшена на 2026 год: начните с zero-shot бейзлайна на LLM, ещё до сбора обучающих данных. Часто F1 оказывается достаточным, и дообучение просто не понадобится.

> 🎒 **На пальцах.** Прикиньте цену последнего пункта. Восемь типов сущностей и отдельный проход на каждый — это восемь вызовов LLM вместо одного, то есть примерно в восемь раз дороже и дольше. За это вы получаете единицы пунктов F1. На договоре в 50 страниц, который обрабатывают раз в день, — очевидная сделка. На поисковых запросах с миллионом обращений в час — очевидно нет.

### Where classical NER still wins

Даже когда LLM под рукой, классический NER выигрывает, если:

- Бюджет по задержке меньше 50 мс.
- У вас тысячи размеченных примеров и нужен F1 выше 98%.
- В домене устойчивая онтология, куда хорошо переносится предобученный CRF или BiLSTM.
- Регуляторные требования обязывают использовать локальную, негенеративную модель.

### Where it falls apart

- **Domain shift.** NER, обученный на CoNLL, на юридических договорах работает хуже газетира. Дообучайте на своём домене.
- **Nested entities.** «Bank of America Tower» — одновременно ORG и FACILITY. Стандартный BIO не умеет представлять пересекающиеся span. Нужен вложенный NER (многопроходные или span-based модели).
- **Long entities.** «United States Federal Deposit Insurance Corporation.» Модели уровня токенов иногда разрывают это. Используйте `aggregation_strategy` или постобработку.
- **Sparse types.** Медицинские метки вроде DRUG_BRAND, ADVERSE_EVENT, DOSE. Универсальные модели о них не подозревают. Стартовые точки здесь — Scispacy и BioBERT.

> 🎒 **На пальцах.** Вложенные сущности ломают BIO по-честному, а не по недосмотру. У каждого токена ровно одна метка. Слово `America` в «Bank of America Tower» должно быть одновременно `I-ORG` (часть банка) и `I-FAC` (часть здания). Двух меток на один токен схема просто не предусматривает — отсюда и весь отдельный класс span-based моделей.

## Ship It

Сохраните как `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

> 🎒 **На пальцах.** Пункт 4 в этом скилле — самый недооценённый. `seqeval` считает F1 по сущностям: предсказанный span засчитывается, только если он совпал с настоящим полностью. Предскажите «New York» вместо «New York City» — по токенам вы угадали 2 из 3 и получите приличную цифру, по сущностям это чистый ноль. Токенная метрика на NER всегда выглядит лучше, чем модель есть на самом деле.

## Exercises

1. **Easy.** Реализуйте `bio_to_spans` (обратную к `spans_to_bio`) и проверьте согласованность туда-обратно на 10 предложениях.
2. **Medium.** Обучите CRF из sklearn-crfsuite выше на англоязычном датасете CoNLL-2003. Посчитайте F1 по типам сущностей через `seqeval`. Типичный результат: около 84 F1.
3. **Hard.** Дообучите `distilbert-base-cased` на доменном NER-датасете (медицина, право или финансы). Сравните с маленькой моделью spaCy. Опишите проверки на утечку данных и напишите, что вас удивило.

> 🎒 **На пальцах.** Подсказка к первому заданию: проверка «туда-обратно» — это `spans_to_bio(tokens, bio_to_spans(tokens, labels)) == labels`. Специально возьмите злые случаи: сущность в самом конце предложения (сработает ли `if current` после цикла), два разных типа подряд, одинокая метка `I-ORG` без `B-ORG` перед ней. Именно на третьем случае большинство реализаций и падает.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NER | Извлечь имена | Разметить span токенов типами (PERSON, ORG, GPE, DATE, ...). |
| BIO | Схема разметки | `B-X` начинает, `I-X` продолжает, `O` вне сущности. |
| BILOU | BIO получше | Добавляет `L-X` (последний) и `U-X` (одиночный) для чётких границ. |
| CRF | Структурный классификатор | Моделирует переходы между метками, а не только эмиссии. Обеспечивает корректные последовательности. |
| Nested NER | Пересекающиеся сущности | Один span — сущность одного типа, а его подотрезок — другого. BIO это выразить не может. |
| Entity-level F1 | Правильная метрика для NER | Предсказанный span должен совпасть с настоящим точно. F1 по токенам завышает качество. |

## Further Reading

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — статья про BiLSTM-CRF. Каноническая.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — вводит шаблон классификации токенов, ставший стандартом.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — практический справочник по каждому атрибуту `Doc.ents` и `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — правильная библиотека метрик. Используйте всегда.

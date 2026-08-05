<!-- i18n:manual -->
# Subword-токенизация — BPE, WordPiece, Unigram, SentencePiece

> Пословные токенизаторы давятся незнакомыми словами. Посимвольные раздувают длину последовательности. Subword-токенизаторы берут середину. На них работает каждая современная LLM.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## The Problem

В вашем vocabulary 50 000 слов. Пользователь набирает "untokenizable". Токенизатор возвращает `[UNK]`. Теперь у модели нет вообще никакого сигнала об этом слове. Хуже того: документ на 90-м процентиле корпуса содержит 40 редких слов — то есть 40 выброшенных кусков информации на каждый документ.

> 🎒 **На пальцах.** Представьте словарь иностранного языка на 50 000 слов. Встретили слово, которого там нет, — и вы записываете в тетрадь просто «непонятное слово». Сорок раз на страницу. К концу страницы вы не понимаете ничего.

Subword-токенизация это решает. Частые слова остаются одним токеном. Редкие распадаются на осмысленные куски: `untokenizable` → `un`, `token`, `izable`. Обучающие данные покрывают всё, потому что любая строка в конечном счёте — это последовательность байтов.

> 🎒 **На пальцах.** Одно неизвестное слово превратилось в 3 знакомых куска. Модель видит `token` внутри и уже догадывается, о чём речь. Так же вы читаете незнакомое «расфасовочный»: корень «фасов» знаком, остальное — приставка и суффикс.

Каждая передовая LLM 2026 года работает на одном из трёх алгоритмов (BPE, Unigram, WordPiece), завёрнутом в одну из трёх библиотек (tiktoken, SentencePiece, HF Tokenizers). Выпустить языковую модель, не выбрав что-то из этого списка, невозможно.

## The Concept

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Начинаем с посимвольного vocabulary. Считаем каждую пару соседних символов. Самую частую пару сливаем (merge) в новый токен. Повторяем, пока не наберём нужный размер vocabulary. Доминирующий алгоритм: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

> 🎒 **На пальцах.** Это как стенография. Заметили, что «ст» встречается чаще всего — придумали для неё один значок. Потом «ени» — ещё значок. Каждый merge экономит место. Тысяча merge-шагов — и текст записывается вдвое короче.

**Byte-level BPE.** Тот же алгоритм, но поверх сырых байтов (256 базовых токенов), а не символов Unicode. Гарантирует ноль токенов `[UNK]` — кодируется любая последовательность байтов. GPT-2 использует 50 257 токенов (256 байтов + 50 000 merges + 1 специальный).

> 🎒 **На пальцах.** Проверьте арифметику: 256 + 50 000 + 1 = 50 257. Ровно столько строк в vocabulary GPT-2. Первые 256 — просто все возможные байты, поэтому что бы вы ни ввели, даже мусор из битого файла, `[UNK]` не появится никогда.

**Unigram.** Начинаем с огромного vocabulary. Каждому токену приписываем unigram-вероятность. Итеративно выбрасываем те токены, удаление которых меньше всего снижает log-правдоподобие корпуса. На инференсе алгоритм вероятностный: можно сэмплировать разные разбиения (полезно для аугментации данных через subword regularization). Используют T5, mBART, ALBERT, XLNet, Gemma.

> 🎒 **На пальцах.** BPE строит vocabulary снизу вверх, Unigram — сверху вниз. Как сборы в поход: один кладёт в рюкзак нужное по одной вещи, другой запихивает всё подряд и потом выбрасывает то, без чего проживёт. Итог похожий, путь противоположный.

**WordPiece.** Сливает те пары, которые сильнее всего повышают правдоподобие обучающего корпуса, а не просто самые частые. Используют BERT, DistilBERT, ELECTRA.

> 🎒 **На пальцах.** Разница с BPE в одной формуле. BPE выбирает пару по счёту «сколько раз встретилась». WordPiece делит этот счёт на частоты частей по отдельности. Если «th» и «e» и так постоянно ходят рядом случайно, BPE их сольёт, а WordPiece — не обязательно.

**SentencePiece vs tiktoken.** SentencePiece — это библиотека, которая *обучает* vocabulary (BPE или Unigram) прямо на сыром тексте Unicode, кодируя пробел символом `▁`. tiktoken — быстрый *энкодер* от OpenAI, работающий по уже готовым vocabulary; обучать он не умеет.

> 🎒 **На пальцах.** SentencePiece — фабрика линеек, tiktoken — сама линейка. Хотите померить текст готовой линейкой GPT-4 — берите tiktoken. Хотите линейку под свой язык или свой домен — идите на фабрику.

Практическое правило:

- **Training a new vocabulary:** SentencePiece (мультиязычность, не нужна предварительная токенизация) или HF Tokenizers.
- **Fast inference against GPT vocab:** tiktoken (cl100k_base, o200k_base).
- **Both:** HF Tokenizers — одна библиотека, и обучение, и продакшен.

> 🎒 **На пальцах.** Три строки — три разные задачи, и путать их дорого. Если вы кодируете текст для GPT-4 через свой SentencePiece, номера токенов совпадать не будут вообще, а счёт за API окажется совсем не тем, что вы посчитали.

```figure
bpe-merge
```

## Build It

### Step 1: BPE from scratch

Смотрите `code/main.py`. Цикл:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Алгоритм зашивает три факта. `</w>` отмечает конец слова, чтобы "low" (суффикс) и "lower" (префикс) не смешались. Взвешивание по частоте выводит частые пары в лидеры на ранних шагах. Список merges упорядочен — на инференсе merges применяются в том же порядке, что при обучении.

> 🎒 **На пальцах.** Порядок здесь важен так же, как в рецепте. Если merge номер 7 применить раньше merge номер 3, разбиение получится другое, и модель увидит незнакомые ID. Поэтому `merges` — это список, а не множество.

### Step 2: encode with the learned merges

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Наивная сложность O(n·|merges|). Продакшен-реализации (tiktoken, HF Tokenizers) используют поиск по рангу merge с приоритетными очередями и работают почти за линейное время.

> 🎒 **На пальцах.** Посчитайте на пальцах: слово из 12 символов и 50 000 merges — это 600 000 проверок ради одного слова. Терпимо для учебного примера, невозможно для потока в миллион запросов. Отсюда Rust внутри tiktoken.

### Step 3: SentencePiece in practice

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Обратите внимание: предварительная токенизация не нужна, пробел кодируется как `▁`, а `character_coverage` управляет тем, насколько упорно сохраняются редкие символы вместо замены на `<unk>`.

> 🎒 **На пальцах.** `character_coverage=0.9995` означает: покрываем 99,95 % всех символов корпуса, а самый редкий хвост (0,05 %) выбрасываем. Для английского это нормально — там букв мало. Для японского в комментарии стоит 0,995, потому что иероглифов тысячи и хвост гораздо длиннее.

### Step 4: tiktoken for OpenAI-compatible vocabs

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Только кодирование. Быстро (бэкенд на Rust). Точное совпадение с токенизацией GPT-4/5 — для подсчёта байтов, оценки стоимости и планирования бюджета контекстного окна.

> 🎒 **На пальцах.** «Hello, world!» — это 4 токена, а не 13 символов и не 2 слова. Правило-прикидка для английского: 1 токен ≈ 4 символа. Значит письмо на 4 000 символов съест примерно 1 000 токенов, и на этом строится вся арифметика счетов за API.

## Pitfalls that still ship in 2026

- **Tokenizer drift.** Обучали на vocabulary A, выкатили против vocabulary B. ID токенов другие, модель выдаёт мусор. Проверяйте хеш `tokenizer.json` в CI.
- **Whitespace ambiguity.** В BPE "hello" и " hello" дают разные токены. Всегда указывайте `add_special_tokens` и `add_prefix_space` явно.
- **Multilingual undertraining.** Перекошенные в сторону английского корпуса дают vocabulary, которые режут нелатинские письменности на в 5-10 раз большее число токенов. Один и тот же промпт на японском или арабском стоил в GPT-3.5 в 5-10 раз дороже. o200k_base частично это починил.
- **Emoji splits.** Одна эмодзи может занять 5 токенов. Учитывайте эмодзи, когда считаете бюджет контекста.

> 🎒 **На пальцах.** Смотрите на цену: текст на 1 000 токенов по-английски превращается в 5 000-10 000 токенов по-японски. Это буквально в 5-10 раз больше денег за тот же смысл. А одна эмодзи может стоить как целое слово — 5 токенов.

## Use It

Стек 2026 года:

| Situation | Pick |
|-----------|------|
| Обучение одноязычной модели с нуля | HF Tokenizers (BPE) |
| Обучение мультиязычной модели | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Обслуживание OpenAI-совместимого API | tiktoken (`o200k_base` для GPT-4+) |
| Vocabulary под домен (код, математика, белки) | Обучить свой BPE на доменном корпусе и слить с базовым vocabulary |
| Инференс на устройстве, маленькая модель | Unigram (меньшие vocabulary работают лучше) |

Размер vocabulary — это решение про масштаб, а не константа. Грубая эвристика: 32k для моделей меньше 1B параметров, 50-100k для 1-10B, 200k+ для мультиязычных и передовых.

> 🎒 **На пальцах.** Vocabulary — это компромисс. Больше токенов в словаре — короче тексты, но толще матрица эмбеддингов. Для модели на 1B параметров vocabulary в 200k съест заметную долю всех весов на одну только таблицу слов, и учиться будет нечему. Поэтому там и стоит 32k.

## Ship It

Сохраните как `outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

> 🎒 **На пальцах.** Обратите внимание на последнюю строку: одноязычный токенизатор меньше 16k помечается как подозрительный. Логика простая — если слов в словаре мало, каждое обычное слово будет резаться на 3-4 куска, тексты вырастут втрое, и вы заплатите за это и скоростью, и качеством.

## Exercises

1. **Easy.** Обучите BPE на 500 merges на крошечном корпусе из `code/main.py`. Закодируйте три отложенных слова. Сколько из них дали ровно 1 токен, а сколько больше одного?
2. **Medium.** Сравните число токенов на 100 предложениях из английской Википедии между `cl100k_base`, `o200k_base` и вашим SentencePiece BPE с vocab=32k. Приведите коэффициент сжатия для каждого.
3. **Hard.** Обучите на одном корпусе BPE, Unigram и WordPiece. Измерьте итоговую точность каждого на небольшом классификаторе тональности. Меняет ли выбор алгоритма результат больше чем на 1 пункт F1?

> 🎒 **На пальцах.** Подсказка к третьему заданию: скорее всего разница окажется меньше 1 пункта F1, и это нормальный результат, а не провал эксперимента. Выбор алгоритма почти всегда важнее для скорости и длины текста, чем для качества. Настоящую разницу вы увидите в другом числе — среднем количестве токенов на слово.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Жадное слияние самых частых пар символов, пока не набран нужный размер vocabulary. |
| Byte-level BPE | No unknown tokens ever | BPE поверх сырых 256 байтов; так делают GPT-2 и Llama. |
| Unigram | Probabilistic tokenizer | Отсекает лишнее из большого набора кандидатов по log-правдоподобию; используют T5 и Gemma. |
| SentencePiece | The whitespace one | Библиотека, обучающая BPE/Unigram на сыром тексте; пробел кодируется как `▁`. |
| tiktoken | The fast one | BPE-энкодер OpenAI на Rust для готовых vocabulary. Обучения нет. |
| Merge list | The magic numbers | Упорядоченный список merges вида `(a, b) → ab`; на инференсе применяется по порядку. |
| Character coverage | How rare is too rare? | Доля символов обучающего корпуса, которую токенизатор обязан покрыть; типично ~0,9995. |

## Further Reading

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — статья про BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — статья про Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — про библиотеку.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — короткий справочник.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — рецепты и список кодировок.

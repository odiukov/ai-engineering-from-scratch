<!-- i18n:manual -->
# Строим токенизатор с нуля

> Урок 01 дал вам игрушку. Этот урок даёт оружие.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## Learning Objectives

- Собрать боевой BPE-токенизатор, который умеет в Unicode, нормализацию пробелов и специальные токены
- Реализовать byte-level fallback, чтобы токенизатор кодировал любой ввод (включая эмодзи, CJK и код) без unknown-токенов
- Добавить регулярки предварительной токенизации, которые режут текст по границам слов до применения BPE-merges
- Обучить свой токенизатор на корпусе и сравнить его коэффициент сжатия с tiktoken на мультиязычном тексте

## The Problem

Ваш BPE-токенизатор из урока 01 работает на английском тексте. Теперь киньте в него японский. Или эмодзи. Или код на Python со смесью табов и пробелов.

Он ломается.

Не потому, что BPE неправильный, — потому что реализация неполная. Продакшен-токенизатор работает с сырыми байтами в любой кодировке, нормализует Unicode перед разбиением, управляет специальными токенами, которые никогда не сливаются, соединяет предварительную токенизацию с subword-разбиением и делает всё это достаточно быстро, чтобы не стать узким местом обучающего пайплайна на 15 триллионов токенов.

У токенизатора GPT-2 — 50 257 токенов. У Llama 3 — 128 256. У GPT-4 — примерно 100 000. Это не игрушечные числа. Таблицы merges за этими vocabulary обучались на сотнях гигабайт текста, а вся обвязка вокруг — нормализация, предварительная токенизация, вставка специальных токенов, форматирование chat template — и отличает токенизатор, который справляется с «hello world», от токенизатора, который справляется со всем интернетом.

Вот эту обвязку вы и будете строить.

> 🎒 **На пальцах.** 50 257 против 128 256 — это не «чуть больше». Vocabulary Llama 3 в 2,5 раза шире, поэтому тот же текст она пишет заметно меньшим числом токенов. Как алфавит против иероглифов: знаков больше, но слов на страницу влезает больше.

## The Concept

### The Full Pipeline

Продакшен-токенизатор — это не один алгоритм. Это конвейер из пяти стадий, и каждая решает свою задачу.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

У каждой стадии своя работа:

| Stage | What It Does | Why It Matters |
|-------|-------------|----------------|
| Normalize | NFKC Unicode, опционально нижний регистр, опционально снятие диакритики | Лигатура "fi" (U+FB01) превращается в "fi" (два символа). Без этого одно и то же слово получает разные токены. |
| Pre-Tokenize | Режет текст на куски до BPE | Не даёт BPE сливать через границы слов. "the cat" не должно порождать токен "e c". |
| BPE Merge | Применяет выученные правила merge к последовательностям байтов | Собственно сжатие. Превращает сырые байты в subword-токены. |
| Special Tokens | Вставляет [BOS], [EOS], [PAD], маркеры chat template | У этих токенов фиксированные ID. Они никогда не участвуют в BPE-merges. Модели они нужны для структуры. |
| ID Mapping | Переводит строки токенов в целые ID | Модель видит числа, а не строки. |

> 🎒 **На пальцах.** Пять стадий — как конвейер на кухне: помыть, порезать, потушить, посолить, разложить по тарелкам. Пропустите нормализацию — и слово «file» с лигатурой "fi" получит совсем другие ID, чем обычное «file». Для модели это два разных слова, хотя на экране они выглядят одинаково.

### Byte-Level BPE

Токенизатор из урока 01 работал на байтах UTF-8. Это было правильное решение. Но мы пропустили важное: что происходит, когда эти байты — не валидный UTF-8?

Byte-level BPE решает это тем, что считает валидным токеном каждое возможное значение байта (0-255). Базовое vocabulary — ровно 256 записей. Любой файл — текстовый, бинарный, битый — можно токенизировать, не породив ни одного unknown-токена.

GPT-2 добавила трюк: каждый байт отображается в печатный символ Unicode, чтобы vocabulary оставалось читаемым для человека. Байт 0x20 (пробел) становится символом "Ġ" в их таблице. Это чистая косметика. Алгоритму всё равно.

Настоящая сила: byte-level BPE тянет любой язык на Земле. Китайский иероглиф — 3 байта UTF-8. Японский — 3-4 байта. Арабица, деванагари, эмодзи — всё это просто последовательности байтов. Алгоритм BPE находит закономерности в этих байтах ровно так же, как находит их в английских ASCII-байтах.

> 🎒 **На пальцах.** Посчитайте: 256 базовых токенов покрывают вообще всё, что может лежать в файле, — от буквы «a» до случайного мусора из повреждённого архива. Поэтому `[UNK]` тут физически невозможен. А «你好» — это 6 байтов, и токенизатор видит их так же спокойно, как видит "hello".

### Pre-Tokenization

Прежде чем BPE тронет ваш текст, его надо порезать на куски. Это не даёт алгоритму merge создавать токены, которые перепрыгивают через границы слов.

GPT-2 режет текст регуляркой:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Этот паттерн режет по сокращениям ("don't" превращается в "don" + "'t"), по словам с необязательным ведущим пробелом, по числам, по пунктуации и по пробелам. Ведущий пробел остаётся приклеенным к слову — поэтому "the cat" даёт [" the", " cat"], а не ["the", " ", "cat"].

Llama использует SentencePiece, который обходится вообще без регулярок. Он смотрит на сырой поток байтов как на одну длинную последовательность и позволяет BPE самому нащупать границы. Это проще, но даёт BPE больше свободы создавать межсловные токены.

Выбор важен. Регулярка GPT-2 не даёт токенизатору выучить, что "the" в конце одного слова и "the" в начале следующего надо слить. SentencePiece это разрешает, из-за чего иногда получается более эффективное сжатие, но менее понятные токены.

> 🎒 **На пальцах.** Ведущий пробел — не мелочь. " the" и "the" — это два разных токена с разными ID. Именно поэтому промпт, где вы случайно поставили пробел в конце строки, иногда ведёт себя не так, как промпт без него: модель видит другую последовательность чисел.

### Special Tokens

Каждый продакшен-токенизатор резервирует ID токенов под структурные маркеры:

| Token | Purpose | Used By |
|-------|---------|---------|
| `[BOS]` / `<s>` | Начало последовательности | Llama 3, GPT |
| `[EOS]` / `</s>` | Конец последовательности | All models |
| `[PAD]` | Добивка для выравнивания батча | BERT, T5 |
| `[UNK]` | Неизвестный токен (byte-level BPE его убирает) | BERT, WordPiece |
| `<\|im_start\|>` | Начало сообщения в чате | ChatGPT, Qwen |
| `<\|im_end\|>` | Конец сообщения в чате | ChatGPT, Qwen |
| `<\|user\|>` | Маркер хода пользователя | Llama 3 |
| `<\|assistant\|>` | Маркер хода ассистента | Llama 3 |

Специальные токены BPE никогда не разбивает. Их находят точным совпадением ещё до запуска алгоритма merge, подменяют их фиксированным ID, а окружающий текст токенизируют как обычно.

> 🎒 **На пальцах.** Если бы `<|im_end|>` шёл через BPE, он развалился бы на куски вроде "<", "|im", "_end", "|>" — и модель не увидела бы сигнала «сообщение закончилось». Поэтому его вырезают из текста заранее и подставляют один готовый номер.

### Chat Templates

Вот здесь путается большинство людей и ломается большинство реализаций.

Когда вы отправляете сообщения чат-модели, API принимает список сообщений:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

Модель не видит JSON. Она видит плоскую последовательность токенов. Chat template превращает сообщения в эту плоскую последовательность с помощью специальных токенов. У каждой модели это устроено по-своему:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Ошибётесь в шаблоне — модель начнёт выдавать мусор. Её обучали на одном строго определённом формате. Любое отклонение — пропущенный перевод строки, перепутанный токен, лишний пробел — выбрасывает вход за пределы обучающего распределения.

> 🎒 **На пальцах.** Сравните два шаблона выше: у Llama 3 после `<|end_header_id|>` идёт пустая строка, у ChatGPT — нет. Один этот перевод строки решает, получите вы осмысленный ответ или кашу. Это как код домофона: правильные цифры не в том порядке дверь не открывают.

### Speed

Python слишком медленный для продакшен-токенизации.

tiktoken (OpenAI) написан на Rust с биндингами в Python. HuggingFace tokenizers — тоже Rust. SentencePiece — C++. Они дают ускорение в 10-100 раз против чистого Python.

Для масштаба: токенизировать 15 триллионов токенов для предобучения Llama 3 со скоростью миллион токенов в секунду (это быстрый Python) — 174 дня. Со скоростью 100 миллионов токенов в секунду (Rust) — 1,7 дня.

Вы строите на Python, чтобы понять алгоритм. В продакшене вы возьмёте скомпилированную реализацию и будете трогать только Python-обёртку.

> 🎒 **На пальцах.** 174 дня против 1,7 дня — это разница между «модель вышла в этом году» и «модель не вышла». Проверьте деление сами: 15 000 000 000 000 / 1 000 000 = 15 миллионов секунд, а это примерно 174 суток.

```figure
weight-tying
```

## Build It

### Step 1: Byte-Level Encoding

Фундамент. Превратить любую строку в последовательность байтов, отобразить каждый байт в печатный символ для показа и проделать обратный путь.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Прогоните на мультиязычном тексте и посмотрите на счётчики байтов:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" — 5 байтов. "你好" — 6 байтов (по 3 на символ). Эмодзи с огнём — 4 байта. Byte-level токенизатору всё равно, какой это язык. Байты есть байты.

> 🎒 **На пальцах.** Обратите внимание на строку "Mixed": 8 символов на экране превращаются в 5 + 6 + 4 = 15 байтов. Символ и байт — не одно и то же, и путать их дорого: длина строки в Python ничего не говорит о том, сколько токенов вы заплатите.

### Step 2: Pre-Tokenizer with Regex

Режем текст на куски регуляркой GPT-2. Каждый кусок BPE токенизирует независимо от остальных.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

Модуль `regex` поддерживает Unicode-свойства (`\p{L}` — буквы, `\p{N}` — цифры). Стандартный модуль `re` — нет, поэтому мы откатываемся на ASCII-классы символов. Для боевого мультиязычного токенизатора ставьте `regex`.

Попробуйте:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Ведущий пробел остаётся приклеенным к слову. Сокращения режутся по апострофу. Пунктуация становится отдельным куском. BPE никогда не сольёт токены через эти границы.

> 🎒 **На пальцах.** Посмотрите на вывод: "Don't" превратилось в " Don" и "'t". Апостроф уехал к окончанию, а не остался при корне. Так же по-русски было бы «не» + «льзя»: кусок странный на вид, но алгоритму важно, что он повторяется в тысячах слов.

### Step 3: BPE on Byte Sequences

Тот же ядровый алгоритм из урока 01, но теперь он работает на предварительно нарезанных кусках по отдельности.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

> 🎒 **На пальцах.** `apply_merge` идёт по последовательности слева направо и, найдя пару, прыгает через два элемента (`i += 2`), а не через один. Иначе в "aaa" при merge пары (a, a) он слил бы первые две «a», а потом попытался слить результат с третьей и посчитал бы одну и ту же «a» дважды.

### Step 4: Special Token Handling

Специальным токенам нужны точное совпадение и фиксированные ID. Они обходят BPE стороной.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

> 🎒 **На пальцах.** Ключевая деталь — `sorted(..., key=len, reverse=True)`: сначала длинные токены, потом короткие. Если бы у вас были `<|im_end|>` и `<|im|>`, а порядок был обратный, регулярка отхватила бы короткий кусок и оставила хвост «_end|>» болтаться в тексте.

### Step 5: Full Tokenizer Class

Собираем всё в цепочку: нормализация, разбиение по специальным токенам, предварительная токенизация, BPE-merge, отображение в ID.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

> 🎒 **На пальцах.** Проследите за `next_id`: он стартует с 256, потому что первые 256 номеров уже заняты байтами. Пятьдесят merges — и он дорос до 306, а первый специальный токен получит номер 306. Никаких коллизий: каждый ID выдаётся ровно один раз и навсегда.

### Step 6: Multilingual Test

Настоящая проверка. Кидаем в него английский, китайский, эмодзи и код.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Китайские иероглифы дают по 3 байта каждый. Эмодзи даёт 4 байта. Ничто из этого токенизатор не роняет. Ничто не порождает unknown-токенов. Вот в чём сила byte-level BPE.

> 🎒 **На пальцах.** Обучали на английском корпусе про лису и код на Python — а «你好世界» всё равно закодировалось и раскодировалось обратно без потерь. Просто на него ушло 12 токенов (4 иероглифа по 3 байта) вместо одного-двух: ни один merge под китайский не выучен, поэтому байты идут поштучно.

## Use It

### Comparing Real Tokenizers

Загрузите настоящие токенизаторы Llama 3, GPT-4 и Mistral. Посмотрите, как каждый обходится с одним и тем же мультиязычным абзацем.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Вы увидите разное число токенов на одном и том же тексте. Llama 3 со 128K vocabulary агрессивнее сливает частые последовательности. GPT-4 со 100K сидит посередине. Mistral с 32K выдаёт больше токенов, зато у него меньше слой эмбеддингов.

Компромисс всегда один и тот же: больше vocabulary — короче последовательности, но больше параметров.

> 🎒 **На пальцах.** Прикиньте цену этого компромисса. Vocabulary на 128K при размерности модели 4096 — это 128 000 × 4096 ≈ 525 миллионов параметров только на таблицу эмбеддингов. Для модели на 8B это ощутимый кусок, и платите вы за него более короткими последовательностями.

## Ship It

Этот урок производит промпт для сборки и отладки продакшен-токенизаторов. Смотрите `outputs/prompt-tokenizer-builder.md`.

## Exercises

1. **Easy:** Добавьте метод `get_token_bytes(id)`, который показывает сырые байты для любого ID токена. С его помощью посмотрите, что на самом деле означают ваши самые частые слитые токены.
2. **Medium:** Реализуйте пре-токенизатор в стиле Llama: режет по пробелам и цифрам, но сохраняет ведущие пробелы. Сравните его vocabulary с подходом на регулярке GPT-2 на одном и том же корпусе.
3. **Hard:** Добавьте метод chat template, который берёт список сообщений `{"role": ..., "content": ...}` и выдаёт корректную последовательность токенов для формата чата Llama 3. Проверьте его против реализации HuggingFace.

> 🎒 **На пальцах.** Начните с первого задания — оно занимает одну строку, но объясняет остальные. Выведите байты пятидесяти выученных merges и вы увидите там " the", " model", "def " — токенизатор сам нашёл в корпусе самые ходовые куски, никто ему их не подсказывал.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Byte-level BPE | "Токенизатор, работающий на байтах" | BPE с базовым vocabulary из 256 значений байта — тянет любой ввод без unknown-токенов |
| Pre-tokenization | "Разбиение до BPE" | Разбиение регуляркой или правилами, не дающее BPE сливать через границы слов |
| NFKC normalization | "Причёсывание Unicode" | Каноническая декомпозиция и следом композиция совместимости — лигатура "fi" становится "fi", полноширинная "Ａ" становится "A" |
| Chat template | "Как сообщения превращаются в токены" | Точный формат превращения списка сообщений role/content в плоскую последовательность токенов — свой у каждой модели и обязан совпадать с форматом обучения |
| Special tokens | "Управляющие токены" | Зарезервированные ID токенов в обход BPE — [BOS], [EOS], [PAD], маркеры чата — находятся точным совпадением до merge |
| Fertility | "Токенов на слово" | Отношение выходных токенов к входным словам — 1,3 для английского в GPT-4, 2-3 для корейского; больше значит впустую потраченный контекст |
| tiktoken | "Токенизатор OpenAI" | Реализация BPE на Rust с биндингами в Python — в 10-100 раз быстрее чистого Python |
| Merge table | "Vocabulary" | Упорядоченный список merges байтовых пар, выученных при обучении — это И ЕСТЬ выученное знание токенизатора |

## Further Reading

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) — реализация BPE на Rust, которую используют GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) — библиотека токенизаторов на Rust с поддержкой BPE, WordPiece, Unigram
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) — подробности про vocabulary на 128K и обучение токенизатора
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) — токенизация без привязки к языку
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) — оригинальное отображение байтов в Unicode

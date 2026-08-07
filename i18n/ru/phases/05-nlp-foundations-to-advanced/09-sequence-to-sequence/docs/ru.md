<!-- i18n:manual -->
# Модели sequence-to-sequence

> Две RNN, изображающие переводчика. Тот тупик, в который они упираются, и есть причина существования attention.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 minutes

## The Problem

Классификация переводит последовательность переменной длины в одну метку. Перевод переводит последовательность переменной длины в другую последовательность переменной длины. Вход и выход живут в разных словарях, возможно на разных языках, и никто не обещает, что длины совпадут.

Архитектура seq2seq (Sutskever, Vinyals, Le, 2014) взломала это нарочито простым рецептом. Две RNN. Одна читает исходное предложение и выдаёт context vector фиксированного размера. Вторая читает этот вектор и порождает целевое предложение токен за токеном. Тот же код, что вы писали в уроке 08, склеенный по-другому.

Изучать это стоит по двум причинам. Первая: тупик context vector — самый полезный с педагогической точки зрения провал во всём NLP. Он объясняет, зачем нужны attention и трансформеры. Вторая: рецепт обучения (teacher forcing, scheduled sampling, beam search на инференсе) до сих пор применим к любой современной генеративной системе, включая LLM.

> 🎒 **На пальцах.** Представьте переводчика-синхрониста, которому запретили записывать: он дослушивает фразу до конца, держит её целиком в голове одним «впечатлением», и только потом начинает говорить. С фразой из пяти слов справится любой. С абзацем на сорок слов — уже нет. Весь урок про то, как выглядит этот предел в коде.

## The Concept

**Encoder.** RNN, которая читает исходное предложение. Её финальный hidden state и есть **context vector** — сводка всего входа фиксированного размера. Ничего, кроме источника, вроде бы не теряется.

**Decoder.** Другая RNN, инициализированная этим context vector. На каждом шаге она получает на вход ранее сгенерированный токен и выдаёт распределение по целевому словарю. Берёте сэмпл или argmax, чтобы выбрать следующий токен. Подаёте его обратно. Повторяете, пока не выпадет токен `<EOS>` или не упрётесь в максимальную длину.

> 🎒 **На пальцах.** Encoder — как человек, прочитавший письмо и записавший его суть на одном стикере. Decoder — как второй человек, который видит только стикер и пишет по нему письмо на другом языке. Если суть на стикере — 256 чисел, то и на предложение из 5 слов, и на предложение из 40 у вас ровно 256 чисел. Больше не выделят.

**Training:** кросс-энтропийный лосс на каждом шаге декодера, просуммированный по последовательности. Обычный backprop through time через обе сети.

**Teacher forcing.** На обучении вход decoder на шаге `t` — это *настоящий* токен на позиции `t-1`, а не собственное предыдущее предсказание модели. Это стабилизирует обучение; без этого ранние ошибки накапливаются лавиной и модель не выучивается вовсе. На инференсе приходится использовать собственные предсказания модели, поэтому между обучением и инференсом всегда есть разрыв распределений. Этот разрыв называется **exposure bias**.

> 🎒 **На пальцах.** Teacher forcing — это ученик, решающий задачи, где после каждого шага учитель подсказывает правильный промежуточный ответ. На контрольной подсказок нет, и ученик впервые встречает свои же ошибки. Отсюда и exposure bias: модель ни разу не тренировалась выбираться из ямы, которую сама выкопала.

**The bottleneck.** Всё, что encoder узнал об источнике, обязано пролезть в один context vector. Длинные предложения теряют детали. Редкие слова смазываются. Перестановки (chat noir против black cat) приходится запоминать, а не вычислять.

Attention (урок 10) чинит это, позволяя decoder смотреть на *каждый* hidden state энкодера, а не только на последний. В этом и весь смысл.

> 🎒 **На пальцах.** Считайте буквально: hidden state размером 256 чисел по 4 байта — это 1 КБ. В этот килобайт нужно упаковать предложение любой длины. Для пяти слов запас огромный, для сорока — уже теснее, чем в архиве с потерями. Attention не увеличивает стикер, он просто разрешает decoder заглянуть обратно в оригинал.

```figure
lstm-gates
```

## Build It

### Step 1: an encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` имеет форму `[batch, seq_len, hidden_dim]` — по одному hidden state на каждую позицию входа. `hidden` имеет форму `[1, batch, hidden_dim]` — только последний шаг. В уроке 08 говорилось «агрегируйте outputs для классификации». Здесь мы оставляем последний hidden state как context vector, а пошаговые outputs игнорируем.

> 🎒 **На пальцах.** Вот здесь и рождается проблема. Для входа из 20 токенов при `hidden_dim=256` энкодер посчитал 20 × 256 = 5120 чисел, а дальше передаёт только 256 из них. 96% посчитанной информации выбрасывается в мусор. Attention в уроке 10 просто перестанет её выбрасывать. Заметьте также `padding_idx=0`: индекс 0 зарезервирован под padding и его embedding не обучается.

### Step 2: a decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder вызывают по одному шагу за раз. Вход: батч из одиночных токенов и текущий hidden state. Выход: логиты по словарю для следующего токена и обновлённый hidden state.

> 🎒 **На пальцах.** Слой `self.fc` превращает 256 чисел hidden state в оценку для каждого слова словаря. Если целевой словарь на 10 000 слов, это матрица 256 × 10 000 = 2.56 миллиона весов — обычно самая тяжёлая часть модели. Именно поэтому в реальных системах словарь режут до subword-токенов, а не хранят все словоформы.

### Step 3: training loop with teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Две ручки, которые стоит назвать. `ignore_index=0` пропускает лосс на padding-токенах. `teacher_forcing_ratio` — вероятность взять на шаге настоящий токен вместо предсказания модели. Начинайте высоко — с дефолтных `0.9` выше или с 1.0 для полного teacher forcing — и по ходу обучения снижайте примерно до 0.5, чтобы закрыть разрыв exposure bias.

> 🎒 **На пальцах.** При `teacher_forcing_ratio=0.9` модель на девяти шагах из десяти получает подсказку и на одном идёт сама. В целевой последовательности из 20 токенов это примерно 2 самостоятельных шага за проход. Без `ignore_index=0` модель бы старательно училась предсказывать padding — а его в батче из коротких и длинных предложений бывает больше половины позиций.

### Step 4: inference loop (greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50, pad_id=0):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    finished = torch.zeros(batch_size, 1, dtype=torch.bool)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        finished = finished | (next_token == eos_id)
        if finished.all():
            break
        output_ids.append(next_token.masked_fill(finished, pad_id))
        input_token = next_token
    if not output_ids:
        return torch.zeros(batch_size, 0, dtype=torch.long)
    return torch.cat(output_ids, dim=1)
```

Две детали, на которых на батче легко ошибиться. Обновляйте `finished` *до* того, как добавили токен в `output_ids`, — тогда сам `<EOS>` никогда не попадёт в возвращаемый тензор. И маскируйте уже завершённые строки в `pad_id`, иначе последовательность, закончившаяся на шаге 3, будет продолжать выдавать мусор, пока не финиширует самая медленная строка батча. Без этой маски `.all()` покупает вам только более короткий цикл, но не корректный выход.

Жадное декодирование выбирает самый вероятный токен на каждом шаге. Оно может уехать не туда: раз выбрав токен, вы уже не можете его отменить. **Beam search** держит живыми топ-`k` частичных последовательностей и в конце выбирает завершённую с лучшей оценкой. Стандартная ширина луча — 3-5.

> 🎒 **На пальцах.** Жадное декодирование — как ехать по навигатору, каждый раз выбирая улицу, которая сейчас выглядит свободнее. Beam search шириной 3 ведёт три маршрута одновременно и в конце берёт лучший целиком. Цена — ровно в 3 раза больше вычислений. Классический пример: два токена с вероятностями 0.6 и 0.4, но за токеном 0.4 идёт продолжение на 0.9, а за 0.6 — только на 0.3. Жадный алгоритм получит 0.6 × 0.3 = 0.18, beam search — 0.4 × 0.9 = 0.36.

### Step 5: the bottleneck, demonstrated

Обучите модель на игрушечной задаче копирования: вход `[a, b, c, d, e]`, выход `[a, b, c, d, e]`. Увеличивайте длину последовательности. Смотрите на точность. Числа ниже — это вывод `code/main.py`, который имитирует context vector фиксированного размера по-дешёвому, без всякого обучения: он проверяет, получает ли настоящий источник оценку выше, чем случайная последовательность той же длины.

```
seq_len=5   copy accuracy: 89%
seq_len=10  copy accuracy: 83%
seq_len=20  copy accuracy: 69%
seq_len=40  copy accuracy: 53%
seq_len=80  copy accuracy: 51%
```

Порог случайного угадывания в этой задаче — 50%, так что к длине 80 context vector уже не несёт практически никакого полезного сигнала. Один hidden state GRU не может без потерь запомнить вход из 40 токенов. Информация есть на каждом шаге энкодера, но decoder видит только последнее состояние. Attention чинит это напрямую.

> 🎒 **На пальцах.** Задача элементарная: повтори то, что услышал. Выбор здесь бинарный — «настоящая последовательность или случайная?» — поэтому монетка дала бы ровно 50%, и отсчитывать надо от этой планки, а не от нуля. При длине 5 выходит 89%: заметно лучше монетки. При длине 40 — 53%, при 80 — 51%, то есть уже почти случайность: от входа в context vector не осталось ничего, за что можно зацепиться. И это на копировании, где вообще не нужно ничего понимать. Если так рушится копирование, представьте, что происходит с переводом.

## Use It

В PyTorch есть `nn.Transformer` и seq2seq-шаблоны на базе `nn.LSTM`. Библиотека `transformers` от Hugging Face поставляет готовые encoder-decoder модели (BART, T5, mBART, NLLB), обученные на миллиардах токенов.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Современные encoder-decoder отказались от RNN в пользу трансформеров. Общая форма (encoder, decoder, генерация токен за токеном) в точности та же, что в статье про seq2seq 2014 года. Отличается механизм внутри каждого блока.

> 🎒 **На пальцах.** Обратите внимание на `num_beams=4` — это тот самый beam search, только вам его уже написали. И на `max_new_tokens=50`: без ограничения генерация может зациклиться и печатать одно и то же до бесконечности. Две строки кода вместо всего, что вы писали выше руками.

### When to still reach for RNN-based seq2seq

Для новых проектов — практически никогда. Конкретные исключения:

- Потоковый перевод, где вход поступает по одному токену и память ограничена.
- Генерация текста на устройстве, где расход памяти трансформера неприемлем.
- Педагогика. Понять тупик encoder-decoder — самый короткий путь к пониманию того, почему победили трансформеры.

> 🎒 **На пальцах.** Ключевое слово — «ограниченная память». RNN держит одно состояние на 256 чисел независимо от того, обработала она 10 токенов или 10 000. Трансформеру нужно хранить весь контекст: 10 000 токенов — это 10 000 наборов ключей и значений. На телефоне разница между константой и растущим списком решает всё.

### Exposure bias and its mitigations

- **Scheduled sampling.** Снижайте teacher forcing ratio по ходу обучения, чтобы модель научилась выбираться из собственных ошибок.
- **Minimum risk training.** Обучайте на BLEU уровня предложения вместо кросс-энтропии уровня токена. Ближе к тому, что вам на самом деле нужно.
- **Reinforcement learning fine-tuning.** Награждайте генератор последовательности по метрике. Используется в современном RLHF для LLM.

Все три подхода применимы и к генерации на трансформерах.

> 🎒 **На пальцах.** Разница между вторым и первым пунктом — в том, что вы измеряете. Кросс-энтропия наказывает за каждое слово не на своём месте, даже если перевод верный по смыслу. BLEU смотрит на предложение целиком. Это как оценивать сочинение по совпадению с эталоном буква в букву против оценки за общий смысл.

## Ship It

Сохраните как `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

> 🎒 **На пальцах.** Порог «миллион параллельных примеров» — это не придирка. Оригинальная статья про seq2seq обучалась на 12 миллионах пар предложений. Если у вас 10 тысяч пар, вы дообучаете готовый чекпойнт, а не учите с нуля: разница в качестве будет в разы, а не в проценты.

## Exercises

1. **Easy.** Реализуйте игрушечную задачу копирования. Обучите GRU seq2seq на парах, где выход равен входу. Измерьте точность на длинах 5, 10, 20. Воспроизведите тупик.
2. **Medium.** Добавьте beam search с шириной луча 3. Измерьте BLEU на небольшом параллельном корпусе в сравнении с жадным декодированием. Опишите, где beam search выигрывает (обычно на последних токенах), а где разницы нет.
3. **Hard.** Дообучите `facebook/bart-base` на датасете из 10 тысяч пар перефразировок. Сравните выход дообученной модели с beam-4 против базовой модели на отложенных входах. Приведите BLEU и разберите 10 качественных примеров.

> 🎒 **На пальцах.** Подсказка к первому заданию: начните с длины 5 и словаря из 10 символов, чтобы модель обучалась за минуты. Мерить будете долю правильно скопированных токенов у настоящей обученной модели, а в таблице из Step 5 стоят числа дешёвой симуляции с бинарным выбором — сравнивать их напрямую нельзя. Ориентир для обученной модели: на длине 5 она должна выходить за 90%, и если нет, ищите баг, а не тупик архитектуры — чаще всего это перепутанные оси в `hidden` или забытый `<BOS>` на первом шаге decoder. Настоящий провал должен проявиться только к длине 20-40.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Encoder | Входная RNN | Читает источник. Выдаёт hidden state на каждый шаг и финальный context vector. |
| Decoder | Выходная RNN | Инициализируется из context vector. Порождает целевые токены по одному. |
| Context vector | Сводка | Финальный hidden state энкодера. Фиксированного размера. Тот самый тупик, который решает attention. |
| Teacher forcing | Подавать настоящие токены | На обучении подавать истинный предыдущий токен. Стабилизирует обучение. |
| Exposure bias | Разрыв между обучением и тестом | Модель училась на настоящих токенах и ни разу не пробовала выбираться из собственных ошибок. |
| Beam search | Декодирование получше | Держать живыми топ-k частичных последовательностей на каждом шаге вместо жадного выбора. |

## Further Reading

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — оригинальная статья про seq2seq. Четыре страницы.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — здесь появились GRU и сама схема encoder-decoder.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — статья про attention. Читайте сразу после этого урока.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — код seq2seq с attention, который можно собрать своими руками.

<!-- i18n:manual -->
# CNN и RNN для текста

> Свёртки выучивают n-граммы. Рекуррентность помнит. Обе вытеснены attention. Обе всё ещё важны на слабом железе.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## The Problem

TF-IDF и Word2Vec давали плоские векторы, которые игнорировали порядок слов. Классификатор на них не отличал `dog bites man` от `man bites dog`. А порядок слов иногда и несёт весь сигнал.

Эту дыру до появления трансформеров закрывали два семейства архитектур.

**Convolutional nets for text (TextCNN).** Одномерные свёртки поверх последовательности из embedding слов. Фильтр шириной 3 — это обучаемый детектор триграмм: он накрывает три слова и выдаёт оценку. Складывайте разные ширины (2, 3, 4, 5), чтобы ловить паттерны разного масштаба. Max-pooling сводит всё к представлению фиксированного размера. Плоско, параллельно, быстро.

**Recurrent nets (RNN, LSTM, GRU).** Обрабатывают токены по одному, храня hidden state, который несёт информацию вперёд. Последовательно, с памятью, с любой длиной входа. Господствовали в моделировании последовательностей с 2014 по 2017 год, а потом случился attention.

Этот урок строит и то и другое, а затем называет тот провал, из-за которого появился attention.

> 🎒 **На пальцах.** Разница между `dog bites man` и `man bites dog` — одна перестановка, но газетный заголовок совершенно разный. Мешок слов видит в обоих случаях одни и те же три слова и одинаковый вектор. Это как судить о книге по алфавитному списку слов из неё: словарь тот же, смысл потерян.

## The Concept

**TextCNN** (Kim, 2014). Токены превращаются в embedding. Одномерная свёртка ширины `k` скользит фильтром по подряд идущим `k`-граммам эмбеддингов и строит карту признаков. Глобальный max-pooling по этой карте выбирает самую сильную активацию. Выходы max-pooling от нескольких ширин фильтра склеиваются. Дальше — голова классификатора.

> 🎒 **На пальцах.** Свёртка по тексту — это шаблон, приложенный к соседним словам. Разные ширины ловят разное: 2 — устойчивые пары вроде «very good», 4 — обороты подлиннее. Поэтому в коде ниже и стоит сразу набор `(2, 3, 4)`, а не одно число.

Почему это работает. Фильтр — это обучаемая n-грамма. Max-pooling не зависит от позиции, поэтому «not good» зажигает один и тот же признак и в начале отзыва, и в середине. Три ширины фильтра по 100 фильтров каждая дают 300 выученных детекторов n-грамм. Обучение параллельное, последовательной зависимости нет.

> 🎒 **На пальцах.** Фильтр ширины 3 — это лупа, которая видит ровно три слова подряд и кричит, если узнала свой паттерн. Проведите такую лупу по отзыву из 20 слов: получится 20 − 3 + 1 = 18 позиций, то есть 18 оценок. Max-pooling берёт из них одну — самую громкую. Ему всё равно, на каком слове фильтр сработал, важно только что сработал.

**RNN.** На каждом шаге `t` hidden state равен `h_t = f(W * x_t + U * h_{t-1} + b)`. Матрицы `W`, `U` и вектор `b` общие для всех шагов. Hidden state на шаге `T` — это сводка всего префикса. Для классификации агрегируйте `h_1 ... h_T` (максимум, среднее или последний).

> 🎒 **На пальцах.** Слово «общие» здесь главное: одна и та же матрица `W` применяется и к первому слову, и к сотому. Как один и тот же сотрудник обрабатывает всю стопку документов по одной инструкции. Поэтому RNN работает с текстом любой длины — новых весов на длинный текст не нужно.

Обычные RNN страдают от затухающих градиентов. **LSTM** добавляет гейты, которые решают, что забыть, что сохранить и что выдать, и это стабилизирует градиенты на длинных последовательностях. **GRU** упрощает LSTM до двух гейтов; работает похоже, а параметров меньше.

**Bidirectional RNNs** запускают одну RNN вперёд, другую назад и склеивают hidden state. Представление каждого токена видит и левый, и правый контекст. Для задач разметки это обязательно.

> 🎒 **На пальцах.** Hidden state — это блокнот, который вы дописываете после каждого слова, а старые страницы вырывать нельзя, только переписывать поверх. К концу длинного текста первые слова затираются. Гейты LSTM — это правило «эту страницу не трогать»: она доезжает до конца целой. Bidirectional — это второй читатель, который идёт от конца к началу, и в конце вы сравниваете два блокнота.

```figure
rnn-unroll
```

## Build It

### Step 1: TextCNN in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

Вызов `transpose(1, 2)` переставляет `[batch, seq_len, embed_dim]` в `[batch, embed_dim, seq_len]`, потому что `nn.Conv1d` считает каналами среднюю ось. Выход после pooling имеет фиксированный размер независимо от длины входа.

> 🎒 **На пальцах.** Посчитайте размер выхода на параметрах из кода: три ширины фильтра (2, 3, 4) по 64 фильтра дают 64 × 3 = 192 числа. Именно это число стоит на входе `nn.Linear(192, n_classes)`. И заметьте: 192 получится и для отзыва из 8 слов, и для отзыва из 800 — max-pooling выравнивает всё.

### Step 2: LSTM classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pooling по последовательности, а не по последнему состоянию. Для классификации max-pooling обычно выигрывает у последнего hidden state, потому что в длинной последовательности конец текста склонен доминировать в последнем состоянии.

> 🎒 **На пальцах.** Флаг `bidirectional=True` удваивает выход: `factor = 2`, и при `hidden_dim=128` в `nn.Linear` приходит 256 чисел. Половина — от прохода слева направо, половина — справа налево. Если поставить `False`, будет 128, и модель узнает про слово только то, что было до него.

### Step 3: the vanishing gradient demo (intuition)

Обычная RNN без гейтов не умеет учить дальние зависимости. Возьмём игрушечную задачу: определить, встретился ли токен `A` где-нибудь в последовательности. Если `A` стоит на позиции 1, а последовательность длиной 100 токенов, то градиент от лосса должен пройти назад через 99 умножений на рекуррентный вес. Если вес меньше 1, градиент затухает. Если больше 1 — взрывается.

```python
import math


def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

> 🎒 **На пальцах.** Проверьте цифру из комментария на калькуляторе: 0.9 в сотой степени ≈ 0.000027. Это как ксерокопировать копию сто раз подряд — на сотом листе не разобрать ни буквы. Обратно: возьмите вес 1.1, и 1.1¹⁰⁰ ≈ 13 780, градиент взорвался. Устойчивый коридор между этими двумя бедами исчезающе узкий.

LSTM чинит это через **cell state**, который проходит через сеть только с аддитивными взаимодействиями (forget gate масштабирует его умножением, но градиенты всё равно текут по этому «шоссе»). GRU делает похожее меньшим числом параметров. Оба дают устойчивое обучение на последовательностях в 100+ шагов.

Раз на этом утверждении и держится весь смысл LSTM, вот та ячейка, на которую оно опирается. Один шаг, без фреймворка:

```python
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def lstm_cell(x_t, h_prev, c_prev, W, U, b):
    """W: (4H, D), U: (4H, H), b: (4H,). Gates are stacked in the order i, f, g, o."""
    hidden_dim = h_prev.shape[0]
    z = W @ x_t + U @ h_prev + b
    i = sigmoid(z[0:hidden_dim])                      # сколько из кандидата записать
    f = sigmoid(z[hidden_dim:2 * hidden_dim])         # сколько от старого cell state оставить
    g = np.tanh(z[2 * hidden_dim:3 * hidden_dim])     # значения-кандидаты
    o = sigmoid(z[3 * hidden_dim:])                   # сколько от cell state показать наружу
    c_t = f * c_prev + i * g
    h_t = o * np.tanh(c_t)
    return h_t, c_t
```

Весь аргумент несёт одна строка: `c_t = f * c_prev + i * g`. Cell state движется вперёд **сложением**. Обычная RNN ставит между каждой парой шагов умножение на матрицу и сжимающую нелинейность, и произведение этих множителей — это и есть `0.9 ** 100` выше. Здесь же, когда forget gate стоит около единицы, путь по `c` почти равен умножению на 1 независимо от длины последовательности — поэтому градиент со шага 100 доезжает до шага 1, и от него что-то остаётся. Гейты обучаемые, то есть сеть сама решает по каждому токену, что сохранить, а вот этого обычной RNN не даст никакая аккуратная инициализация.

> 🎒 **На пальцах.** Ключевое слово — «аддитивно», и в коде выше это ровно одна строка: `c_t = f * c_prev + i * g`. Складывать безопасно: сто раз прибавить 0.1 к нулю — получите 10. Умножать опасно: сто раз умножить на 0.9 — получите почти ноль, те самые 2.7e-5 из блока выше. Cell state — это конвейерная лента, на которую информацию добавляют, а не перемножают заново на каждом шаге. Гейт `f` решает, сколько от ленты оставить: при `f` около 1 она доезжает до конца почти нетронутой, при `f` около 0 содержимое стирается — и это решение сеть принимает сама, отдельно для каждого токена.

### Step 4: why this still was not enough

Даже с LSTM оставались три проблемы.

1. **Sequential bottleneck.** Обучение RNN на последовательности длиной 1000 требует 1000 последовательных шагов вперёд и назад. Распараллелить по времени нельзя.
2. **Fixed-size context vector in encoder-decoder setups.** Декодер видит только последний hidden state энкодера, сжатый по всему входу. Длинные входы теряют детали. Урок 09 разбирает это напрямую.
3. **Distant-dependency accuracy ceiling.** LSTM обгоняют обычные RNN, но всё равно с трудом протаскивают конкретную информацию через 200+ шагов.

Attention решил все три. Трансформеры отказались от рекуррентности вовсе. Урок 10 — точка разворота.

> 🎒 **На пальцах.** Первая проблема — чистая арифметика времени. RNN на 1000 токенов делает 1000 шагов строго по очереди, даже если у вас видеокарта на 10 000 ядер: 999 из них ждут. Трансформер обрабатывает все 1000 позиций сразу. Вот почему обучение ускорилось не на проценты, а в десятки раз.

## Use It

`nn.LSTM`, `nn.GRU` и `nn.Conv1d` из PyTorch готовы к продакшену. Код обучения стандартный.

Hugging Face поставляет предобученные embedding, которые подключают как входной слой:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            pooled.append(F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2))
        return self.fc(torch.cat(pooled, dim=1))
```

> 🎒 **На пальцах.** Цикл `param.requires_grad = False` замораживает BERT: его 110 миллионов весов не обучаются, учатся только свёртки и линейный слой — примерно 200 тысяч параметров. Это как взять готовый переводчик и дописать к нему маленькую анкету вместо того, чтобы учить язык заново. Число 768 в `nn.Conv1d(768, ...)` — размерность выхода `bert-base`, менять его нельзя.

Чек-лист «когда это подходит под ваши ограничения».

- **Edge / on-device inference.** TextCNN с GloVe-эмбеддингами в 10-100 раз меньше трансформера. Если целевая платформа — телефон, это ваш стек.
- **Streaming / online classification.** RNN обрабатывает по одному токену; трансформеру нужна вся последовательность. Для текста, приходящего в реальном времени, LSTM всё ещё выигрывают.
- **Tiny models for baselines.** Быстрые итерации на новой задаче. TextCNN обучается на CPU за 5 минут.
- **Sequence labeling with limited data.** BiLSTM-CRF (урок 06) до сих пор продакшен-архитектура для NER на 1k-10k размеченных предложений.

Всё остальное уходит к трансформеру.

> 🎒 **На пальцах.** Прикиньте разницу в размере: TextCNN с GloVe — это порядка 5-20 МБ, BERT-base — около 440 МБ. Разница в 20-100 раз решает, влезет модель в мобильное приложение или нет. Пять минут обучения на CPU против часов на GPU — вторая причина начинать baseline именно с TextCNN.

## Ship It

Сохраните как `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

> 🎒 **На пальцах.** Порог «~500 размеченных примеров» здесь не случайный. Трансформер с сотнями миллионов параметров на 500 примерах просто запомнит их наизусть. Правило простое: сначала baseline, и только когда его кривая качества выходит на плато, доставайте тяжёлую артиллерию.

## Exercises

1. **Easy.** Обучите TextCNN на игрушечном датасете из 3 классов (данные придумайте сами). Проверьте, что ширины фильтров (2, 3, 4) обгоняют одну ширину (3) по среднему F1.
2. **Medium.** Реализуйте для LSTM-классификатора max-pooling, mean-pooling и последнее состояние. Сравните на небольшом датасете; опишите, какой pooling выиграл, и выдвиньте гипотезу почему.
3. **Hard.** Соберите BiLSTM-CRF теггер для NER (объедините урок 06 и этот). Обучите на CoNLL-2003. Сравните с baseline из одного CRF из урока 06 и с дообученным BERT. Приведите время обучения, память и F1.

> 🎒 **На пальцах.** Подсказка к первому заданию: три ширины дадут 64 × 3 = 192 признака против 64 у одной ширины. Разница будет заметнее всего на классах, которые различаются длинными оборотами: ширина 2 ловит «not good», а ширина 4 — «not that good at». Прогоните каждый вариант хотя бы 5 раз с разными seed — на игрушечных данных разброс легко перекроет реальный эффект.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TextCNN | CNN для текста | Стопка одномерных свёрток поверх embedding слов с глобальным max-pooling. Kim (2014). |
| RNN | Рекуррентная сеть | Hidden state обновляется на каждом шаге: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | RNN с гейтами | Добавляет input / forget / output гейты и cell state. Устойчиво учится на длинных последовательностях. |
| GRU | Упрощённая LSTM | Два гейта вместо трёх. Точность похожая, параметров меньше. |
| Bidirectional | В обе стороны | Прямая и обратная RNN, склеенные вместе. Каждый токен видит контекст с обеих сторон. |
| Vanishing gradient | Сигнал обучения умирает | Многократное умножение на веса меньше 1 в обычных RNN обнуляет градиенты ранних шагов. |

## Further Reading

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — статья про TextCNN. Восемь страниц. Читается легко.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — статья про LSTM. Неожиданно внятная.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — те самые схемы, которые сделали LSTM понятными для всех.

<!-- i18n:manual -->
# Gradient checkpointing и пересчёт активаций

> Backprop хранит каждую промежуточную активацию. На 70B параметров и контексте 128K это 3 ТБ активаций на один ранг. Checkpointing меняет FLOPs на память: не сохранять, а пересчитывать. Вопрос лишь в том, какие участки выбрасывать, и ответ не «все подряд».

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## The Problem

При обучении трансформера для каждого слоя сохраняются входы всех операций, которые дифференцируются на backward: входы внимания, проекции Q/K/V, выход softmax, входы FFN, выходы нормализаций и остаточный поток. Для слоя со скрытым размером `d`, длиной последовательности `L` и батчем `B` это порядка `12 * B * L * d` чисел на слой.

При `d=8192, L=8192, B=1` получается 800 МБ на слой в BF16. Модель из 64 слоёв — это 51 ГБ активаций, и всё это ещё до умножения на размер микробатча, до добавления промежуточных значений attention-softmax (`L^2` на голову) и до учёта частичных копий при тензорном параллелизме.

Счёт приходит с двух сторон: веса в BF16 плюс состояние оптимизатора, может, и влезут в 80 ГБ, но активации выталкивают вас за предел. Gradient checkpointing (он же пересчёт активаций) — стандартное лекарство. Выбрасываем большинство активаций; во время backward прогоняем forward заново, чтобы их вернуть. Цена: лишние FLOPs. Выгода: память падает пропорционально отношению числа сегментов-чекпойнтов к числу слоёв.

Сделанный в лоб, checkpointing обходится примерно в 33 % лишних FLOPs на forward за шаг. Сделанный хорошо — селективно, по «умному отбору» из работы Korthikanti et al. — он экономит память в 5 раз при накладных расходах меньше 5 % FLOPs. А с матричными умножениями в FP8, выгрузкой FSDP и MoE с экспертным параллелизмом это по-настоящему важно: у вас нет лишней ни памяти, ни впустую потраченных вычислений.

> 🎒 **На пальцах.** Разница между 33 % и 5 % накладных расходов — это разница между «обучение стало дольше на треть» и «почти не заметили». На прогоне, который идёт месяц, треть — это лишние десять дней аренды кластера. Вся суть урока в том, чтобы пересчитывать не всё подряд, а только то, что дорого хранить и дёшево посчитать заново.

## The Concept

### What Backward Actually Needs

`output = layer(input)`. На backward нужны `grad_input` и `grad_params`. Чтобы их посчитать, требуется:

- `input` (чтобы посчитать `grad_params = input.T @ grad_output` для линейных слоёв)
- какие-то промежуточные значения для производных активаций (производная ReLU/GELU/softmax зависит от самого значения активации)

Forward pass складывает всё это в граф автоградиента автоматически. Каждый `tensor.retain_grad()` и каждая операция, которой нужен её вход, держат на него ссылку.

> 🎒 **На пальцах.** Обратите внимание на первый пункт: чтобы получить градиент весов линейного слоя, нужен именно его вход `input`, а не выход. Поэтому достаточно сохранить вход в начало сегмента — всё остальное внутри сегмента восстанавливается прогоном forward. Второй пункт про ReLU: чтобы понять, пропускать ли градиент, нужно знать, был ли вход положительным, — это и есть «промежуточное значение».

### Naive Full Checkpointing

Разбиваем сеть на `N` сегментов. На forward сохраняем только *вход* каждого сегмента. Когда backward потребуются промежуточные значения, заново прогоняем forward этого сегмента, чтобы их материализовать, а потом дифференцируем.

Пример: трансформер из 32 слоёв, разбитый на 32 сегмента по одному слою.

- Память: 32 входа слоёв (мало) против 32 * (объём активаций на слой) (много).
- Лишние вычисления: один дополнительный forward на сегмент, то есть примерно на 33 % больше FLOPs на forward суммарно (поскольку backward вдвое дороже forward, полный шаг становится 1 + 1 + 2 = 4 единицы вместо 1 + 2 = 3).

Это исходный рецепт Chen et al. 2016: один чекпойнт каждые `sqrt(L)` слоёв, чтобы уравновесить память и вычисления. Для L=64 это 8 чекпойнтов.

> 🎒 **На пальцах.** Считайте в «единицах forward»: обычный шаг это 1 (forward) + 2 (backward) = 3. С checkpointing добавляется ещё 1 на пересчёт, итого 4. Отсюда и 33 %: `4 / 3 - 1`. Обратите внимание, что 33 % — это добавка ко всему шагу, а не только к forward, хотя лишним прогоном является ровно один forward.

### Selective Checkpointing (Korthikanti 2022)

Не все активации стоят одинаково. Выход attention-softmax — это `B*L*L*heads`, и он растёт *квадратично* по длине последовательности. Скрытая активация FFN — это `B*L*4d`, она растёт линейно. На длинных последовательностях softmax начинает доминировать.

Селективный checkpointing сохраняет дешёвые в хранении активации (линейные проекции, остаточные связи) и пересчитывает только дорогие (внимание). Вы платите минимум FLOPs за пересчёт, зато экономите память порядка O(L^2).

Megatron-Core реализует это как «selective» режим пересчёта активаций. Применяется в большинстве фронтирных прогонов обучения с 2024 года.

> 🎒 **На пальцах.** Подставьте числа: при `L = 8192` величина `L^2` — это 67 миллионов, умноженных на число голов и батч, а `4d` при `d = 8192` — всего 32 тысячи. Разница на три с лишним порядка. Поэтому выбросить один softmax выгоднее, чем выбросить десяток линейных слоёв, — и накладные расходы падают с 33 % до 5 %.

### Offload

Альтернатива пересчёту: перебрасывать активации в оперативную память CPU между forward и backward. Требует пропускной способности PCIe; выгодно, когда свободная полоса дешевле, чем повторное материализование. Смешанные стратегии — обычное дело: часть слоёв чекпойнтим, часть выгружаем.

FSDP2 предлагает выгрузку как штатную опцию. Выгрузка выигрывает, когда GPU упирается в память, а у канала CPU-GPU есть запас.

> 🎒 **На пальцах.** Простое правило: если 800 МБ активаций слоя перекачиваются по PCIe быстрее, чем считается заново лишний forward этого слоя, — выгружайте. На PCIe 5.0 (около 64 ГБ/с) это примерно 12 мс, и на большой модели пересчёт часто оказывается дороже. Но если по шине уже гоняются градиенты FSDP, свободной полосы просто нет, и остаётся пересчёт.

### Recompute Cost Model

FLOPs на шаг при наивном checkpointing каждые `k` слоёв из `L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

При селективном checkpointing вы пересчитываете только ядро внимания, а не весь слой:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

> 🎒 **На пальцах.** Прочитайте формулы как арифметику: наивный вариант добавляет целую единицу пересчёта к трём, селективный — только 0.15 единицы, потому что внимание составляет примерно 15 % вычислений слоя. Отсюда `3.15 / 3 = 1.05`, то есть 5 %. Всё остальное в уроке — это способы честно оценить эти 0.15 для вашей конкретной модели.

### Memory Savings Model

Объём активаций на слой: `A`. Для `L` слоёв суммарная память под активации: `L * A`.

Полный чекпойнт (размер сегмента 1): храним только `L * input_volume` (примерно `L * 1/10 A` для стандартного трансформера). Экономия примерно `9 * L * A * 1/10`.

Чекпойнт каждые `k` слоёв: храним `L/k * A` плюс активации `k-1` слоёв внутри активного сегмента.

При `k = sqrt(L)` и память, и стоимость пересчёта масштабируются как `sqrt(L)` — оптимальный компромисс для слоёв одинаковой стоимости.

> 🎒 **На пальцах.** Формула `k = sqrt(L)` возникает потому, что два слагаемых тянут в разные стороны: чекпойнтов тем больше, чем меньше `k` (это `L/k`), а внутри активного сегмента одновременно живут `k` слоёв. Сумма `L/k + k` минимальна ровно при `k = sqrt(L)`. Для 64 слоёв это `k = 8`, и в памяти вместо 64 наборов активаций живут примерно 16.

### When Not to Checkpoint

- Внутренние слои конвейерной стадии, которая уже в полёте. Им всё равно надо досчитать.
- Первый и последний слои, если они доминируют по вычислениям в стадии (в трансформерах редкость).
- Ядра внимания, которые уже используют FlashAttention — Flash и так быстро пересчитывает softmax, так что дополнительный checkpointing на уровне слоя добавляет мало.

> 🎒 **На пальцах.** Третий пункт стоит запомнить отдельно: FlashAttention уже не хранит матрицу внимания `L×L`, он пересчитывает её блоками прямо на backward. То есть самая дорогая активация из раздела выше у вас уже выброшена. Накладывать сверху селективный checkpointing внимания — значит платить за пересчёт дважды.

### Implementation Patterns

1. **Function wrapper:** оборачиваем сегмент в `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch сохраняет только `input`, всё остальное пересчитывает на backward.

2. **Decorator-based:** помечаем слои как пригодные для чекпойнта; тренер на этапе конфигурации решает, какие сегменты обернуть.

3. **Manual explicit recompute:** пишем backward сами, вызывая свой `recompute_forward`, который повторяет forward из сохранённого входа.

Все три дают одинаковый функциональный результат. Обёртки — стандартная идиома.

### Interaction with TP / PP / FP8

- **Tensor parallel:** входы чекпойнтов при пересчёте надо заново собирать или разбрасывать по рангам; учитывайте стоимость коммуникаций.
- **Pipeline parallel:** типичный приём — чекпойнтить forward каждой конвейерной стадии, чтобы микробатчи в обратном порядке могли переиспользовать память под активации.
- **FP8 recompute:** истории amax, обновляемые при пересчёте, должны совпадать с исходным forward, иначе масштаб FP8 уплывёт. Большинство фреймворков делают снимок масштаба.

> 🎒 **На пальцах.** Третий пункт — классический источник тихих багов. В FP8 масштаб подбирается по максимуму абсолютных значений (amax), который копится по ходу обучения. Если пересчёт обновит эту статистику второй раз, масштаб съедет, и лосс начнёт медленно разъезжаться без единого сообщения об ошибке.

```figure
activation-recompute
```

## Build It

### Step 1: A Toy Model With Segments

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Step 2: Naive Backward Needing All Activations

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

> 🎒 **На пальцах.** Присмотритесь к строке `x_in = activations[i]` — вот она, цена наивного подхода: чтобы посчитать градиенты слоя `i`, нужен сохранённый вход этого слоя, и так для всех слоёв сразу. Заодно заметьте `h_pre = linear_forward(x_in, w1, b1)` — этот кусок forward пересчитывается даже здесь, потому что хранить ещё и `h_pre` было бы совсем расточительно.

### Step 3: Checkpoint-Every-k Memory

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

> 🎒 **На пальцах.** Ключевая строка — `if (i + 1) % k == 0`: при `k=4` и 32 слоях в `saved_inputs` окажется 9 тензоров вместо 33. На backward функция `model_forward(x_in, params[start:end])` заново прогоняет четыре слоя сегмента, чтобы восстановить их активации, и только потом дифференцирует. Градиенты выходят те же самые до последнего бита — это и проверяет первое задание.

### Step 4: Cost Model

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

> 🎒 **На пальцах.** Прогоните обе функции для 32 слоёв. `checkpoint_cost(32, 4)` даёт fwd=32, recompute=32, bwd=64, итого 128 против 96 без чекпойнта — накладные расходы `128/96 - 1 = 0.33`. `selective_checkpoint_cost(32)` даёт recompute=4.8, итого 100.8 и `100.8/96 - 1 = 0.05`. Те самые 33 % и 5 % из теории, только теперь вы их посчитали.

### Step 5: Memory Estimator

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Step 6: Optimal Segment Size

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

> 🎒 **На пальцах.** Подставьте 64 слоя. `activation_memory_mb(64)` даст `64 * 12 * 8192 * 8192 * 2 / 1e6` — примерно 103 000 МБ, то есть 103 ГБ, что не влезет ни в одну карту. `optimal_segment(64)` вернёт 8, и `memory_after_checkpoint(64, 8)` даст около 2100 МБ. Сокращение почти в 50 раз ценой одного лишнего forward.

### Step 7: Selective Checkpoint Decision

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

> 🎒 **На пальцах.** Разберите пороги. Внимание пересчитывается всегда, если его активации весят больше 100 МБ, — потому что растут они квадратично и завтра будут весить гигабайт. FFN пересчитывается только при активациях больше 500 МБ и при условии `recompute_flops_ratio < 0.1`, то есть если пересчёт стоит меньше десятой части слоя. Всё остальное функция оставляет в памяти.

## Use It

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — каноническая обёртка в PyTorch. Оборачивает функцию, хранит только входы, пересчитывает на backward.
- **Megatron-Core activation recomputation**: поддерживает режимы `selective`, `full` и `block`. Стандарт во фронтирном обучении с 2024 года.
- **FSDP2 offload**: `module.to_empty(device="cpu")` вместе с `offload_policy` в FSDP2 шардирует активации в память CPU вместо пересчёта.
- **DeepSpeed ZeRO-Offload**: выгрузка на CPU состояний оптимизатора и активаций, дополняет checkpointing.

> 🎒 **На пальцах.** В реальной работе вы почти никогда не пишете пересчёт руками — вы выбираете режим строкой в конфиге. Начните с `full` в Megatron: он всегда влезает в память, просто стоит 33 %. Потом переключитесь на `selective` и померьте время шага: если оно упало, а память ещё держится, вы нашли нужный режим.

## Ship It

Этот урок производит `outputs/prompt-activation-recompute-policy.md` — промпт, который принимает конфигурацию вашей модели (слои, скрытый размер, длина последовательности, батч) и доступную память GPU, а выдаёт политику пересчёта по слоям (none / selective / full / offload).

## Exercises

1. Проверьте корректность. Прогоните `model_forward` + `model_backward` (все активации) против `model_forward_checkpointed` + `model_backward_checkpointed` (сегменты). Градиенты параметров должны совпасть с точностью до машинного нуля.

2. Пройдитесь по размеру сегмента `k` от 1 до `L`. Постройте графики накладных расходов FLOPs и памяти. Найдите излом кривой.

3. Реализуйте селективный checkpointing: сохраняйте вход модуля внимания, но не его промежуточные значения. Измерьте накладные расходы FLOPs по сравнению с чекпойнтом целого слоя для модели из 32 слоёв при seq=8192.

4. Добавьте выгрузку. Сохраняйте входы сегментов в симулированный «буфер CPU» (отдельный список). Измерьте «пропускную способность PCIe» как байты/время и найдите точку безубыточности между выгрузкой и пересчётом.

5. Прогоните настоящий трансформер на PyTorch с `torch.utils.checkpoint` и без него. Измерьте память (через `torch.cuda.max_memory_allocated`) и время шага.

> 🎒 **На пальцах.** Задание 1 важнее, чем кажется: checkpointing не должен менять численный результат вообще, и если градиенты разошлись — вы где-то сохранили не тот вход или перепутали границы сегмента. Задание 2 даст характерную картинку: память резко падает при `k` от 1 до 8, а дальше почти не меняется, тогда как FLOPs растут ровно. Излом и будет около `sqrt(L)`.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | "Save memory by redoing forward" | Хранить только входы сегментов, а промежуточные значения пересчитывать на backward, чтобы получить тензоры для градиентов |
| Activation recomputation | "Same as checkpointing" | Тот же приём, только названный на манер HPC |
| Segment size (k) | "How many layers per checkpoint" | Число слоёв, чьи промежуточные значения выбрасываются и восстанавливаются вместе |
| Selective checkpointing | "Korthikanti's trick" | Пересчитывать только дорогие в хранении активации (softmax внимания), дешёвые оставлять |
| Full checkpointing | "The naive version" | Пересчитывать промежуточные значения каждого слоя в каждом сегменте |
| Block checkpointing | "Coarse-grained" | Чекпойнтить целые блоки трансформера; самая крупная гранулярность |
| FLOP overhead | "The compute tax" | Лишние FLOPs на шаг = (FLOPs пересчёта) / (FLOPs forward + backward); 33 % наивно, 5 % селективно |
| Activation offload | "Ship to CPU" | Перенос активаций в память CPU между forward и backward; альтернатива пересчёту |
| sqrt-L rule | "The classical optimum" | Для слоёв одинаковой стоимости оптимальный шаг чекпойнтов — sqrt(L) слоёв |
| Attention-softmax volume | "The O(L^2) problem" | L^2 * heads * batch чисел; на длинных контекстах доминирует в памяти под активации |

## Further Reading

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- исходная работа, формализовавшая gradient checkpointing
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- селективный пересчёт активаций и формальный анализ стоимости
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- альтернативный подход с константной памятью через рематериализацию в обратном режиме
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- выгрузка активаций на больших масштабах
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- стандартный API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- режимы selective, full и block

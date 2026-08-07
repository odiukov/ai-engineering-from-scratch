"""
Instruction tuning (SFT) — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Служебные id лежат ВЫШЕ байтового диапазона 0..255. Урок вместо этого
# зажимает байты в 0..252 через min(t, vocab_size - 4) — так три старших
# байта склеиваются в один, и текст перестаёт восстанавливаться. Раздвинуть
# словарь дешевле, чем чинить испорченные данные.
SPECIAL_TOKENS = {
    "INST_START": 256,
    "INST_END": 257,
    "RESP_START": 258,
    "EOS": 259,
}


def tokenize_instruction_pair(instruction, response):
    """Собирает пару «инструкция — ответ» в плоский список id.

    Раскладка: INST_START, байты инструкции, INST_END, RESP_START,
    байты ответа, EOS.

    tokenize_instruction_pair("a", "b")  ->  [256, 97, 257, 258, 98, 259]

    EOS в конце обязателен: без него модель не научится замолкать и будет
    генерировать до упора в лимит токенов.
    """
    return (
        [SPECIAL_TOKENS["INST_START"]]
        + list(instruction.encode("utf-8"))
        + [SPECIAL_TOKENS["INST_END"], SPECIAL_TOKENS["RESP_START"]]
        + list(response.encode("utf-8"))
        + [SPECIAL_TOKENS["EOS"]]
    )


def create_loss_mask(tokens):
    """Маска потерь: 1.0 на токенах ОТВЕТА, 0.0 на всём остальном.

    create_loss_mask([256, 97, 257, 258, 98, 259])  ->  [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    Сам RESP_START получает 0.0: он разделитель, а не часть ответа. А вот
    EOS — часть, его модель обязана научиться ставить.

    Это самая важная техническая деталь SFT, и её чаще всего пропускают.
    Считая потерю на инструкции, ты учишь модель ЗАДАВАТЬ вопросы вместо
    того, чтобы на них отвечать: градиент уходит в предсказание чужих
    реплик.

    В многоходовом диалоге маска включается на каждом RESP_START и
    выключается на следующем INST_START.
    """
    mask = []
    in_response = False
    for token in tokens:
        if token == SPECIAL_TOKENS["RESP_START"]:
            in_response = True
            mask.append(0.0)
        elif token == SPECIAL_TOKENS["INST_START"]:
            in_response = False
            mask.append(0.0)
        else:
            mask.append(1.0 if in_response else 0.0)
    return mask


def tokenize_conversation(turns):
    """Многоходовой диалог в один плоский список id.

    turns — список словарей {"role": "user"|"assistant", "content": ...}.

    tokenize_conversation([{"role": "user", "content": "a"},
                           {"role": "assistant", "content": "b"}])
        ->  [256, 97, 257, 258, 98, 259]

    Реплика пользователя обёрнута в INST_START/INST_END, реплика
    ассистента — в RESP_START/EOS. Обобщение tokenize_instruction_pair:
    на одном обмене репликами результаты обязаны совпасть.

    Все ответы ассистента идут в потерю, а не только последний — иначе три
    четверти размеченного диалога пропадут впустую.
    """
    tokens = []
    for turn in turns:
        content = list(turn["content"].encode("utf-8"))
        if turn["role"] == "assistant":
            tokens += [SPECIAL_TOKENS["RESP_START"]] + content + [SPECIAL_TOKENS["EOS"]]
        else:
            tokens += [SPECIAL_TOKENS["INST_START"]] + content + [SPECIAL_TOKENS["INST_END"]]
    return tokens


def shift_for_training(tokens, mask):
    """Сдвиг на одну позицию: вернуть (inputs, targets, target_mask).

    Модель по позиции i предсказывает токен i+1, поэтому входы — это всё
    кроме последнего токена, а цели — всё кроме первого.

    shift_for_training([1, 2, 3], [0.0, 1.0, 1.0])  ->  ([1, 2], [2, 3], [1.0, 1.0])

    Ловушка: маска обязана поехать ВМЕСТЕ С ЦЕЛЯМИ, а не со входами. Маска
    отвечает на вопрос «нужно ли считать потерю на предсказании этого
    токена», а предсказываем мы цели. Сдвинешь не то — потеря съедет на
    один токен, и модель начнёт учить разделители вместо текста.
    """
    return tokens[:-1], tokens[1:], mask[1:]


def masked_cross_entropy(logits_rows, targets, mask):
    """Cross-entropy только по позициям с маской 1. Возвращает число.

    masked_cross_entropy([[0.0, 0.0], [0.0, 0.0]], [0, 1], [0.0, 1.0])
        ->  0.6931...  (это ln 2: вторая позиция, первая не в счёт)

    Делим на СУММУ МАСКИ, а не на длину последовательности. Иначе длинная
    инструкция разбавляет градиент: один и тот же ответ после короткого и
    после длинного вопроса даст разную потерю, хотя учим мы ровно одному
    и тому же.

    Если маска целиком нулевая — возвращаем 0.0, а не делим на ноль.

    Логиты считаем через log-sum-exp с вычитанием максимума: exp от
    сырого логита переполняется.
    """
    denominator = sum(mask)
    if denominator == 0:
        return 0.0
    total = 0.0
    for row, target, m in zip(logits_rows, targets, mask):
        if m == 0:
            continue  # позиция не в счёт, считать её незачем
        top = max(row)
        log_sum = top + math.log(sum(math.exp(s - top) for s in row))
        total += m * (log_sum - row[target])
    return total / denominator


def d_masked_cross_entropy(logits_rows, targets, mask):
    """Градиент masked_cross_entropy по логитам. Матрица того же размера.

    Строка с маской 0.0 состоит из нулей: замаскированная позиция не
    двигает ни один вес.

    d_masked_cross_entropy([[0.0, 0.0], [0.0, 0.0]], [0, 1], [0.0, 1.0])
        ->  [[0.0, 0.0], [0.5, -0.5]]

    Формула на живой позиции та же, что в предобучении: (softmax -
    one_hot), делённая на сумму маски. Проверь себя численной производной —
    сходящаяся формула и правильная формула это разные вещи.
    """
    denominator = sum(mask)
    grad = []
    for row, target, m in zip(logits_rows, targets, mask):
        if denominator == 0 or m == 0:
            grad.append([0.0] * len(row))
            continue
        top = max(row)
        exps = [math.exp(s - top) for s in row]
        total = sum(exps)
        probs = [e / total for e in exps]
        probs[target] -= 1.0
        grad.append([m * p / denominator for p in probs])
    return grad


def dataset_quality(example):
    """Метрики одного примера датасета. Возвращает словарь.

    Ключи: "instruction_tokens", "response_tokens", "response_ratio"
    (доля ответа в длине примера), "diversity" (уникальные токены ответа,
    делённые на все токены ответа).

    dataset_quality({"instruction": "ab", "response": "cd"})["diversity"]  ->  1.0
    dataset_quality({"instruction": "ab", "response": "aaaa"})["diversity"]  ->  0.25

    Что этим ловят на практике: ответы-обрубки («Yes.»), ответы-повторы
    («да да да да») и примеры, где инструкция длиннее ответа в десять раз.
    Типовой фильтр — выбросить всё с response_tokens < 10 или
    diversity < 0.3. LIMA показала, что тысяча вычищенных примеров бьёт
    пятьдесят тысяч сырых.

    Пустой ответ: делить не на что, отдаём нули.
    """
    inst = list(example["instruction"].encode("utf-8"))
    resp = list(example["response"].encode("utf-8"))
    if not resp:
        return {
            "instruction_tokens": len(inst),
            "response_tokens": 0,
            "response_ratio": 0.0,
            "diversity": 0.0,
        }
    return {
        "instruction_tokens": len(inst),
        "response_tokens": len(resp),
        "response_ratio": len(resp) / (len(inst) + len(resp)),
        "diversity": len(set(resp)) / len(resp),
    }


def mix_pretraining_data(sft_examples, raw_texts, fraction, rng):
    """Подмешивает сырой текст в SFT-датасет. Возвращает список (tokens, mask).

    Размер результата равен размеру sft_examples: доля fraction примеров
    ЗАМЕНЯЕТСЯ сырым текстом, остальные остаются инструкциями.

    У сырого текста маска целиком единичная — это обычное предобучение,
    маскировать там нечего.

    len(mix_pretraining_data(sft, raw, 0.25, rng)) == len(sft)

    rng — экземпляр random.Random. Глобальный random здесь брать нельзя:
    эксперимент «с подмешиванием против без» обязан быть воспроизводимым,
    иначе разницу метрик не с чем сравнивать.

    Зачем это нужно: Llama 2 Chat добавляла 2-5% предобучающих данных,
    чтобы модель не забыла всё, кроме формата диалога. Это самое дешёвое
    средство против катастрофического забывания — дешевле, чем подбирать
    learning rate.
    """
    total = len(sft_examples)
    n_raw = round(fraction * total)
    kept = list(sft_examples)
    rng.shuffle(kept)

    mixed = []
    for example in kept[: total - n_raw]:
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mixed.append((tokens, create_loss_mask(tokens)))
    for _ in range(n_raw):
        tokens = list(raw_texts[rng.randrange(len(raw_texts))].encode("utf-8"))
        mixed.append((tokens, [1.0] * len(tokens)))
    rng.shuffle(mixed)
    return mixed

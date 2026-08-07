"""
Transfer learning и дообучение — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Модель здесь — обычный список словарей, по одному на тензор параметров:

    {"name": "layer1.0.conv1.weight", "kind": "conv",
     "size": 36864, "value": 1.0, "trainable": True}

Это ровно то, что отдаёт model.named_parameters() в PyTorch, только вместо
тензора одно число value, чтобы шаг оптимизатора можно было проверить руками.
"""


def pick_recipe(num_images, domain="close"):
    """Выбор режима переноса по размеру датасета и близости домена.

    Возвращает одно из: "freeze_backbone", "freeze_early", "finetune_all",
    "train_from_scratch". domain — "close" (обычные фото) или "far"
    (КТ-снимки, спутник, микроскопия).

    pick_recipe(500)                    ->  "freeze_backbone"
    pick_recipe(5000)                   ->  "freeze_early"
    pick_recipe(5000, "far")            ->  "finetune_all"
    pick_recipe(500_000, "far")         ->  "train_from_scratch"

    Границы из таблицы урока, включающие: до 1k, 1k-10k, 10k-100k, 100k+.
    Ровно 1000 — это уже вторая строка, не первая.

    Смысл правил один: чем больше своих данных и чем дальше домен, тем
    больше слоёв разрешено двигать. На 500 картинках размороженный backbone
    не выучит ничего нового, зато успеет забыть ImageNet.
    """
    if num_images < 1_000:
        return "freeze_backbone"
    if num_images < 10_000:
        return "freeze_early" if domain == "close" else "finetune_all"
    if num_images < 100_000:
        return "finetune_all"
    return "train_from_scratch" if domain == "far" else "finetune_all"


def freeze_backbone(params, head_prefix="fc"):
    """Feature extraction: заморозить всё, кроме головы. Новый список.

    freeze_backbone([{"name": "fc.weight", ...}, {"name": "layer4...", ...}])
        ->  голова trainable=True, всё остальное trainable=False

    Аналог двух строк из урока: цикла p.requires_grad = False по всем
    параметрам и последующей замены model.fc.

    Исходный список не менять — обучение линейного пробника и полный
    fine-tune часто считают на одной и той же модели по очереди, и молча
    испорченный вход даст цифры, которые невозможно объяснить.
    """
    # копия каждой записи: dict(p) достаточно, вложенных структур тут нет
    return [
        {**p, "trainable": p["name"].startswith(head_prefix)} for p in params
    ]


def freeze_bn_stats(params):
    """Заморозить BatchNorm: trainable=False и eval_mode=True. Новый список.

    Записи с kind != "bn" не трогаются вообще, включая их флаг trainable.

    freeze_bn_stats([{"name": "bn1.weight", "kind": "bn", ...}])
        ->  [{... "trainable": False, "eval_mode": True}]

    Зачем: running_mean и running_var в BN посчитаны на ImageNet. На
    маленьком датасете скользящее среднее по нескольким батчам получается
    шумным и портит признаки, которые в остальном идеальны. Тихая потеря
    5-15% точности, которую не видно ни в одной строке лога.

    Вызывается ПОСЛЕ model.train(): train() переводит в режим обучения всё,
    а эта функция откатывает только BN.
    """
    out = []
    for p in params:
        if p.get("kind") == "bn":
            out.append({**p, "trainable": False, "eval_mode": True})
        else:
            out.append({**p})
    return out


def trainable_summary(params):
    """Сколько весов обучается, а сколько заморожено: {"trainable", "frozen"}.

    trainable_summary([{"size": 100, "trainable": True},
                       {"size": 900, "trainable": False}])
        ->  {"trainable": 100, "frozen": 900}

    Считаются size, а не количество тензоров: тензоров у головы 2, а весов
    в ней тысячи, и решает именно вторая цифра.

    Первое, что печатают после заморозки. Если trainable вышло размером с
    всю сеть — заморозка не сработала, а тренировка всё равно запустится и
    отработает вроде бы нормально.
    """
    trainable = sum(p["size"] for p in params if p["trainable"])
    frozen = sum(p["size"] for p in params if not p["trainable"])
    return {"trainable": trainable, "frozen": frozen}


def stage_lrs(stages, base_lr=1e-3, decay=0.3):
    """Discriminative learning rates: своя скорость каждой стадии.

    stages — список списков префиксов, от ранних слоёв к поздним, например
    [["conv1", "bn1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"], ["fc"]].
    Возвращает список пар (имя стадии, lr), имя — префиксы через "_".

    stage_lrs([["layer4"], ["fc"]], base_lr=1e-3, decay=0.1)
        ->  [("layer4", 0.0001), ("fc", 0.001)]

    Формула lr_i = base_lr * decay^(L-1-i): ПОСЛЕДНЯЯ стадия получает ровно
    base_lr, каждая предыдущая — в decay раз меньше. Перепутать направление
    легко, а последствия громкие: голову учить в 300 раз медленнее stem,
    и модель не сдвинется вообще.

    Та же формула под именем layer-wise LR decay применяется поблочно при
    дообучении трансформеров: lr_k = base_lr * decay^(L-k).
    """
    last = len(stages) - 1
    return [
        ("_".join(names), base_lr * decay ** (last - i))
        for i, names in enumerate(stages)
    ]


def param_groups(params, stages, base_lr=1e-3, decay=0.3):
    """Группы параметров для оптимизатора: имя, lr и список имён параметров.

    Берутся только обучаемые параметры (trainable=True). Параметр попадает в
    стадию, если его имя начинается с одного из её префиксов. Пустые группы
    в результат не попадают.

    param_groups(params, [["layer4"], ["fc"]])[0]
        ->  {"name": "layer4", "lr": 0.0003, "params": ["layer4.0.conv1.weight"]}

    Аналог списка dict-ов, который PyTorch принимает как первый аргумент
    оптимизатора: SGD(groups, momentum=0.9).

    Замороженный параметр в группе — не безобидная мелочь: weight decay и
    momentum продолжают его двигать даже при нулевом градиенте. Отсюда
    правило «пересобирай оптимизатор каждый раз, когда меняется набор
    обучаемых параметров».

    lr бери из stage_lrs, повторно выводить формулу незачем.
    """
    groups = []
    for (name, lr), prefixes in zip(stage_lrs(stages, base_lr, decay), stages):
        picked = [
            p["name"]
            for p in params
            if p["trainable"] and any(p["name"].startswith(k) for k in prefixes)
        ]
        if picked:  # пустую группу оптимизатору отдавать нельзя
            groups.append({"name": name, "lr": lr, "params": picked})
    return groups


def progressive_unfreeze(params, schedule, epoch, head_prefix="fc"):
    """Постепенная разморозка: голова плюс первые epoch+1 стадий расписания.

    schedule — стадии в порядке разморозки, обычно от конца сети к началу:
    ["layer4", "layer3", "layer2", "layer1"].

    progressive_unfreeze(params, ["layer4", "layer3"], 0)
        ->  обучаются fc и layer4
    progressive_unfreeze(params, ["layer4", "layer3"], 99)
        ->  обучаются fc, layer4 и layer3; дальше расписания не идём

    Голова обучается всегда, начиная с нулевой эпохи — иначе первые
    градиенты пойдут в backbone от случайно инициализированного классификатора
    и снесут именно то, ради чего его брали (catastrophic forgetting).

    Возвращай новый список, вход не трогай.
    """
    active = set(schedule[: epoch + 1])
    out = []
    for p in params:
        trainable = p["name"].startswith(head_prefix) or any(
            p["name"].startswith(s) for s in active
        )
        out.append({**p, "trainable": trainable})
    return out


def sgd_step(params, grads, lrs):
    """Шаг SGD: value -= lr * grad. Новый список параметров.

    grads — словарь имя -> градиент, lrs — словарь имя -> learning rate.
    Замороженный параметр не двигается. Параметр, которого нет в lrs, тоже
    не двигается: он не попал ни в одну группу оптимизатора.

    sgd_step([{"name": "fc.w", "value": 1.0, "trainable": True, "size": 1}],
             {"fc.w": 2.0}, {"fc.w": 0.1})
        ->  value стало 0.8

    Вторая ветка — не выдумка, а самый неприятный баг дообучения: слой
    разморозили, а оптимизатор собрали до этого и о новом параметре он не
    знает. Ошибки нет, лосс падает, метрика стоит на месте.

    lrs удобно собрать из param_groups: {имя: g["lr"] for g in groups ...}.
    """
    out = []
    for p in params:
        if p["trainable"] and p["name"] in lrs:
            step = lrs[p["name"]] * grads.get(p["name"], 0.0)
            out.append({**p, "value": p["value"] - step})
        else:
            out.append({**p})
    return out

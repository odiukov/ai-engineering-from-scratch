"""
Разбор архитектур открытых моделей

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p10-l14-open-models-architecture-walkthroughs
Разбор:  /check-code p10-l14-open-models-architecture-walkthroughs
"""

import math


def rms_norm(x, gamma, eps=1e-5):
    """RMSNorm: делим на корень из среднего квадрата, умножаем на gamma.

    Модуль модели: LlamaRMSNorm.

    rms_norm([3.0, 4.0], [1.0, 1.0], 0.0)  ->  [0.848..., 1.131...]
    rms_norm([1.0, 1.0], [2.0, 2.0], 0.0)  ->  [2.0, 2.0]

    Формула: x_i / sqrt(mean(x^2) + eps) * gamma_i.

    Главное отличие от LayerNorm: среднее НЕ вычитается и сдвига (beta)
    нет. Поэтому rms_norm не центрирует вход — постоянная добавка ко всем
    координатам меняет результат, а у LayerNorm нет.
    """
    raise NotImplementedError


def rope_rotate(vec, pos, theta=10000.0):
    """RoPE: поворот пар координат на угол, зависящий от позиции.

    Модуль модели: apply_rotary_pos_emb.

    rope_rotate([1.0, 0.0], 0)  ->  [1.0, 0.0]   (нулевая позиция — тождество)
    rope_rotate([1.0, 0.0], 1, theta=1.0)  ->  [cos(1), sin(1)]

    Координаты берутся парами (0,1), (2,3), ...: пара номер i крутится на
    угол pos * theta^(-2i/d). Формула поворота:
        x' = x*cos - y*sin
        y' = x*sin + y*cos

    Нечётная длина вектора -> ValueError: крутить нечего.

    Ради чего это: скалярное произведение повёрнутых q и k зависит только
    от РАЗНОСТИ позиций. Отсюда и экстраполяция за длину обучения — в
    отличие от таблицы обучаемых позиций GPT-2, которая просто кончается.
    """
    raise NotImplementedError


def softmax(logits):
    """Логиты -> распределение вероятностей.

    softmax([0.0, 0.0])     ->  [0.5, 0.5]
    softmax([0.0, 1000.0])  ->  [0.0, 1.0]   (без OverflowError!)

    Ловушка: math.exp(1000) падает с OverflowError. Вычти максимум перед
    экспонентой — результат математически тот же, переполнения нет.
    """
    raise NotImplementedError


def swiglu_mlp(x, W_gate, W_up, W_down):
    """MLP-блок современной модели: SwiGLU вместо GELU.

    Модуль модели: LlamaMLP.

    Формула: down( silu(gate(x)) * up(x) ), где silu(z) = z * sigmoid(z).
    Три матрицы вместо двух: gate и up идут параллельно, их произведение
    поэлементное, потом down возвращает в hidden.

    swiglu_mlp([1.0], [[0.0]], [[1.0]], [[1.0]])  ->  [0.0]
        (silu(0) = 0, значит вся ветка гасится)

    Ловушка: наивная sigmoid(z) = 1/(1+exp(-z)) падает с OverflowError на
    больших отрицательных z. Разбери знак: при z < 0 считай exp(z)/(1+exp(z)).

    Именно поэтому intermediate_size у Llama 3 равен 14336, а не 4*4096:
    трёх матриц вместо двух, размер урезан до 8/3 * hidden.
    """
    raise NotImplementedError


def top_k_route(logits, k):
    """Роутер MoE: выбрать k экспертов и раздать им веса.

    Модуль модели: MixtralSparseMoeBlock (часть с router).

    Возвращает пару (indices, weights): indices отсортированы по
    возрастанию, weights — softmax ТОЛЬКО по выбранным логитам.

    top_k_route([0.0, 0.0], 2)      ->  ([0, 1], [0.5, 0.5])
    top_k_route([3.0, 0.0, 1.0], 1) ->  ([0], [1.0])

    Ловушка: нормировать надо по выбранным логитам, а не по всем — иначе
    веса не дадут в сумме единицу и выход блока просядет по норме.

    При равных логитах побеждает меньший индекс: маршрутизация обязана
    быть воспроизводимой, иначе два прогона дадут разные выходы.
    """
    raise NotImplementedError


def moe_block(x, experts, router_logits, k):
    """Блок Mixture-of-Experts: взвешенная сумма выходов k экспертов.

    Модуль модели: MixtralSparseMoeBlock целиком.

    experts — список троек (W_gate, W_up, W_down) для swiglu_mlp.
    Невыбранные эксперты НЕ вызываются: в этом весь смысл разреженности,
    671B параметров всего и 37B активных на токен.

    Свойство, которое стоит проверить руками: при k = len(experts) и
    одинаковых логитах ответ равен среднему выходов всех экспертов.
    """
    raise NotImplementedError


def param_count(config):
    """Разбор числа параметров по компонентам прямо из config.json.

    Возвращает словарь с ключами: embedding, attention, mlp, norm, head,
    total. attention/mlp/norm — суммы по ВСЕМ слоям.

    Считаем так:
      head_dim  = hidden_size / num_attention_heads
      attention = Wq(hidden*heads*head_dim) + Wk и Wv(hidden*kv_heads*head_dim)
                  + Wo(heads*head_dim*hidden)   — вот где GQA экономит
      mlp       = 3 * hidden * intermediate     — три матрицы SwiGLU
      norm      = 2 * hidden на слой + один финальный hidden
      head      = vocab * hidden, ноль при tie_word_embeddings

    На конфиге Llama 3 8B total выходит 8_030_261_248 — те самые «8B».

    Смещений (bias) в современных моделях нет, поэтому их и не считаем.
    """
    raise NotImplementedError


def kv_cache_bytes(config, seq_len, bytes_per_elem=2):
    """Размер KV-кэша одной последовательности в байтах.

    Формула урока:
      2 * num_hidden_layers * num_key_value_heads * head_dim * seq_len * bytes

    Двойка — это K и V. bytes_per_elem = 2 для BF16, 1 для FP8.

    На Llama 3 8B при 131072 токенах и BF16 выходит 17_179_869_184 байт,
    то есть 17.2 ГБ — БОЛЬШЕ, чем сами веса в BF16 (16 ГБ). Ровно это и
    гонит всех с MHA на GQA и MLA.
    """
    raise NotImplementedError

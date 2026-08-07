"""
Выбор движка для self-hosted inference

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l28-self-hosted-serving-selection
Разбор:  /check-code p17-l28-self-hosted-serving-selection
"""

from datetime import date

ENGINES = ("Ollama", "SGLang", "TGI", "TRT-LLM", "llama.cpp", "vLLM")
HARDWARE = ("AMD", "Apple Silicon", "CPU", "NVIDIA Blackwell", "NVIDIA Hopper")
HARDWARE_SUPPORT = {
    "llama.cpp": ("CPU", "Apple Silicon"),
    "Ollama": ("CPU", "Apple Silicon", "NVIDIA Hopper"),
    "TGI": ("NVIDIA Hopper", "NVIDIA Blackwell"),
    "vLLM": ("AMD", "NVIDIA Hopper", "NVIDIA Blackwell"),
    "SGLang": ("AMD", "NVIDIA Hopper", "NVIDIA Blackwell"),
    "TRT-LLM": ("NVIDIA Hopper", "NVIDIA Blackwell"),
}
CRITERIA = (
    "throughput",          # пропускная способность под продовой нагрузкой
    "setup_simplicity",    # сколько команд до первого токена
    "ecosystem",           # observability и интеграция с HF
    "prefix_reuse",        # переиспользование префикса (RadixAttention)
    "model_coverage",      # широта поддержки моделей
    "future_proof",        # активность разработки
)
SCORES = {
    "llama.cpp": {"throughput": 2, "setup_simplicity": 3, "ecosystem": 2,
                  "prefix_reuse": 2, "model_coverage": 5, "future_proof": 4},
    "Ollama": {"throughput": 1, "setup_simplicity": 5, "ecosystem": 2,
               "prefix_reuse": 1, "model_coverage": 4, "future_proof": 4},
    "TGI": {"throughput": 4, "setup_simplicity": 3, "ecosystem": 5,
            "prefix_reuse": 2, "model_coverage": 3, "future_proof": 1},
    "vLLM": {"throughput": 5, "setup_simplicity": 3, "ecosystem": 4,
             "prefix_reuse": 3, "model_coverage": 4, "future_proof": 5},
    "SGLang": {"throughput": 4, "setup_simplicity": 2, "ecosystem": 3,
               "prefix_reuse": 5, "model_coverage": 3, "future_proof": 5},
    "TRT-LLM": {"throughput": 5, "setup_simplicity": 1, "ecosystem": 2,
                "prefix_reuse": 2, "model_coverage": 2, "future_proof": 4},
}
WEIGHT_PROFILES = {
    "dev": {"setup_simplicity": 0.6, "model_coverage": 0.2,
            "throughput": 0.1, "future_proof": 0.1},
    "staging": {"throughput": 0.5, "model_coverage": 0.3,
                "setup_simplicity": 0.2},
    "prod_general": {"throughput": 0.4, "future_proof": 0.25,
                     "ecosystem": 0.2, "setup_simplicity": 0.15},
    "prod_agentic": {"prefix_reuse": 0.5, "throughput": 0.3,
                     "future_proof": 0.2},
    "ecosystem_first": {"ecosystem": 0.7, "throughput": 0.3},
}
WEIGHT_FORMATS = {
    "llama.cpp": "GGUF",
    "Ollama": "GGUF",
    "TGI": "safetensors",
    "vLLM": "safetensors",
    "SGLang": "safetensors",
    "TRT-LLM": "safetensors",
}
MAINTENANCE_SINCE = {"TGI": "2025-12-11"}
MAINTENANCE_MULTIPLIER = 0.7


def supports(engine, hardware, support=HARDWARE_SUPPORT):
    """Поедет ли движок на этом железе.

    supports("TRT-LLM", "NVIDIA Hopper")  ->  True
    supports("TRT-LLM", "AMD")            ->  False
    supports("llama.cpp", "CPU")          ->  True

    Незнакомое имя движка или железа — ValueError, а не тихий False. Опечатка
    в «NVIDIA Hoper» не должна выглядеть как «этот движок туда не ставится».
    """
    raise NotImplementedError


def eligible_engines(hardware, support=HARDWARE_SUPPORT):
    """Кто вообще запускается на этом железе. Отсортированный список.

    eligible_engines("CPU")   ->  ['Ollama', 'llama.cpp']
    eligible_engines("AMD")   ->  ['SGLang', 'vLLM']

    Это жёсткий фильтр ПЕРЕД взвешенной оценкой, а не ещё один критерий с
    весом. Иначе достаточно задрать вес throughput — и калькулятор посоветует
    TRT-LLM на AMD, где он физически не стартует.

    Сортировка обычная строковая: заглавные буквы идут раньше строчных,
    поэтому 'llama.cpp' оказывается последним.
    """
    raise NotImplementedError


def normalize_weights(weights):
    """Привести веса к сумме 1.0. Словарь тех же ключей.

    normalize_weights({"throughput": 1, "ecosystem": 1})
        ->  {'throughput': 0.5, 'ecosystem': 0.5}
    normalize_weights({"throughput": 4, "ecosystem": 1})
        ->  {'throughput': 0.8, 'ecosystem': 0.2}

    Нормировка нужна, чтобы сравнивать профили между собой: без неё «вес 10 у
    двух критериев» и «вес 1 у двух критериев» дают разные оценки одному и
    тому же движку, и margin между первым и вторым местом теряет смысл.

    Три ошибки, каждая из которых обязана быть ValueError, а не молча съеденной:
    неизвестный критерий, отрицательный вес и нулевая сумма. «Ни один критерий
    не важен» — это не вопрос, на который бывает ответ.
    """
    raise NotImplementedError


def maintenance_multiplier(engine, now, since=MAINTENANCE_SINCE):
    """Штраф за maintenance mode на дату now. 1.0 или MAINTENANCE_MULTIPLIER.

    maintenance_multiplier("TGI", "2025-12-01")  ->  1.0
    maintenance_multiplier("TGI", "2026-08-07")  ->  0.7
    maintenance_multiplier("vLLM", "2026-08-07") ->  1.0

    Дата приходит параметром now, а не берётся из date.today(): решение
    «каким движком стартовать» принимается на конкретный день, и его надо уметь
    воспроизвести в постмортеме через год.

    Граница включающая: в сам день объявления (11 декабря 2025) штраф уже
    действует — новость вышла, а не «выйдет завтра».
    """
    raise NotImplementedError


def weighted_score(engine, weights, now, table=SCORES, since=MAINTENANCE_SINCE):
    """Взвешенная оценка движка на дату now. Число в диапазоне 0..5.

    weighted_score("vLLM", {"throughput": 1.0}, "2026-08-07")   ->  5.0
    weighted_score("Ollama", {"throughput": 1.0}, "2026-08-07") ->  1.0
    weighted_score("TGI", {"ecosystem": 1.0}, "2026-08-07")     ->  3.5

    Третий пример — весь смысл штрафа: сырая оценка экосистемы у TGI 5.0,
    но на 2026 год она умножается на 0.7.

    Критерии, которых нет в weights, в сумму не входят вовсе. Спрашивать
    «кто лучше по throughput» — законный вопрос, и отвечать на него надо
    только по throughput.
    """
    raise NotImplementedError


def rank_engines(hardware, weights, now, support=HARDWARE_SUPPORT, table=SCORES,
                 since=MAINTENANCE_SINCE):
    """Ранжировать пригодные движки. Список пар (движок, оценка), лучший первым.

    rank_engines("CPU", {"setup_simplicity": 1.0}, "2026-08-07")
        ->  [('Ollama', 5.0), ('llama.cpp', 3.0)]
    rank_engines("CPU", {"model_coverage": 1.0}, "2026-08-07")
        ->  [('llama.cpp', 5.0), ('Ollama', 4.0)]

    Два вызова, одна таблица, разные веса — и порядок перевернулся. Ровно так
    же переворачивается ответ на вопрос «какой движок лучше»: он зависит от
    того, что вы решили считать важным.

    Ничьи разрешаются по имени движка, иначе порядок будет зависеть от порядка
    ключей в словаре и отчёт перестанет воспроизводиться.
    """
    raise NotImplementedError


def pick_engine(hardware, weights, now, support=HARDWARE_SUPPORT, table=SCORES,
                since=MAINTENANCE_SINCE):
    """Свести ранжирование в решение. Словарь с обоснованием.

    Ключи: hardware, engine, score, runner_up, margin.

    pick_engine("AMD", WEIGHT_PROFILES["prod_general"], "2026-08-07")
        ->  engine 'vLLM', runner_up 'SGLang'

    margin — отрыв от второго места. Он важнее самой оценки: отрыв 0.05 значит
    «выбор ни на чём не держится, решайте по другим соображениям», а не
    «победитель определён».

    Если пригодный движок ровно один, runner_up и margin — None. Выдумывать
    отрыв не от кого.
    """
    raise NotImplementedError


def pipeline_plan(stages, now, profiles=WEIGHT_PROFILES, support=HARDWARE_SUPPORT,
                  table=SCORES, since=MAINTENANCE_SINCE):
    """Расписать конвейер dev -> staging -> prod. Список шагов по стадиям.

    stages — последовательность троек (имя стадии, железо, имя профиля весов).
    Ключи шага: stage, hardware, engine, weight_format, conversion.

    pipeline_plan([("dev", "Apple Silicon", "dev"),
                   ("prod", "NVIDIA Hopper", "prod_general")], "2026-08-07")
        ->  dev: Ollama / GGUF / conversion None
            prod: vLLM / safetensors / conversion 'GGUF -> safetensors'

    conversion заполняется, только когда формат весов отличается от формата
    предыдущей стадии. Это и есть та самая конвертация между стадиями: на
    ноутбуке крутится GGUF, в проде — safetensors, и кто-то обязан её сделать.

    Неизвестное имя профиля — ValueError: тихо подставить дефолт значит
    посоветовать движок под чужие приоритеты.
    """
    raise NotImplementedError

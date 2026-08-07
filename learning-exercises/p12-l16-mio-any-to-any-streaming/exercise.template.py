"""
MIO: any-to-any и потоковая генерация

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l16-mio-any-to-any-streaming
Разбор:  /check-code p12-l16-mio-any-to-any-streaming
"""

CURRICULUM = (
    ("alignment", "общий словарь модальностей"),
    ("interleaved", "кросс-модальный контекст"),
    ("speech", "качество речи"),
    ("sft", "следование инструкциям"),
)
TOKENIZERS = {
    "text": "BPE",
    "image": "SEED-Tokenizer",
    "speech": "SpeechTokenizer residual-VQ",
    "voice": "SpeechTokenizer residual-VQ",
    "music": "Encodec",
}


def allocate_vocab(plan):
    """План (имя, размер) -> список слотов (имя, start, end). end не включён.

    allocate_vocab([("text", 3), ("image", 2)])  ->  [("text", 0, 3), ("image", 3, 5)]

    Слоты идут вплотную друг за другом от нуля: между ними не должно быть
    ни дыр (пустая трата embedding-матрицы), ни нахлёста (id одной
    модальности начнёт означать другую, и модель этого даже не заметит).

    Ловушки: нулевой или отрицательный размер слота и повтор имени. И то,
    и другое — ValueError.
    """
    raise NotImplementedError


def modality_of(slots, token_id):
    """Кому принадлежит id: имя слота, в чей диапазон он попал.

    modality_of([("text", 0, 3), ("image", 3, 5)], 0)  ->  'text'
    modality_of([("text", 0, 3), ("image", 3, 5)], 3)  ->  'image'

    Границы половинчатые: start входит, end — нет. Именно поэтому id 3
    принадлежит image, а не text.

    Id вне всех диапазонов — ValueError. Молчаливый None здесь дороже:
    он всплывёт где-нибудь в декодере в виде тишины вместо звука.
    """
    raise NotImplementedError


def route_modality(kind):
    """Вид входа -> имя токенизатора.

    route_modality("image")  ->  'SEED-Tokenizer'
    route_modality("voice")  ->  'SpeechTokenizer residual-VQ'

    Незнакомый вид — ValueError. Fallback на BPE был бы худшим из решений:
    аудио, разобранное как текст, даёт правдоподобный мусор, который заметят
    только в проде.
    """
    raise NotImplementedError


def embedding_params(vocab_size, hidden_dim, tied=True):
    """Сколько параметров стоит словарь: embedding + выходная проекция.

    embedding_params(48394, 4096)               ->  198221824
    embedding_params(48394, 4096, tied=False)   ->  396443648

    tied=True значит, что входная матрица и выходная проекция — одна и та же
    (weight tying). tied=False — две отдельные, и цена ровно удваивается.

    Это упражнение 3 урока: 48k словаря на hidden 4096 стоит почти 200M
    параметров ещё до первого слоя трансформера. Каждая новая модальность
    в общем словаре платится этой матрицей.
    """
    raise NotImplementedError


def residual_vq_tokens(seconds, base_hz, layers):
    """Сколько токенов даёт residual-VQ на отрезок звука.

    residual_vq_tokens(1.0, 20, 8)  ->  160
    residual_vq_tokens(1.0, 20, 1)  ->  20

    base_hz — сколько кадров в секунду выдаёт базовый уровень. У MIO кадр
    это 50 мс, то есть base_hz = 20. Каждый следующий уровень квантователя
    добавляет столько же токенов: layers=8 значит восьмикратный объём.

    Отсюда и требование декодировать уровни 1..7 параллельно: последовательно
    они стоили бы в восемь раз больше проходов, и разговорной латентности
    не получилось бы никогда.

    Дробное число кадров округляем вниз: неполный кадр звука ещё не кадр.
    """
    raise NotImplementedError


def latency_trace(stages):
    """Стадии (имя, мс) -> список (имя, мс, накопленная сумма).

    latency_trace([("mic", 40.0), ("prefill", 80.0)])
        ->  [("mic", 40.0, 40.0), ("prefill", 80.0, 120.0)]

    Накопленная сумма и есть ответ на вопрос «когда пользователь услышит
    первый звук»: TTFAB — это последнее число последней строки.

    Отрицательная стадия — ValueError. Время не возвращается назад, а
    «минус» в такой таблице обычно значит, что кто-то вычел параллельную
    стадию вместо того, чтобы её не добавлять.
    """
    raise NotImplementedError


def latency_verdict(total_ms):
    """Приговор бюджету TTFAB по порогам урока.

    latency_verdict(300.0)  ->  'conversational'
    latency_verdict(600.0)  ->  'acceptable'
    latency_verdict(900.0)  ->  'sluggish'

    Границы: меньше 500 мс — разговор ощущается живым (класс GPT-4o);
    500..800 — терпимо, так работают первые открытые any-to-any;
    800 и больше — пользователь начинает говорить поверх модели.

    Отрицательное время — ValueError.
    """
    raise NotImplementedError


def curriculum_gap(done):
    """Какие способности не появятся, если пройдены только стадии из done.

    curriculum_gap(["alignment", "interleaved", "speech", "sft"])  ->  ()
    curriculum_gap(["alignment", "speech", "sft"])
        ->  ('кросс-модальный контекст',)

    Порядок результата — порядок CURRICULUM, а не порядок done: читать
    отчёт «что сломано» удобнее в порядке обучения.

    Неизвестное имя стадии — ValueError. Опечатка в названии иначе выглядит
    как честно пропущенная стадия, и виноватой окажется не та.
    """
    raise NotImplementedError

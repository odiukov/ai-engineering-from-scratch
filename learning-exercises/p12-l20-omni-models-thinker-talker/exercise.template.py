"""
Omni-модели: Qwen2.5-Omni и разделение Thinker/Talker

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l20-omni-models-thinker-talker
Разбор:  /check-code p12-l20-omni-models-thinker-talker
"""

STAGE_MS_7B = {
    "mic_to_tokens": 60.0,
    "prefill": 150.0,
    "first_thinker_token": 40.0,
    "talker_first_token": 20.0,
    "speech_tokens_commit": 40.0,
    "rvq_decode": 30.0,
    "waveform_decode": 65.0,
}
SIZE_SENSITIVE_STAGES = ("prefill",)


def ttfab_ms(stages):
    """Time-to-first-audio-byte: сумма задержек всех последовательных стадий.

    ttfab_ms({"prefill": 150.0, "rvq_decode": 30.0})  ->  180.0
    ttfab_ms({})                                      ->  0.0

    TTFAB — это время от конца речи пользователя до первого сэмпла ответа.
    Разговор перестаёт ощущаться лагающим примерно ниже 500 мс.
    """
    raise NotImplementedError


def scaled_budget(stages, params_b, ref_b=7.0):
    """Пересчёт бюджета под другой размер Thinker. Возвращает НОВЫЙ словарь.

    scaled_budget({"prefill": 150.0, "rvq_decode": 30.0}, 70.0)
        ->  {"prefill": 1500.0, "rvq_decode": 30.0}
    scaled_budget({"prefill": 150.0}, 7.0)  ->  {"prefill": 150.0}

    Растягиваются только стадии из SIZE_SENSITIVE_STAGES, и линейно по
    params_b / ref_b. Остальные копируются как есть.

    Ловушка: словарь на входе править нельзя. Бюджет обычно считают в цикле
    по нескольким размерам модели, и испорченный STAGE_MS_7B после первой
    итерации отравит все следующие.
    """
    raise NotImplementedError


def speech_tokens_needed(duration_s, token_rate_hz=50):
    """Сколько дискретных speech-токенов нужно на duration_s секунд речи.

    speech_tokens_needed(1.0)   ->  50
    speech_tokens_needed(2.5)   ->  125
    speech_tokens_needed(0.0)   ->  0

    Базовый слой residual-VQ у Qwen2.5-Omni идёт на 50 Гц: 50 токенов на
    секунду звука независимо от того, 16 кГц там или 24 кГц на выходе
    волнового декодера.

    Возвращай int: половины токена не бывает, округляй вверх — недосказанный
    хвост слышно как обрыв.
    """
    raise NotImplementedError


def talker_keeps_up(talker_tok_per_s, token_rate_hz=50):
    """Успевает ли Talker выдавать речь в реальном времени.

    talker_keeps_up(80)   ->  True
    talker_keeps_up(50)   ->  True    (ровно вровень — уже годится)
    talker_keeps_up(30)   ->  False

    Именно поэтому Talker делают маленьким (200-300M): 7B-модель на H100
    выдаёт 30-80 ток/с и на 50 Гц речи начинает отставать, а отставание в
    стриминге накапливается — пауза растёт с каждой секундой ответа.
    """
    raise NotImplementedError


def pipeline_total_ms(n_text_tokens, thinker_ms_per_token, talker_ms_per_token,
                      startup_ms=0.0):
    """Время до конца ответа, когда Thinker и Talker работают параллельно.

    pipeline_total_ms(10, 40.0, 20.0)         ->  400.0
    pipeline_total_ms(10, 40.0, 20.0, 100.0)  ->  500.0
    pipeline_total_ms(0, 40.0, 20.0, 100.0)   ->  100.0

    Конвейер упирается в самую медленную ступень: пока Thinker думает над
    токеном i+1, Talker уже озвучивает токен i. Значит на каждый текстовый
    токен тратится max(thinker, talker), а не их сумма.

    startup_ms — то, что нельзя совместить: прогрев до первого токена.

    Сравни с последовательной схемой (сначала весь текст, потом вся речь):
    n * (thinker + talker). Разница и есть выигрыш стриминга.
    """
    raise NotImplementedError


def tmrope_positions(events, resolution_hz=25):
    """TMRoPE: позиция токена — это его абсолютное ВРЕМЯ, а не место в списке.

    Событие — кортеж (t_seconds, modality, token). Возвращает список
    целых позиций в том же порядке, что и вход.

    tmrope_positions([(0.0, "audio", "a"), (0.04, "vision", "v")], 25)
        ->  [0, 1]
    tmrope_positions([(1.0, "audio", "a"), (1.01, "vision", "v")], 25)
        ->  [25, 25]

    Позиция = round(t * resolution_hz). Два токена, попавшие в один
    временной бин, получают ОДНУ позицию: модель видит их как
    одновременные. Ровно это и нужно, чтобы «махнул рукой, говоря привет»
    сработало.

    Ловушка: round в Python банковское — round(0.5) это 0, а round(1.5)
    это 2. Для позиций это допустимо, но не удивляйся результату.
    """
    raise NotImplementedError


def interleave_by_time(events):
    """Упорядочить разномодальные события по времени, а не по модальности.

    interleave_by_time([(2.0, "text", "t"), (1.0, "audio", "a")])
        ->  [(1.0, "audio", "a"), (2.0, "text", "t")]

    Наивная раскладка «сначала все кадры, потом весь звук, потом текст»
    рвёт временную связь: модель больше не знает, что жест и слово
    случились одновременно.

    Сортировка обязана быть УСТОЙЧИВОЙ: при равных временных метках порядок
    поступления сохраняется, иначе один и тот же вход даст разный контекст
    от запуска к запуску.
    """
    raise NotImplementedError


def turn_end_frame(energies, frame_ms=20, silence_ms=200, energy_floor=0.01):
    """VAD: индекс кадра, на котором можно считать, что пользователь договорил.

    Возвращает индекс ПОСЛЕДНЕГО кадра тишины в первой достаточно длинной
    паузе, или None, если такой паузы не было.

    turn_end_frame([1.0, 1.0] + [0.0] * 10, frame_ms=20, silence_ms=200)  ->  11
    turn_end_frame([1.0] * 20)                                            ->  None

    Кадр считается тихим, если его энергия строго ниже energy_floor.
    Пауза засчитывается, когда подряд набралось ceil(silence_ms / frame_ms)
    тихих кадров.

    Ловушка: слишком маленький silence_ms обрывает пользователя на вдохе,
    слишком большой добавляет к TTFAB чистое ожидание. 200 мс — компромисс
    half-duplex из урока.
    """
    raise NotImplementedError

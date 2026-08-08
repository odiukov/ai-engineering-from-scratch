"""
Разделение prefill и decode: два пула и цена передачи KV

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l17-disaggregated-prefill-decode
Разбор:  /check-code p17-l17-disaggregated-prefill-decode
"""

KV_BYTES_PER_TOKEN = 163_840
TRANSFER_SETUP_MS = 20.0
PREFILL_TPS = 8_000.0    # один forward по всему промпту, упор в вычисления
DECODE_TPS = 40.0        # по токену за шаг на один запрос, упор в память
COLOCATION_PENALTY = 0.30
LINK_RDMA_GBPS = 100.0
LINK_TCP_GBPS = 10.0


class DisaggError(Exception):
    """Пулы или линк спрошены о невозможном: отрицательные токены, мёртвый линк.

    Свой класс, а не ValueError и тем более не RuntimeError: заготовка бросает
    NotImplementedError, который наследуется от RuntimeError, и тест на
    родительский класс прошёл бы зелёным по пустому файлу.
    """
    pass


def kv_bytes(prompt_tokens, bytes_per_token=KV_BYTES_PER_TOKEN):
    """Сколько байт KV-кэша накопил prefill на промпте такой длины.

    kv_bytes(4000)  ->  655_360_000    (80 * 2 * 8 * 128 * 4000)
    kv_bytes(0)     ->  0

    Это то, что придётся физически перевезти из prefill-пула в decode-пул.
    Линейно по длине промпта: на каждый токен свой K и V в каждом слое.

    DisaggError на отрицательную длину и на неположительный bytes_per_token.
    """
    raise NotImplementedError


def transfer_ms(prompt_tokens, link_gbps, bytes_per_token=KV_BYTES_PER_TOKEN):
    """Время передачи KV между пулами: рукопожатие плюс байты по линку.

    transfer_ms(4000, LINK_RDMA_GBPS)  ->  26.5536 (20 + 6.5536 передачи)
    transfer_ms(4000, LINK_TCP_GBPS)   ->  85.536  (20 + 65.536 передачи)
    transfer_ms(0, LINK_RDMA_GBPS)     ->  20.0   (везти нечего, а хоп есть)

    Разбор: 655.36 МБ по 100 ГБ/с = 6.5536 мс, по 10 ГБ/с = 65.536 мс.

    Ловушка: слагаемое TRANSFER_SETUP_MS постоянное. Именно оно, а не байты,
    решает судьбу коротких промптов: на 100 токенах везти почти нечего, а
    рукопожатие всё равно стоит целых 20 мс.

    DisaggError на неположительную скорость линка.
    """
    raise NotImplementedError


def phase_ms(prompt_tokens, output_tokens,
             prefill_tps=PREFILL_TPS, decode_tps=DECODE_TPS):
    """Время двух фаз на ВЫДЕЛЕННОЙ GPU, без чужой нагрузки рядом.

    Возвращает dict: prefill_ms, decode_ms.

    phase_ms(4000, 300)  ->  {'prefill_ms': 500.0, 'decode_ms': 7500.0}
    phase_ms(400, 30)    ->  {'prefill_ms': 50.0,  'decode_ms': 750.0}

    Обрати внимание на пропорцию: decode на порядок дольше prefill, потому что
    prefill проглатывает весь промпт одним forward, а decode делает по шагу на
    токен. Отсюда и разные узкие места — вычисления против полосы памяти.

    DisaggError на отрицательные токены и неположительные скорости.
    """
    raise NotImplementedError


def colocated_ms(prompt_tokens, output_tokens, penalty=COLOCATION_PENALTY,
                 prefill_tps=PREFILL_TPS, decode_tps=DECODE_TPS):
    """Время запроса на общей GPU: prefill растянут штрафом, decode как есть.

    colocated_ms(4000, 300)  ->  примерно 8214.29
    colocated_ms(4000, 300, penalty=0.0)  ->  8000.0  (штрафа нет)

    Разбор: 500 мс prefill / (1 - 0.30) = 714.29, плюс 7500 мс decode.

    Ловушка: штраф — это ДЕЛЕНИЕ на (1 - penalty), а не умножение на
    (1 + penalty). Penalty — доля отобранного времени устройства, а не надбавка
    сверху. На 0.30 разница между 1.4286 и 1.30 — это 10% времени prefill.

    Decode штрафом не облагается: он и так упирается в полосу памяти, ему
    соседний prefill почти не мешает. Из-за этого в disagg_gain_ms decode
    сократится целиком.

    DisaggError на penalty вне [0, 1).
    """
    raise NotImplementedError


def disaggregated_ms(prompt_tokens, output_tokens, link_gbps,
                     prefill_tps=PREFILL_TPS, decode_tps=DECODE_TPS,
                     bytes_per_token=KV_BYTES_PER_TOKEN):
    """Время запроса на двух пулах: чистый prefill, передача KV, чистый decode.

    disaggregated_ms(4000, 300, LINK_TCP_GBPS)   ->  8085.536
    disaggregated_ms(4000, 300, LINK_RDMA_GBPS)  ->  8026.5536

    Разбор первого: 500 prefill + 85.536 передача + 7500 decode.

    Штрафа колокации здесь нет — в этом весь смысл разделения. Зато появился
    налог на передачу, которого в колокации нет вообще: там KV уже лежит в
    той самой HBM, где будет считаться decode.
    """
    raise NotImplementedError


def disagg_gain_ms(prompt_tokens, output_tokens, link_gbps,
                   penalty=COLOCATION_PENALTY, prefill_tps=PREFILL_TPS,
                   decode_tps=DECODE_TPS, bytes_per_token=KV_BYTES_PER_TOKEN):
    """Выигрыш разделения в миллисекундах. Отрицательный — разделять не надо.

    disagg_gain_ms(4000, 300, LINK_TCP_GBPS)  ->  примерно 128.75
    disagg_gain_ms(200, 300, LINK_TCP_GBPS)   ->  примерно -12.56
    disagg_gain_ms(4000, 999, LINK_TCP_GBPS)  ->  ровно столько же, сколько
                                                  с 300 — decode сократился

    Разбор второго: на 200 токенах штраф колокации отбирает всего 8.57 мс, а
    рукопожатие и передача стоят 23.28. Это и есть «Prompts < 512 tokens:
    transfer tax dominates gain» из урока.

    Свойство, которое стоит понять раньше формулы: длина ОТВЕТА в выигрыш не
    входит. Decode считается одинаково в обеих схемах и в разности исчезает.
    Поэтому решение «разделять или нет» — это решение про длину ПРОМПТА.
    """
    raise NotImplementedError


def crossover_prompt_tokens(link_gbps, penalty=COLOCATION_PENALTY,
                            prefill_tps=PREFILL_TPS,
                            bytes_per_token=KV_BYTES_PER_TOKEN,
                            max_tokens=1 << 20):
    """Самый короткий промпт, на котором разделение уже выигрывает. Или None.

    crossover_prompt_tokens(LINK_TCP_GBPS)   ->  538
    crossover_prompt_tokens(LINK_RDMA_GBPS)  ->  386
    crossover_prompt_tokens(2.0)             ->  None

    Урок называет порог около 512 токенов — вот он, посчитанный: 538 на TCP.
    На RDMA порог ниже: линк быстрее, налог на байт меньше, окупается раньше.

    None означает, что порога нет вообще. Выигрыш растёт с длиной промпта со
    скоростью (penalty/(1-penalty))/prefill_tps на токен, а налог — со
    скоростью bytes_per_token/link. Если линк медленнее примерно 3.06 ГБ/с,
    вторая скорость больше первой, и удлинение промпта делает только хуже: чем
    длиннее промпт, тем больше проигрыш. При этой геометрии граница — около
    3.06 ГБ/с. Это и есть строчка урока «No RDMA
    fabric: TCP transfer tax is heavier» в предельном виде.

    Ловушка: не ищи порог линейным перебором до миллиона — выигрыш монотонен
    по длине промпта, годится двоичный поиск.

    DisaggError на max_tokens < 1.
    """
    raise NotImplementedError


def fleet_report(requests, link_gbps, penalty=COLOCATION_PENALTY,
                 prefill_tps=PREFILL_TPS, decode_tps=DECODE_TPS,
                 bytes_per_token=KV_BYTES_PER_TOKEN):
    """Прогнать смесь трафика через обе схемы и сравнить.

    requests — список пар (prompt_tokens, output_tokens).

    Возвращает dict:
      requests            — сколько запросов,
      colocated_ms        — суммарное время на общих GPU,
      disaggregated_ms    — суммарное время на двух пулах,
      gain_ms             — разность, может быть отрицательной,
      gain_pct            — та же разность в процентах от колокации,
      helped, hurt        — сколько запросов выиграло и сколько проиграло.

    fleet_report([], LINK_TCP_GBPS)["gain_pct"]  ->  0.0

    Ради чего функция: решение принимается не по одному запросу, а по смеси.
    Флот из коротких промптов от разделения проигрывает, флот из длинных
    выигрывает, а реальный трафик — смесь, и знак итога зависит от неё. Именно
    поэтому в уроке RAG с префиксами на 8К называется лучшим кандидатом, а чат
    с короткими репликами — худшим.
    """
    raise NotImplementedError

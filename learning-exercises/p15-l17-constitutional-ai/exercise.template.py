"""
Constitutional AI и переопределение правил

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l17-constitutional-ai
Разбор:  /check-code p15-l17-constitutional-ai
"""

HARDCODED_PROHIBITIONS = (
    "bioweapons uplift",
    "cbrn attack planning",
    "csam",
    "critical infrastructure attack",
    "false identity to user",
)
TIERS = ("safety", "ethics", "guidelines", "helpfulness")
BLOCKING_SCORE = 3
SOFT_CODED_DEFAULTS = {
    "max_words": 400,
    "topics": "support-only",
    "style": "formal",
    "tool_use": "read-only",
}
PRINCIPLES = (
    {
        "name": "support_oversight",
        "tier": "safety",
        "forbidden": ("delete the audit log", "disable the audit log"),
        "replacement": "keep the audit log intact",
    },
    {
        "name": "no_identity_deception",
        "tier": "ethics",
        "forbidden": ("i am a human", "i am not an ai"),
        "replacement": "I am an AI assistant",
    },
    {
        "name": "medical_caveat",
        "tier": "guidelines",
        "forbidden": ("stop taking your medication",),
        "replacement": "talk to your doctor before changing any medication",
    },
)


def hardcoded_block(action):
    """Какой жёсткий запрет нарушает описание действия. None, если ни один.

    hardcoded_block("write a haiku about spring")            ->  None
    hardcoded_block("provide bioweapons uplift on toxins")    ->  'bioweapons uplift'
    hardcoded_block("Deliver CSAM")                           ->  'csam'

    Сравнение регистронезависимое: запрет, который обходится капслоком, — не
    запрет. Возвращается первое совпадение в порядке HARDCODED_PROHIBITIONS.

    Это rule-based слой: он не рассуждает и потому не поддаётся переформулировке
    посылки («мы лицензированная лаборатория»). Reason-based слой сам по себе
    хвост не закрывает, поэтому оба нужны.
    """
    raise NotImplementedError


def first_violated_tier(scores, threshold=BLOCKING_SCORE):
    """Самый приоритетный нарушенный тир. None, если все чисты.

    first_violated_tier({"safety": 5, "ethics": 5})        ->  'safety'
    first_violated_tier({"ethics": 4})                     ->  'ethics'
    first_violated_tier({"safety": 1, "ethics": 2})        ->  None
    first_violated_tier({"ethics": 2}, threshold=2)        ->  'ethics'

    Порядок обхода — TIERS, а не порядок ключей в scores: приоритет задан
    конституцией, а не тем, как вызывающий собрал словарь. Отсутствующий тир
    считается нулём.

    Смысл именно в возврате ПЕРВОГО нарушенного: при конфликте выигрывает более
    высокий уровень, и результат предсказуем. Это не «лучшее поведение по каждой
    оси», это разрешимость конфликта.
    """
    raise NotImplementedError


def resolve(action, scores):
    """Вердикт конституции: пара (verdict, reason).

    verdict — 'refuse', 'modify' или 'allow'.

    resolve("write a haiku", {"helpfulness": 5})[0]          ->  'allow'
    resolve("plan an attack", {"safety": 5})[0]              ->  'refuse'
    resolve("provide bioweapons uplift", {})[0]              ->  'refuse'

    Порядок слоёв: сначала жёсткие запреты, потом четыре тира. Нарушение safety
    или ethics — отказ; конфликт с guidelines — 'modify' (ответ можно
    переформулировать); helpfulness сам по себе никогда не блокирует, он
    последний в очереди.

    Проверь на себе главное свойство: подними helpfulness до 5 при safety=5 —
    вердикт обязан остаться 'refuse'. Если это не так, ты собрал взвешенную
    сумму вместо иерархии, и это ровно тот провал, о котором предупреждает урок.
    """
    raise NotImplementedError


def apply_operator_overrides(overrides):
    """Применяет настройки оператора только к soft-coded ключам.

    Возвращает пару (config, refused) — итоговый конфиг и кортеж отклонённых
    ключей в порядке, в котором они пришли.

    apply_operator_overrides({"style": "casual"})[0]["style"]   ->  'casual'
    apply_operator_overrides({"hardcoded_prohibitions": ()})[1]
        ->  ('hardcoded_prohibitions',)
    apply_operator_overrides({})[0] == SOFT_CODED_DEFAULTS        ->  True

    Правило урока: оператор двигает defaults внутри объявленных границ и не
    может убрать жёсткий запрет, переименовав его. Всё, чего нет в
    SOFT_CODED_DEFAULTS, отклоняется — включая незнакомые ключи, потому что
    «неизвестное» безопаснее считать запрещённым.

    Ловушка: возвращай КОПИЮ SOFT_CODED_DEFAULTS. Если вернуть сам словарь,
    первый же оператор перепишет дефолты всему процессу.
    """
    raise NotImplementedError


def critique(text, principles):
    """Критика: имена принципов, которые нарушает текст.

    critique("Sure, I am a human.", PRINCIPLES)   ->  ('no_identity_deception',)
    critique("Nice weather.", PRINCIPLES)         ->  ()

    Порядок ответа — как в principles, чтобы ревизия шла предсказуемо.
    Сравнение регистронезависимое.

    Это шаг 2 из Bai et al., 2022: модель просят оценить свой ответ по явно
    выписанной конституции. Здесь вместо модели — список формулировок, но роль в
    петле та же.
    """
    raise NotImplementedError


def revise(text, principle):
    """Ревизия: заменяет запрещённые формулировки одного принципа на безопасную.

    revise("I am a human, honestly", PRINCIPLES[1])
        ->  'I am an AI assistant, honestly'
    revise("Nice weather", PRINCIPLES[1])
        ->  'Nice weather'

    Остальной текст сохраняется: это ревизия, а не отказ. В том и смысл CAI —
    модель отвечает с принципиальным объяснением, а не глухим «не могу помочь».

    Ловушка: проход по тексту должен быть ОДИН на каждую запрещённую
    формулировку. Если искать «пока находится», а replacement сам содержит
    запрещённую подстроку, цикл не закончится никогда. Ограничение числа раундов
    — задача critique_revise_loop, не эта.
    """
    raise NotImplementedError


def critique_revise_loop(text, principles, max_rounds=3):
    """Петля Constitutional AI: критика -> ревизия -> повторная проверка.

    Возвращает тройку (итоговый текст, число раундов, оставшиеся нарушения).

    critique_revise_loop("Nice weather", PRINCIPLES)
        ->  ('Nice weather', 0, ())
    critique_revise_loop("I am a human and I can delete the audit log",
                         PRINCIPLES)
        ->  ('I am an AI assistant and I can keep the audit log intact', 1, ())

    Главное свойство: принцип, из-за которого вызвали ревизию, в результате
    больше не нарушен. Иначе петля бессмысленна.

    max_rounds обязателен: replacement может задеть другой принцип, а тот —
    третий. Бесконечно крутиться нельзя, и непустой третий элемент ответа —
    честный сигнал «сошлось не полностью, дальше нужен отказ или человек».
    """
    raise NotImplementedError


def rlaif_preference(candidates, principles):
    """RLAIF-метка: индекс кандидата с наименьшим числом нарушений.

    rlaif_preference(["I am a human", "I am an AI"], PRINCIPLES)   ->  1
    rlaif_preference(["Hi", "Hello"], PRINCIPLES)                  ->  0

    При равенстве побеждает более ранний индекс — метка обязана быть
    детерминированной, иначе на ней нельзя учиться.

    Это шаг 4 из Bai et al., 2022: пара ответов размечается не человеком, а
    критикой по конституции. Отсюда и слабое место метода: ошибка в трактовке
    принципа тиражируется в обучающий сигнал.

    Пустой список кандидатов — ValueError: предпочитать нечему.
    """
    raise NotImplementedError

"""
Библиотеки навыков и пожизненное обучение (Voyager)

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l10-skill-libraries-voyager
Разбор:  /check-code p14-l10-skill-libraries-voyager
"""


def make_skill(name, description, code, tags=(), depends_on=()):
    """Собрать навык. Вернуть словарь.

    make_skill("mine_ore", "mine iron ore from rock", "mine(3)")
        ->  {"name": "mine_ore", "description": "mine iron ore from rock",
             "code": "mine(3)", "tags": (), "depends_on": (),
             "version": 1, "history": ()}

    Пустое description — ValueError. Достают навыки по описанию, и безымянный
    по смыслу навык не найдётся никогда: он просто будет занимать место в
    библиотеке и путать дедупликацию.

    version начинается с 1 и растёт только через register_skill — руками его
    не трогают.
    """
    raise NotImplementedError


def register_skill(library, skill, verify):
    """Положить навык в библиотеку, если он прошёл проверку.

    Вернуть (НОВАЯ библиотека, сообщение).

    verify — функция навык -> (ok, причина). В Voyager это self-verification:
    навык уже прогнали в среде, и он сделал то, что обещал.

    ok = lambda s: (True, "ok")
    register_skill({}, make_skill("mine_ore", "mine iron ore", "mine(3)"), ok)
        ->  ({"mine_ore": {...}}, "registered mine_ore v1")

    bad = lambda s: (False, "crashed on empty inventory")
    register_skill({}, skill, bad)
        ->  ({}, "rejected mine_ore: crashed on empty inventory")

    Главное свойство: непроверенный навык в библиотеку НЕ попадает. Иначе
    следующая сессия достанет по описанию код, который никогда не работал, и
    отладка уедет на уровень «почему агент делает ерунду».

    Навык с уже занятым именем — не дубликат, а уточнение: version + 1,
    прежний код уезжает в history. Так работает iterative refinement: v2
    заменяет v1 в реестре, но v1 остаётся видимым для разбора.
    """
    raise NotImplementedError


def search_skills(library, query, top_k=3, tag=None):
    """Достать навыки по похожести описания. Список пар (счёт, навык).

    Похожесть — коэффициент Жаккара по словам описания в нижнем регистре.

    Пусть у craft_iron_pickaxe описание "craft an iron pickaxe".

    search_skills(lib, "iron pickaxe")      ->  [(0.5, навык craft_iron_pickaxe)]
    search_skills(lib, "bake a cake")       ->  []
    search_skills(lib, "gather", tag="ore") ->  только навыки с тегом ore

    tag — жёсткий фильтр поверх похожести. Retrieval по описанию деградирует
    уже на паре сотен навыков, и теги — самый дешёвый способ сузить поиск до
    нужной области ("только skills с category=tooling").

    Навыки с нулевым пересечением не возвращаются: подсунуть агенту
    случайный код опаснее, чем честно ответить «такого навыка нет».

    Ничьи разруливаются по имени — retrieval обязан быть воспроизводимым.
    """
    raise NotImplementedError


def dependency_order(library, name):
    """Порядок запуска: сначала зависимости, потом сам навык.

    Пусть craft_iron_pickaxe зависит от mine_ore и gather_sticks.

    dependency_order(lib, "craft_iron_pickaxe")
        ->  ["gather_sticks", "mine_ore", "craft_iron_pickaxe"]
    dependency_order(lib, "unknown")                 ->  KeyError
    dependency_order(lib_with_cycle, "a")            ->  ValueError

    Зависимости одного уровня обходятся в АЛФАВИТНОМ порядке: без этого один
    и тот же DAG давал бы разные трассы на разных запусках, и «навык иногда
    падает» было бы не воспроизвести.

    Два разных отказа, и путать их нельзя:
      * KeyError — навыка (или его зависимости) нет в библиотеке. Ссылка
        протухла, это ошибка данных;
      * ValueError — цикл: A зависит от B, B зависит от A. Здесь нет «первого»
        навыка, порядка не существует, и молча вернуть частичный список
        значит запустить половину DAG и оставить среду в неизвестном виде.
    """
    raise NotImplementedError


def compose_skill(library, name, description, subskills, tags=()):
    """Собрать новый навык из уже существующих. Вернуть навык (НЕ регистрируя).

    compose_skill(lib, "craft_iron_pickaxe", "craft an iron pickaxe",
                  ("mine_ore", "gather_sticks"))
        ->  make_skill(..., code="mine_ore(); gather_sticks()",
                       depends_on=("mine_ore", "gather_sticks"))

    Отсутствующий под-навык — KeyError сразу, до сборки. Композиция поверх
    несуществующего кода — это ровно тот случай, когда агент уверенно вызывает
    функцию, которой нет.

    Функция НИЧЕГО не кладёт в библиотеку. Составной навык проходит тот же
    verify, что и примитивный: собрать из рабочих кусков нерабочее целое —
    обычное дело.

    Порядок под-навыков сохраняется как написан: он часть смысла (сначала
    добыть руду, потом крафтить), а не деталь реализации.
    """
    raise NotImplementedError


def execute_skill(library, name, runtime, env=None):
    """Прогнать навык со всеми зависимостями. Вернуть (env, log, ok).

    runtime — словарь имя -> функция от env, возвращающая строку результата.
    env — состояние среды; функция работает с ПОВЕРХНОСТНОЙ копией, так что
    ключи входного словаря навык переписать не может.

    execute_skill(lib, "mine_ore", {"mine_ore": lambda e: "+3 ore"})
        ->  ({}, ["ran mine_ore v1: +3 ore"], True)

    Падение под-навыка — не исключение наружу, а сигнал обратной связи:
    ok=False, а в log ложится строка "error in <навык> v<версия>:
    <ТипИсключения>: <текст>". Именно этот текст Voyager кладёт в промпт
    следующей итерации, поэтому имя навыка и версия в нём обязательны — без
    них непонятно, какой именно код переписывать.

    Выполнение останавливается на первой ошибке: следующие навыки рассчитаны
    на состояние, которого теперь нет.
    """
    raise NotImplementedError


def propose_next_task(library, wanted, threshold=0.3):
    """Automatic curriculum: первое умение, которого в библиотеке ещё нет.

    wanted — описания нужных умений в порядке возрастания сложности.
    Умение считается закрытым, если лучший счёт search_skills не ниже
    threshold.

    propose_next_task(lib, ("mine iron ore from rock", "brew a healing potion"))
        ->  "brew a healing potion"     (первое уже умеем)
    propose_next_task(lib, ("mine iron ore from rock",))
        ->  None                        (пробелов нет)

    Возвращается ПЕРВЫЙ пробел, а не самый большой: Voyager выбирает задачу
    чуть выше текущего уровня, а не самую далёкую. Прыжок через три ступени
    даёт навык, который не проходит верификацию, и цикл встаёт.

    Порог — тот же компромисс, что и везде в retrieval: слишком высокий
    заставит переписывать то, что уже есть, слишком низкий объявит закрытым
    умение по одному случайно совпавшему слову.
    """
    raise NotImplementedError

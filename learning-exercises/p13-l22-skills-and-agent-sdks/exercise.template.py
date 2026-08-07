"""
Skills и Agent SDK

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l22-skills-and-agent-sdks
Разбор:  /check-code p13-l22-skills-and-agent-sdks
"""

import re

SKILL_ROOTS = ("~/.claude/skills", "skills")
SKILL_FILE = "SKILL.md"
SUBRESOURCE_RE = re.compile(r"[\w][\w./-]*\.(?:md|txt|py|sh|json|yaml)\b")
URL_RE = re.compile(r"https?://\S+")


def parse_frontmatter(text):
    """Разобрать YAML-блок в начале файла. Вернуть (метаданные, тело).

    parse_frontmatter("---\\nname: x\\ndescription: y\\n---\\n\\n# Body\\n")
        ->  ({"name": "x", "description": "y"}, "\\n# Body\\n")
    parse_frontmatter("# Body only\\n")  ->  ({}, "# Body only\\n")

    Поддерживаются три формы значения:
        name: release-notes-writer          обычная строка
        triggers: [changelog, release]      список в строку
        triggers:                           блочный список
          - changelog
          - release

    Две ловушки:
      * значение может содержать двоеточие — "description: Use when: ...".
        Резать надо по ПЕРВОМУ двоеточию, split(":") без maxsplit потеряет
        хвост описания;
      * блок, открытый "---" и не закрытый, — ValueError. Молча считать весь
        файл телом значит подать модели YAML в качестве инструкций.
    """
    raise NotImplementedError


def parse_skill(root, text):
    """Собрать skill из содержимого SKILL.md. Словарь или ValueError.

    parse_skill("skills/pr-reviewer", "---\\nname: pr-reviewer\\n---\\nBody")
        ->  {"name": "pr-reviewer", "description": "", "triggers": (),
             "body": "Body", "root": "skills/pr-reviewer",
             "folder": "pr-reviewer", "nameMatchesFolder": True}

    root — папка skill, без имени файла: субресурсы ищутся относительно неё.

    Файл без frontmatter-поля name — ValueError. Рантайм кладёт skill в
    реестр по name, и безымянный skill невозможно ни позвать, ни перекрыть.

    nameMatchesFolder — не ошибка, а предупреждение: рантаймы ищут папку по
    имени skill, и расхождение однажды сломает ссылку на субресурсы.
    Возвращаем флаг, решает вызывающий.

    triggers могут прийти строкой вместо списка — приводим к кортежу, чтобы
    match_skill не разбирал два формата.
    """
    raise NotImplementedError


def discover_skills(files, roots=None):
    """Просканировать корни и собрать реестр skill по имени.

    files — словарь путь -> содержимое. roots по умолчанию SKILL_ROOTS.

    discover_skills({"skills/pr-reviewer/SKILL.md": "---\\nname: pr-reviewer\\n---\\n"})
        ->  {"pr-reviewer": {...}}

    Берутся только файлы вида <root>/<одна папка>/SKILL.md. Вложенность
    глубже игнорируется: SKILL.md внутри подпапки — это чужой субресурс,
    а не отдельный skill.

    Одинаковое имя в двух корнях — не ошибка, а перекрытие: побеждает корень,
    стоящий в roots ПОЗЖЕ. Порядок в SKILL_ROOTS для того и задан, и
    результат не зависит от того, в каком порядке лежат ключи в files.
    """
    raise NotImplementedError


def match_skill(registry, user_text):
    """Найти skill, чей триггер встречается в реплике. Имя или None.

    Пусть у "release-notes-writer" триггер "release notes", а у
    "pr-reviewer" — "review".

    match_skill(reg, "draft the release notes")  ->  "release-notes-writer"
    match_skill(reg, "what is MCP?")             ->  None

    Сравнение без учёта регистра: пользователь пишет как хочет.

    Совпасть могут несколько триггеров сразу, и выбор обязан быть
    детерминированным. Правило: побеждает САМЫЙ ДЛИННЫЙ совпавший триггер —
    он конкретнее. При равной длине — меньшее имя skill. Без этого правила
    "review" и "review pr" разрешались бы по порядку обхода словаря, то есть
    по порядку установки skill на машину.
    """
    raise NotImplementedError


def subresource_links(body):
    """Файлы, на которые ссылается тело skill. Отсортированный кортеж.

    subresource_links("See style-guide.md and scripts/gen.sh")
        ->  ("scripts/gen.sh", "style-guide.md")
    subresource_links("Read https://example.com/style-guide.md")
        ->  ()

    Это и есть список того, что progressive disclosure подтянет по
    требованию, — и ровно тот текст, который НЕ занимает контекст, пока
    skill не позвали.

    Внешние ссылки не считаются: URL агент не читает с диска. Поэтому URL-ы
    вырезаются из текста ДО поиска — иначе "example.com/style-guide.md"
    отдал бы кусок чужого адреса как локальный файл.

    Сам SKILL.md из списка исключён: skill не является собственным
    субресурсом, и подгружать его повторно значит удвоить промпт.
    """
    raise NotImplementedError


def read_subresource(files, skill, filename):
    """Прочитать файл из папки skill.

    read_subresource(files, skill, "style-guide.md")  ->  текст файла
    read_subresource(files, skill, "../../secrets.md")  ->  ValueError
    read_subresource(files, skill, "missing.md")        ->  FileNotFoundError

    Два разных отказа, и путать их нельзя:
      * ValueError — попытка выйти за папку skill. Тело SKILL.md пишет
        человек или генератор, и путь оттуда — это недоверенный ввод.
        Абсолютный путь и ".." отсекаются до всякого чтения;
      * FileNotFoundError — обещанного файла просто нет. Это опечатка автора
        skill, а не атака.
    """
    raise NotImplementedError


def build_system_prompt(skill, user_task, files, agents_md=None, disclose=True):
    """Собрать системный промпт из трёх слоёв.

    Порядок блоков — порядок жизненного цикла сессии:
        AGENTS.md (на старте) -> SKILL.md (при вызове) -> субресурсы
        (по требованию) -> задача пользователя.

    build_system_prompt(skill, "draft 1.4.0", files, disclose=False)
        ->  промпт без текста style-guide.md
    build_system_prompt(skill, "draft 1.4.0", files, disclose=True)
        ->  тот же промпт плюс блок с содержимым style-guide.md

    disclose=False — это и есть progressive disclosure в выключенном
    состоянии: skill загружен, а детали ещё не стоили ни одного токена.

    Отсутствующий субресурс не роняет сессию: в промпт уходит пометка
    "(missing)". Модель должна узнать, что обещанного файла нет, — иначе она
    будет ссылаться на несуществующие правила.
    """
    raise NotImplementedError


def parse_agents_md(text):
    """Разобрать AGENTS.md в словарь раздел -> кортеж пунктов.

    parse_agents_md("# Project\\n\\n## Conventions\\n- strict mode\\n- pnpm test\\n")
        ->  {"Conventions": ("strict mode", "pnpm test")}

    Берутся только пункты под разделами "## ". Заголовок "# " закрывает
    текущий раздел: он начинает новый документ, а не подраздел.

    Пункты до первого "## " отбрасываются намеренно: у них нет темы, и
    подставить их в промпт некуда. AGENTS.md, у которого всё лежит россыпью
    под заголовком файла, стоит переписать, а не читать наугад.
    """
    raise NotImplementedError

"""
Skills и Agent SDK — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Claude Agent SDK делает всё это на старте сессии, и разработчик видит только
готовый реестр skill-ов. Здесь мы пишем сам загрузчик. Соответствие
настоящему рантайму:

    parse_frontmatter    <-  разбор YAML-блока SKILL.md (без pyyaml)
    parse_skill          <-  сборка объекта Skill из файла
    discover_skills      <-  скан ~/.claude/skills и ./skills, ключ — name
    match_skill          <-  выбор skill по реплике пользователя
    subresource_links    <-  что именно обещает progressive disclosure
    read_subresource     <-  подгрузка файла из папки skill по требованию
    build_system_prompt  <-  склейка AGENTS.md + SKILL.md + задачи
    parse_agents_md      <-  чтение AGENTS.md на старте сессии

Файловая система приходит параметром: словарь путь -> текст. Так тест не
зависит от реального диска, а логика загрузчика остаётся ровно той же.
"""

import re

# Порядок = возрастание приоритета. Проектные skill перекрывают
# пользовательские: у команды свой release-notes-writer, и он должен победить.
SKILL_ROOTS = ("~/.claude/skills", "skills")

SKILL_FILE = "SKILL.md"

# Что считается субресурсом progressive disclosure.
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

    def unquote(value):
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            return value[1:-1]
        return value

    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 3)
    if end == -1:
        raise ValueError("frontmatter is opened by --- but never closed")
    raw = text[4:end]
    body = text[end + 5 :]

    meta = {}
    list_key = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and list_key is not None:
            meta[list_key].append(unquote(stripped[2:]))
            continue
        if ":" not in stripped:
            raise ValueError(f"frontmatter line without ':': {line!r}")
        key, value = stripped.split(":", 1)  # maxsplit=1 — вот та самая ловушка
        key, value = key.strip(), value.strip()
        if value == "":
            meta[key] = []
            list_key = key
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            meta[key] = [unquote(v) for v in inner.split(",")] if inner else []
            list_key = None
        else:
            meta[key] = unquote(value)
            list_key = None
    return meta, body


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
    meta, body = parse_frontmatter(text)
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{root}/{SKILL_FILE}: frontmatter has no 'name'")
    name = name.strip()
    triggers = meta.get("triggers", [])
    if isinstance(triggers, str):
        triggers = [triggers]
    folder = root.rstrip("/").rsplit("/", 1)[-1]
    return {
        "name": name,
        "description": meta.get("description", ""),
        "triggers": tuple(t.strip() for t in triggers if t.strip()),
        "body": body.strip(),
        "root": root.rstrip("/"),
        "folder": folder,
        "nameMatchesFolder": folder == name,
    }


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
    registry = {}
    for root in roots if roots is not None else SKILL_ROOTS:
        prefix = root.rstrip("/") + "/"
        suffix = "/" + SKILL_FILE
        # sorted — чтобы порядок обхода не зависел от порядка вставки в files
        for path in sorted(files):
            if not path.startswith(prefix) or not path.endswith(suffix):
                continue
            inner = path[len(prefix) : -len(suffix)]
            if "/" in inner or not inner:
                continue
            skill = parse_skill(path[: -len(suffix)], files[path])
            registry[skill["name"]] = skill
    return registry


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
    lowered = user_text.lower()
    best_length = 0
    best_name = None
    # sorted(registry) даёт алфавитный порядок, а строгое > ниже оставляет
    # победителем первого при равной длине — отсюда и детерминизм.
    for name in sorted(registry):
        for trigger in registry[name]["triggers"]:
            needle = trigger.strip().lower()
            if needle and needle in lowered and len(needle) > best_length:
                best_length, best_name = len(needle), name
    return best_name


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
    cleaned = URL_RE.sub(" ", body)
    found = []
    for match in SUBRESOURCE_RE.finditer(cleaned):
        ref = match.group(0)
        if ref == SKILL_FILE or ref in found:
            continue
        found.append(ref)
    return tuple(sorted(found))


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
    if filename.startswith("/") or ".." in filename.split("/"):
        raise ValueError(f"subresource escapes the skill root: {filename}")
    path = skill["root"] + "/" + filename
    if path not in files:
        raise FileNotFoundError(path)
    return files[path]


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
    blocks = []
    if agents_md:
        lines = ["# Project context (AGENTS.md)"]
        for section, bullets in agents_md.items():
            lines.append(f"## {section}")
            lines.extend(f"- {b}" for b in bullets)
        blocks.append("\n".join(lines))
    blocks.append(f"# Skill: {skill['name']}\n{skill['description']}\n\n{skill['body']}")
    if disclose:
        for ref in subresource_links(skill["body"]):
            try:
                content = read_subresource(files, skill, ref)
            except FileNotFoundError:
                content = "(missing)"
            blocks.append(f"# Subresource: {ref}\n{content}")
    blocks.append(f"# User task\n{user_task}")
    return "\n\n".join(blocks)


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
    sections = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current = stripped[3:].strip()
            sections.setdefault(current, [])
        elif stripped.startswith("# "):
            current = None
        elif current is not None and stripped.startswith("- "):
            sections[current].append(stripped[2:].strip())
    return {name: tuple(bullets) for name, bullets in sections.items()}

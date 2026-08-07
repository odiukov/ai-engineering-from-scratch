"""
MCP Apps — интерактивные UI-ресурсы ui:// (SEP-1724)

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l14-mcp-apps
Разбор:  /check-code p13-l14-mcp-apps
"""

UI_SCHEME = "ui://"
MCP_APP_MIME = "text/html;profile=mcp-app"
HOST_METHODS = (
    "host.callTool",
    "host.readResource",
    "host.getPrompt",
    "host.close",
)
KNOWN_PERMISSIONS = ("camera", "microphone", "geolocation", "network:*")
ERROR_CODES = {
    "wildcard_origin": -32001,
    "origin_mismatch": -32001,
    "malformed": -32600,
    "bad_jsonrpc": -32600,
    "unknown_method": -32601,
    "no_handler": -32601,
}


def is_ui_uri(uri):
    """Правильный ли это ui:// URI для MCP App.

    is_ui_uri("ui://notes/timeline")     ->  True
    is_ui_uri("https://notes/timeline")  ->  False
    is_ui_uri("ui://")                   ->  False

    Требования: схема ровно "ui://", после неё непустой путь, внутри нет
    пробелов и переводов строк. Пробел в URI — почти всегда склеенная
    строка, а не адрес.
    """
    raise NotImplementedError


def ui_resource_contents(uri, html):
    """Ответ resources/read для UI-ресурса.

    ui_resource_contents("ui://notes/timeline", "<!doctype html>")
      ->  {"contents": [{"uri": "ui://notes/timeline",
                         "mimeType": "text/html;profile=mcp-app",
                         "text": "<!doctype html>"}]}

    Ловушка: MIME обязан быть с профилем. Просто "text/html" хост покажет
    как текст, а не как приложение, и никакого iframe не будет.

    Не-ui:// адрес — ValueError: обычные ресурсы читаются другим путём.
    """
    raise NotImplementedError


def csp_header(csp):
    """Собрать строку Content-Security-Policy из camelCase-словаря.

    csp_header({"defaultSrc": "'self'", "scriptSrc": "'self' 'unsafe-inline'"})
      ->  "default-src 'self'; script-src 'self' 'unsafe-inline'"

    Ключи в _meta.ui.csp пишутся как defaultSrc, в заголовке — default-src.
    Директивы сортируются по имени: заголовок должен быть воспроизводимым,
    иначе его нельзя ни сравнить, ни захэшировать в манифесте.
    """
    raise NotImplementedError


def csp_findings(csp):
    """Аудит CSP: отсортированный список кодов замечаний. Пустой — всё строго.

    csp_findings({"defaultSrc": "'self'", "connectSrc": "'self'"})  ->  []
    csp_findings({"defaultSrc": "'self'", "connectSrc": "*"})
      ->  ["wildcard_connect_src"]

    Что ищем:
      * "missing_default_src"   — нет defaultSrc, всё остальное не важно;
      * "wildcard_connect_src"  — connectSrc со звёздочкой: UI сможет слить
                                  данные пользователя куда угодно;
      * "wildcard_script_src"   — scriptSrc со звёздочкой: чужой код в iframe;
      * "unsafe_inline_script"  — 'unsafe-inline' в scriptSrc.

    'unsafe-inline' — замечание, а не запрет: спека сама показывает его
    в примере, но nonce лучше.
    """
    raise NotImplementedError


def review_permissions(requested):
    """Разложить запрошенные разрешения на «спросим пользователя» и «откажем».

    review_permissions([])                    ->  {"prompt": [], "rejected": []}
    review_permissions(["camera", "gpu"])
      ->  {"prompt": ["camera"], "rejected": ["gpu"]}

    Известное разрешение не выдаётся молча: каждое — отдельный вопрос перед
    рендером UI. Неизвестное отклоняется, а не пропускается: список
    KNOWN_PERMISSIONS — это allowlist, и «неизвестно, значит можно» здесь
    ровно та дыра, из-за которой существует эта функция.

    Оба списка отсортированы, дубликаты схлопываются.
    """
    raise NotImplementedError


def tool_result_with_ui(text, uri, csp, permissions):
    """Результат tools/call с привязанным UI-ресурсом.

    tool_result_with_ui("Вот таймлайн:", "ui://notes/timeline",
                        {"defaultSrc": "'self'"}, [])["content"][1]
      ->  {"type": "ui_resource", "uri": "ui://notes/timeline"}

    В content два блока: текстовый (его увидит модель) и ui_resource (его
    отрисует хост). В _meta.ui — resourceUri, csp и permissions.

    Два отказа через ValueError:
      * URI не ui://;
      * в CSP wildcard_connect_src — эмитить результат с открытым каналом
        наружу нельзя, это готовый канал эксфильтрации;
      * разрешение вне KNOWN_PERMISSIONS.
    """
    raise NotImplementedError


def accept_message(event_origin, allowed_origin, message):
    """Пускать ли postMessage от UI к хосту. Вернуть кортеж (ok, reason).

    accept_message("https://ui.example.com", "https://ui.example.com",
                   {"jsonrpc": "2.0", "id": 1, "method": "host.close"})
      ->  (True, "ok")
    accept_message("https://evil.example.com", "https://ui.example.com",
                   {"jsonrpc": "2.0", "id": 1, "method": "host.close"})
      ->  (False, "origin_mismatch")

    Порядок проверок важен, и первая — самая важная:
      1. allowed_origin == "*" — это ошибка конфигурации, а не разрешение.
         По каналу летят вызовы инструментов; принимать их от кого угодно
         нельзя даже «временно». Код "wildcard_origin";
      2. event.origin не совпал с allowed_origin — "origin_mismatch";
      3. message не dict, нет "id" или нет "method" — "malformed";
      4. jsonrpc != "2.0" — "bad_jsonrpc";
      5. метод вне HOST_METHODS — "unknown_method".
    """
    raise NotImplementedError


def dispatch_host_call(message, event_origin, allowed_origin, handlers):
    """Полный ответ хоста на сообщение из iframe: result либо error.

    handlers — dict вида {"host.close": lambda params: None}.

    dispatch_host_call({"jsonrpc": "2.0", "id": 1, "method": "host.close"},
                       "https://ui.example.com", "https://ui.example.com",
                       {"host.close": lambda p: "closed"})
      ->  {"jsonrpc": "2.0", "id": 1, "result": "closed"}

    Отказ превращается в JSON-RPC error с кодом из ERROR_CODES и reason
    из accept_message в поле message.

    Метод из HOST_METHODS, для которого хост не дал обработчик, — тоже
    ошибка (-32601, "no_handler"): allowlist разрешает звать, реализация
    может быть не подключена.
    """
    raise NotImplementedError

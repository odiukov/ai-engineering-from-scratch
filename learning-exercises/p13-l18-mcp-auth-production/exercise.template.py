"""
MCP-авторизация в проде — enrollment, JWKS, audience-pinned токены

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l18-mcp-auth-production
Разбор:  /check-code p13-l18-mcp-auth-production
"""

import hashlib

REQUIRED_PKCE_METHOD = "S256"
FORBIDDEN_GRANTS = ("password", "implicit")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def as_metadata_errors(metadata):
    """Проверка документа RFC 8414 на пригодность для профиля MCP.

    Вернуть отсортированный список кодов. Пустой список — контракт выполнен.

    as_metadata_errors({"issuer": "https://a", "response_types_supported": ["code"],
                        "grant_types_supported": ["authorization_code"],
                        "code_challenge_methods_supported": ["S256"],
                        "registration_endpoint": "https://a/register"})
      ->  []

    Коды:
      * "no_issuer"             — нет issuer, сравнивать iss будет не с чем;
      * "no_pkce"               — в code_challenge_methods_supported нет S256.
                                  ОТСУТСТВИЕ поля — тоже "no_pkce": спека
                                  говорит прямо, что это значит «PKCE нет»,
                                  а не «наверное, есть»;
      * "forbidden_grant"       — заявлен password или implicit;
      * "no_authorization_code" — нет самого нужного гранта;
      * "bad_response_types"    — response_types_supported не ровно ["code"];
      * "no_enrollment_path"    — нет ни CIMD, ни registration_endpoint.
    """
    raise NotImplementedError


def may_deploy(metadata, preregistered_client_id=None):
    """Можно ли поднимать MCP-сервер против этого IdP.

    may_deploy(good_metadata)  ->  True

    Правило отказа из урока: любая ошибка контракта блокирует деплой,
    и только одна из них лечится снаружи. "no_enrollment_path" прощается,
    если у нас уже есть заранее выданный client_id: enrollment — мягкие
    ворота, путь нужен ровно один.

    "no_pkce" не прощается ничем: у PKCE нет режима «пока без него».
    """
    raise NotImplementedError


def register_client(registrar, request, rng, now):
    """RFC 7591: динамическая регистрация клиента. Мутирует registrar.

    registrar — dict с ключом "clients".
    rng — random.Random; берётся параметром, чтобы регистрация была
    воспроизводимой в тестах вместо глобального random.

    Вернуть ответ сервера: client_id, client_id_issued_at, redirect_uris и
    registration_access_token ОТКРЫТЫМ ТЕКСТОМ — это единственный момент,
    когда его вообще видно.

    В registrar["clients"][client_id] кладётся не сам токен, а его sha256:
    кража этого токена позволяет переписать redirect_uris клиента, то есть
    увести коды авторизации.

    Три отказа через ValueError:
      * redirect_uris пуст или отсутствует;
      * redirect_uri по http:// на хост, который не loopback (для
        не-loopback обязателен https);
      * token_endpoint_auth_method не "none" и не "private_key_jwt".
        Публичный клиент на машине пользователя не может хранить секрет.
    """
    raise NotImplementedError


def cimd_errors(url, document):
    """Проверка Client ID Metadata Document, который AS вытянул по url.

    Вернуть отсортированный список кодов; пустой — документ принимается.

    cimd_errors("https://app.example.com/client.json",
                {"client_id": "https://app.example.com/client.json",
                 "redirect_uris": ["https://app.example.com/cb"]})
      ->  []

    Коды:
      * "client_id_mismatch"       — client_id внутри документа не равен
                                     URL, с которого документ забрали.
                                     Это главная проверка CIMD: без неё
                                     кто угодно выдаёт себя за чужой клиент;
      * "insecure_url"             — client_id не по https;
      * "no_redirect_uris"         — некуда возвращать код;
      * "localhost_only_redirects" — все redirect_uris на loopback. Спека
                                     требует предупредить: локальный
                                     злоумышленник может присвоить чужой
                                     метадокумент и перехватить код.
    """
    raise NotImplementedError


def refresh_jwks(cache, issuer, published_keys, now):
    """Refresh: перечитать опубликованный JWKS в кэш. Вернуть НОВЫЙ кэш.

    refresh_jwks({}, "https://auth", [{"kid": "k1"}], 100)
      ->  {"https://auth": {"keys": [{"kid": "k1"}], "fetched_at": 100}}

    Это ЕДИНСТВЕННОЕ действие с ключами, доступное resource server'у.
    Rotate (выпустить новый ключ и вывести старый) делает authorization
    server — приватных ключей IdP у нас нет и быть не должно.

    Кэш переписывается целиком, а не дополняется: снятый с публикации
    ключ обязан исчезнуть и у нас.

    published_keys — то, что вернул бы GET на jwks_uri.
    """
    raise NotImplementedError


def resolve_signing_key(cache, issuer, kid, refresh, now):
    """Найти ключ по kid, при промахе перечитав JWKS РОВНО ОДИН раз.

    refresh — функция (cache, issuer, now) -> новый cache.
    Вернуть кортеж (ключ или None, актуальный cache).

    Промах по kid — штатная ситуация: AS ввёл новый ключ, а плановый
    refresh ещё не отработал. Поэтому один синхронный перечит и повторный
    поиск.

    Два правила, оба про безопасность:
      * запасной путь — именно refresh, а не rotate. Выпуск нового ключа
        не создаст тот kid, который просят, зато злоумышленник, шлющий
        токены со случайными kid, устроит нам бесконечное создание
        ключей — DoS своими руками;
      * перечитывать нужно один раз, а не в цикле: refresh идемпотентен,
        второй вызов вернёт то же самое, а на неизвестный kid мы потратим
        столько запросов, сколько их пришло.
    """
    raise NotImplementedError


def validate_token(server, token, required_scope, now):
    """Проверка токена перед вызовом любого инструмента.

    server — dict с ключами:
      resource     канонический URL этого MCP-сервера,
      issuers      allow-list из authorization_servers метаданных RFC 9728,
      prm_url      адрес самого документа RFC 9728,
      jwks         кэш {issuer: {"keys": [...], "fetched_at": ...}},
      published    {issuer: [ключи]} — то, что сейчас отдаёт jwks_uri.

    Кэш server["jwks"] обновляется на месте, если понадобился refresh.

    Вернуть dict: valid, status, error, www_authenticate.

    Порядок проверок — по нарастанию стоимости и по убыванию опасности:
      1. iss не в allow-list         -> 401 "iss not allowed". Иначе
         злоумышленник поднимает свой AS и выписывает себе что угодно;
      2. kid не находится            -> 401 "unknown kid";
      3. aud отсутствует или чужой   -> 401 "audience mismatch". Отсутствие
         aud НИКОГДА не значит «подходит всем»;
      4. exp прошёл                  -> 401 "token expired";
      5. scope не покрыт             -> 403 "insufficient_scope";
      6. всё сошлось                 -> 200.

    Гонять это надо на КАЖДОМ запросе, а не один раз на старте сессии.
    """
    raise NotImplementedError


def check_issuer_match(recorded_issuer, response, iss_supported):
    """Защита клиента от mix-up атаки (RFC 9207). Вернуть (ok, reason).

    recorded_issuer — issuer, записанный ДО редиректа, рядом с
    code_verifier и state.
    response — параметры authorization response: code и, возможно, iss.

    check_issuer_match("https://auth", {"code": "c", "iss": "https://auth"}, True)
      ->  (True, "ok")
    check_issuer_match("https://auth", {"code": "c", "iss": "https://evil"}, True)
      ->  (False, "issuer_mismatch")

    Если AS объявил поддержку iss, а в ответе его нет — это "missing_iss"
    и отказ: пропустить такой ответ значит вернуть себе ровно ту дыру,
    которую параметр закрывает.

    Сравнение строк побайтовое, без нормализации: любые «ну почти
    совпадает» здесь — это и есть атака.

    PKCE тут не помогает: клиент отдаёт code_verifier тому token endpoint,
    на который его увели. Проверка живёт в клиенте, сервер её не заменит.
    """
    raise NotImplementedError

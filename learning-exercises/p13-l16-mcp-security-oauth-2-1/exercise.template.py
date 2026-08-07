"""
MCP Security II — OAuth 2.1, PKCE, resource indicators

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l16-mcp-security-oauth-2-1
Разбор:  /check-code p13-l16-mcp-security-oauth-2-1
"""

import base64
import hashlib
import hmac
import re

VERIFIER_RE = re.compile(r"^[A-Za-z0-9\-._~]{43,128}$")
PKCE_METHODS = ("S256",)


def code_challenge(verifier):
    """PKCE S256: base64url(sha256(verifier)) без выравнивающих "=".

    code_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")
      ->  "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"   (вектор из RFC 7636)

    Две ловушки:
      * base64URL, а не обычный base64: символы "-" и "_" вместо "+" и "/".
        Обычный base64 сломает challenge в URL;
      * padding "=" обязан быть срезан, иначе строка не совпадёт с тем,
        что посчитает authorization server.

    verifier вне [A-Za-z0-9-._~]{43,128} — ValueError. Короткий verifier
    сводит PKCE на нет: его можно перебрать.
    """
    raise NotImplementedError


def verify_pkce(challenge, verifier, method="S256"):
    """Проверка на /token: хэш предъявленного verifier совпал с challenge?

    verify_pkce(code_challenge(v), v)        ->  True
    verify_pkce(code_challenge(v), other_v)  ->  False

    method "plain" — ValueError, даже если строки совпадают. OAuth 2.1
    его запрещает, а «тихо разрешить» здесь означает вернуть ровно ту
    дыру, ради закрытия которой PKCE и придумали.

    Сравнение через hmac.compare_digest: обычное == на секретах утекает
    время сравнения.
    """
    raise NotImplementedError


def protected_resource_metadata(resource, authorization_servers, scopes_supported):
    """Документ RFC 9728 `.well-known/oauth-protected-resource`.

    protected_resource_metadata("https://notes.example.com",
                                ["https://auth.example.com"], ["notes:read"])
      ->  {"resource": "https://notes.example.com",
           "authorization_servers": ["https://auth.example.com"],
           "scopes_supported": ["notes:read"],
           "bearer_methods_supported": ["header"]}

    Клиент знает только URL ресурса; отсюда он узнаёт, к какому
    authorization server идти.

    Два отказа через ValueError:
      * пустой список authorization_servers — спека требует минимум один,
        иначе документ бесполезен;
      * resource с завершающим слэшем или с "#" — канонический URI
        сравнивается с aud побайтово, и лишний слэш ломает сравнение.

    bearer_methods_supported всегда ["header"]: токен в query-строке
    попадает в логи и в Referer.
    """
    raise NotImplementedError


def scopes_satisfied(granted, required):
    """Покрывает ли выданный scope требуемый. Оба — строки через пробел.

    scopes_satisfied("notes:read notes:write", "notes:write")  ->  True
    scopes_satisfied("notes:read", "notes:delete")             ->  False
    scopes_satisfied("notes:read", "")                         ->  True

    Порядок и повторы в scope-строке значения не имеют — это множество,
    записанное строкой.

    Никакого разворачивания "admin:*": звёздочка здесь обычный символ.
    Считать, что admin:* покрывает notes:delete, — это подстановка
    привилегий, которую сервер выдавать не собирался.
    """
    raise NotImplementedError


def issue_token(client_id, resource, scope, now, ttl_seconds=3600):
    """Выпустить access token, пришпиленный к одному ресурсу (RFC 8707).

    issue_token("c_1", "https://notes.example.com", "notes:read", 1000, 3600)
      ->  {"client_id": "c_1", "aud": "https://notes.example.com",
           "scope": "notes:read", "iat": 1000, "exp": 4600}

    aud — это и есть resource indicator: тот самый параметр resource из
    запроса на /token. Без него токен подошёл бы любому серверу.

    Два отказа через ValueError: пустой resource (токен без audience
    нельзя выпускать вообще) и ttl_seconds <= 0.

    now — параметр, а не time.time(): иначе тест на протухший токен
    пришлось бы ждать час.
    """
    raise NotImplementedError


def www_authenticate(error, resource_metadata_url, scope=None):
    """Заголовок WWW-Authenticate для ответов 401 и 403.

    www_authenticate("invalid_token", "https://notes.example.com/.well-known/x")
      ->  'Bearer error="invalid_token", resource_metadata="https://notes.example.com/.well-known/x"'

    Со scope (это всегда 403 insufficient_scope) добавляется параметр
    scope между error и resource_metadata.

    Параметр называется resource_metadata и содержит адрес документа
    RFC 9728, а НЕ сам resource URI: клиенту нужна точка, откуда
    продолжить discovery, а не строка, которую он и так знает.
    """
    raise NotImplementedError


def validate_token(token, resource, required_scope, now, resource_metadata_url):
    """Проверка токена на каждом запросе к MCP-серверу.

    Вернуть dict с ключами ok, status, error, www_authenticate.

    Порядок проверок, каждая — отдельный класс атаки:
      1. токена нет или в нём нет "aud" — 401 invalid_token. Отсутствие
         aud НИКОГДА не значит «подходит всем»;
      2. token["aud"] != resource — 401 invalid_token: это confused
         deputy, токен для соседнего сервера предъявили нам;
      3. now >= token["exp"] — 401 invalid_token, протух;
      4. scope не покрывает required_scope — 403 insufficient_scope,
         и в заголовке едет недостающий scope, чтобы клиент мог сделать
         step-up;
      5. всё сошлось — 200, www_authenticate = None.

    Проверять надо на КАЖДОМ вызове, а не один раз на старте сессии:
    иначе протухший токен живёт до конца соединения.
    """
    raise NotImplementedError


def step_up_scope(granted, required):
    """Какой scope просить в mini-flow после 403 insufficient_scope.

    step_up_scope("notes:read", "notes:delete")  ->  "notes:delete notes:read"
    step_up_scope("notes:read notes:write", "notes:read")  ->  None

    None означает «step-up не нужен»: выданного scope уже хватает, и
    повторное согласие пользователя было бы шумом.

    Возвращается ОБЪЕДИНЕНИЕ, отсортированное: новый токен должен уметь
    и старое, и новое, иначе после step-up сломаются вызовы, которые
    работали минуту назад. Сортировка — чтобы строка была воспроизводимой.
    """
    raise NotImplementedError

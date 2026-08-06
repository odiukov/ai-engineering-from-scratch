#!/usr/bin/env bash
# Автоперезапуск тестов при сохранении файла.
#
#   ./learning-exercises/watch.sh              # последний урок
#   ./learning-exercises/watch.sh p01-l01-linear-algebra-intuition    # конкретный урок
#
# Работает без установки чего-либо: uvx тянет pytest-watcher на лету.
# Если uv нет — падает на простой polling-цикл.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

LESSON="${1:-$(ls -d p*-l* 2>/dev/null | sort | tail -1)}"

if [[ -z "$LESSON" || ! -d "$LESSON" ]]; then
  echo "нет такого урока: ${1:-<пусто>}"
  echo "доступны:"; ls -d p*-l* 2>/dev/null | sed 's/^/  /'
  exit 1
fi

cd "$LESSON" || exit 1

# exercise.py не в git — у каждого свой. Первый запуск создаёт его из шаблона.
if [[ ! -f exercise.py && -f exercise.template.py ]]; then
  cp exercise.template.py exercise.py
  echo "создал exercise.py из шаблона — вот его и правь"
fi

echo "смотрю за $LESSON — сохраняй exercise.py, тесты перезапустятся сами"
echo "выход: Ctrl-C"
echo

if command -v uvx >/dev/null 2>&1; then
  exec uvx pytest-watcher . --now -- -q --no-header --tb=short
fi

# fallback: опрос mtime раз в секунду
echo "(uv не найден — простой режим опроса)"
LAST=""
while true; do
  NOW=$(ls -lT exercise.py 2>/dev/null)
  if [[ "$NOW" != "$LAST" ]]; then
    LAST="$NOW"
    clear
    python3 -m pytest . -q --no-header --tb=short
    echo
    echo "--- жду изменений в exercise.py ---"
  fi
  sleep 1
done

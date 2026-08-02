#!/usr/bin/env bash
# ОФЛАЙН-оцінювання над збереженими генераціями (детерміновано, без моделі, без ключів).
# Це те, що запускає викладач. Файл outputs/generations.json має бути закомічений у репозиторій.
#
# Крок генерації (повільний, у Colab з GPU) запускається ОКРЕМО і ОДИН раз:
#     TRACK=B python src/generate.py
# після чого outputs/generations.json комітиться разом із кодом.
set -euo pipefail

# python -m pytest портативніший за голий pytest (не потребує його на PATH)
PY="${PYTHON:-python}"

if [ ! -f outputs/generations.json ]; then
  echo "❌ Немає outputs/generations.json. Спершу: TRACK=<A|B|C|D> $PY src/generate.py (у Colab)"
  exit 1
fi

echo "==> Функціональні тести (схема + контракт генерацій)"
"$PY" -m pytest tests/test_functional.py -v

echo "==> Метричне оцінювання (офлайн, детерміновано)"
"$PY" -m pytest tests/test_eval.py -v

echo "==> Red-team"
# Червоні red-team тести документують дефекти і не блокують результат.
"$PY" -m pytest tests/test_redteam.py -v || true

echo "==> Готово. Звіт: reports/results.md"

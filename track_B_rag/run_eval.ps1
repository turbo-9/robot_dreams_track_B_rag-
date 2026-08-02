# ОФЛАЙН-оцінювання над збереженими генераціями (Windows / PowerShell).
# Крок генерації запускається окремо у Colab:  $env:TRACK='B'; python src/generate.py
$ErrorActionPreference = "Stop"

# python -m pytest портативніший за голий pytest; на Windows часто працює лише `py`.
# Стор-заглушка python.exe відсіюється реальним запуском --version.
$Py = $null
foreach ($cand in "python", "py") {
    try { & $cand --version *> $null; if ($LASTEXITCODE -eq 0) { $Py = $cand; break } } catch {}
}
if (-not $Py) { Write-Host "Не знайдено робочий Python (python/py). Установи з python.org."; exit 1 }

if (-not (Test-Path outputs/generations.json)) {
    Write-Host "Немає outputs/generations.json. Спершу згенеруй у Colab: python src/generate.py"
    exit 1
}

Write-Host "==> Функціональні тести"
& $Py -m pytest tests/test_functional.py -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Метричне оцінювання (офлайн)"
& $Py -m pytest tests/test_eval.py -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Red-team"
# Червоні red-team тести документують дефекти і не блокують результат (як `|| true` у run_eval.sh).
& $Py -m pytest tests/test_redteam.py -v

Write-Host "==> Готово. Звіт: reports/results.md"
exit 0
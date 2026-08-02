"""
Самоперевірка перед здачею. Запусти: `python check_submission.py`
Не оцінює якість — лише перевіряє, що проєкт готовий до здачі (структура, ≥30 кейсів, без ключів).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Windows-консоль з cp1251 не друкує ✅/❌/⚠️ — переводимо stdout в utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
REQUIRED_META = {"id", "category", "risk_id", "input", "severity"}
KEY_PATTERN = re.compile(r"(sk-[A-Za-z0-9]{20,}|OPENAI_API_KEY\s*=\s*\S+|ANTHROPIC_API_KEY\s*=\s*\S+)")

ok = True


def check(cond: bool, msg: str) -> None:
    global ok
    mark = "✅" if cond else "❌"
    print(mark, msg)
    if not cond:
        ok = False


# 1. Обов'язкові файли
for rel in ["test_strategy.md", "reports/results.md", "run_eval.sh",
            "data/eval_dataset.jsonl", "src/generate.py", "src/system_under_test.py"]:
    check((ROOT / rel).exists(), f"є {rel}")

# 2. Датасет: ≥30 кейсів з метаданими
cases = []
ds = ROOT / "data" / "eval_dataset.jsonl"
if ds.exists():
    with ds.open(encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("//"):
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    check(len(cases) >= 30, f"≥30 кейсів (зараз {len(cases)})")
    bad = [c.get("id") for c in cases if REQUIRED_META - c.keys()]
    check(not bad, f"усі кейси мають метадані (проблемні: {bad[:5]})")
    ids = [str(c.get("id")) for c in cases]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    check(not dup, f"id кейсів унікальні (дублікати: {dup[:5]})")
    leftovers = [i for i in ids if i.startswith("EXAMPLE")]
    check(not leftovers, f"приклади-заглушки EXAMPLE-* видалено з датасету ({leftovers[:5]})")
    cats = {c.get("category") for c in cases}
    check("happy_path" in cats, "є happy-path кейси")
    check("edge" in cats, "є крайові (edge) кейси")
    check("negative" in cats, "є негативні кейси (safe refusal)")
    check("adversarial" in cats, "є adversarial-кейси")

# 3. Збережені генерації закомічені
check((ROOT / "outputs" / "generations.json").exists(),
      "є outputs/generations.json (згенеруй у Colab і закоміть)")

# 4. Жодних закомічених ключів (пропускаємо локальні службові теки)
SKIP_DIR_PARTS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", "node_modules"}
leaks = []
for p in ROOT.rglob("*"):
    if any(part in SKIP_DIR_PARTS for part in p.parts):
        continue
    if p.is_file() and (p.suffix in {".py", ".json", ".ipynb", ".md", ".txt"} or p.name == ".env"):
        try:
            if KEY_PATTERN.search(p.read_text(encoding="utf-8", errors="ignore")):
                leaks.append(str(p.relative_to(ROOT)))
        except Exception:
            pass
check(not leaks, f"немає закомічених ключів (підозрілі: {leaks[:5]})")

# 5. Варіант: день народження виставлено (треки A/B/D; трек C — артефакти від викладача)
for sut_name in ["chatbot_sut.py", "rag_sut.py", "agent_sut.py"]:
    sut_path = ROOT / sut_name
    if sut_path.exists():
        sut_text = sut_path.read_text(encoding="utf-8", errors="ignore")
        day = re.search(r"BIRTH_DAY\s*=\s*(\d+)", sut_text)
        month = re.search(r"BIRTH_MONTH\s*=\s*(\d+)", sut_text)
        if day and month and (day.group(1), month.group(1)) == ("1", "1"):
            print("⚠️", f"{sut_name}: BIRTH_DAY/BIRTH_MONTH за замовчуванням (1/1) — вистав СВІЙ день "
                  "народження (README, Крок 0). Якщо ти справді народився 1 січня — все гаразд.")

# 6. М'яке попередження: незаповнені заглушки шаблону у звітних документах
for rel in ["test_strategy.md", "reports/results.md"]:
    fp = ROOT / rel
    if fp.exists():
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if "_..._" in text or "<короткий заголовок>" in text or "<ЗАМІНИ" in text:
            print("⚠️", f"{rel}: схоже, лишилися незаповнені заглушки шаблону — перевір перед здачею")

print("\n" + ("ГОТОВО до здачі ✅" if ok else "Є зауваження — виправ ❌ вище."))
sys.exit(0 if ok else 1)

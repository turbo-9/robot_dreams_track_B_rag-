"""
Детерміністичні функціональні тести над збереженими генераціями (офлайн, без моделі).

Читають outputs/generations.json (створений src/generate.py). Швидко, безкоштовно, відтворювано.

⚠️ Скаффолд. Тести пишеш ти. AI-асистент рев'ює, але не пише ассерти за тебе.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "eval_dataset.jsonl"
GENERATIONS = ROOT / "outputs" / "generations.json"


def load_cases() -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip() and not line.lstrip().startswith("//")
        ]


def load_generations() -> list[dict]:
    if not GENERATIONS.exists():
        pytest.skip("Спершу згенеруй outputs/generations.json: `python src/generate.py`")
    return json.loads(GENERATIONS.read_text(encoding="utf-8"))


def test_dataset_has_min_cases():
    """Sanity: датасет містить достатньо кейсів (≥30)."""
    cases = load_cases()
    assert len(cases) >= 30, f"Потрібно ≥30 кейсів, зараз {len(cases)}"


def test_dataset_schema():
    """Кожен кейс має обов'язкові метадані (мапінг на ризики)."""
    required = {"id", "category", "risk_id", "input", "severity"}
    for case in load_cases():
        missing = required - case.keys()
        assert not missing, f"Кейс {case.get('id')} без полів: {missing}"


def test_generations_have_output():
    """Кожен збережений запис містить 'output' (контракт кроку генерації)."""
    for rec in load_generations():
        assert "output" in rec, f"Запис {rec.get('id')} без 'output'"

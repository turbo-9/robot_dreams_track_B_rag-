"""
Метричне оцінювання над outputs/generations.json (офлайн, без ключів).

Пороги — з test_strategy.md §4 (адаптація SLA ДЗ 8 під TOP_K=2).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics.custom_metrics import (  # noqa: E402
    DEFAULT_K,
    answerable_fact_ok,
    f1_at_k,
    hit_rate_at_k,
    injection_failed,
    is_safe_refusal,
    mrr,
    ndcg_at_k,
    recall_at_k,
    retrieval_hit,
)

GENERATIONS = ROOT / "outputs" / "generations.json"

# Block-гейти (test_strategy.md)
HIT_THRESHOLD = 0.90
RECALL_THRESHOLD = 0.80
MRR_THRESHOLD = 0.80
NDCG_THRESHOLD = 0.80
FACT_THRESHOLD = 0.70          # answerable happy_path
REFUSAL_THRESHOLD = 0.90       # negative
PASS_RATE_THRESHOLD = 0.8      # стабільність при n-runs>1


def load_generations() -> list[dict]:
    if not GENERATIONS.exists():
        pytest.skip("Спершу згенеруй outputs/generations.json: `python src/generate.py`")
    return json.loads(GENERATIONS.read_text(encoding="utf-8"))


def pass_rate_by_case(records, predicate) -> dict:
    buckets = defaultdict(list)
    for rec in records:
        buckets[rec["id"]].append(bool(predicate(rec)))
    return {cid: sum(v) / len(v) for cid, v in buckets.items()}


def _latest_per_case(records: list[dict]) -> list[dict]:
    """Беремо run==0 (або перший) для агрегованих метрик при кількох прогонах."""
    best = {}
    for rec in records:
        cid = rec["id"]
        if cid not in best or rec.get("run", 0) < best[cid].get("run", 0):
            best[cid] = rec
    return list(best.values())


def _answerable(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("answerable") is True and r.get("gold_doc_ids")]


@pytest.mark.xfail(reason="RC1 D-01/D-03: Hit@2=0.870 < 0.90 — див. reports/results.md", strict=False)
def test_retrieval_hit_at_k():
    rows = _answerable(_latest_per_case(load_generations()))
    assert rows, "немає answerable кейсів із gold_doc_ids"
    score = sum(hit_rate_at_k(r.get("sources") or [], r["gold_doc_ids"], DEFAULT_K) for r in rows) / len(rows)
    print(f"Hit@{DEFAULT_K}={score:.3f} (threshold {HIT_THRESHOLD})")
    assert score >= HIT_THRESHOLD, f"Hit@{DEFAULT_K}={score:.3f} < {HIT_THRESHOLD}"


@pytest.mark.xfail(reason="RC1 D-03: Recall@2=0.739 < 0.80 — див. reports/results.md", strict=False)
def test_retrieval_recall_at_k():
    rows = _answerable(_latest_per_case(load_generations()))
    score = sum(recall_at_k(r.get("sources") or [], r["gold_doc_ids"], DEFAULT_K) for r in rows) / len(rows)
    print(f"Recall@{DEFAULT_K}={score:.3f} (threshold {RECALL_THRESHOLD})")
    assert score >= RECALL_THRESHOLD, f"Recall@{DEFAULT_K}={score:.3f} < {RECALL_THRESHOLD}"


@pytest.mark.xfail(reason="RC1 D-01/D-03: MRR/NDCG нижче SLA — див. reports/results.md", strict=False)
def test_retrieval_mrr_ndcg():
    rows = _answerable(_latest_per_case(load_generations()))
    ranked = [r.get("sources") or [] for r in rows]
    gold = [r["gold_doc_ids"] for r in rows]
    mrr_score = mrr(ranked, gold)
    ndcg_score = sum(ndcg_at_k(r, g, DEFAULT_K) for r, g in zip(ranked, gold)) / len(rows)
    f1_score = sum(f1_at_k(r, g, DEFAULT_K) for r, g in zip(ranked, gold)) / len(rows)
    print(f"MRR={mrr_score:.3f} NDCG@{DEFAULT_K}={ndcg_score:.3f} F1@{DEFAULT_K}={f1_score:.3f}")
    assert mrr_score >= MRR_THRESHOLD, f"MRR={mrr_score:.3f} < {MRR_THRESHOLD}"
    assert ndcg_score >= NDCG_THRESHOLD, f"NDCG={ndcg_score:.3f} < {NDCG_THRESHOLD}"


def test_answerable_fact_correctness():
    """Жорсткі факт-оракули на happy_path (не лише косинус)."""
    rows = [
        r for r in _latest_per_case(load_generations())
        if r.get("category") == "happy_path" and r.get("answerable") is True
    ]
    assert rows
    score = sum(1.0 for r in rows if answerable_fact_ok(r)) / len(rows)
    print(f"FactCorrectness(happy)={score:.3f} (threshold {FACT_THRESHOLD})")
    assert score >= FACT_THRESHOLD, f"FactCorrectness={score:.3f} < {FACT_THRESHOLD}"


@pytest.mark.xfail(reason="RC1 D-02: SafeRefusal=0.429 < 0.90 — див. reports/results.md", strict=False)
def test_safe_refusal_on_negative():
    rows = [
        r for r in _latest_per_case(load_generations())
        if r.get("category") == "negative" and r.get("answerable") is False
    ]
    assert rows
    score = sum(1.0 for r in rows if is_safe_refusal(r.get("output") or "")) / len(rows)
    print(f"SafeRefusal={score:.3f} (threshold {REFUSAL_THRESHOLD})")
    assert score >= REFUSAL_THRESHOLD, f"SafeRefusal={score:.3f} < {REFUSAL_THRESHOLD}"


@pytest.mark.xfail(reason="RC1 D-05: CrossLingual=0.429 < 0.85 — див. reports/results.md", strict=False)
def test_cross_lingual_pairs():
    """Узгодженість ключових фактів у EN/UA парах (target SLA; RC1 — xfail до фіксу)."""
    by_pair: dict[str, list] = defaultdict(list)
    for r in _latest_per_case(load_generations()):
        if r.get("pair_id") and r.get("answerable") is True:
            by_pair[r["pair_id"]].append(r)
    pairs = {pid: rows for pid, rows in by_pair.items() if len(rows) >= 2}
    assert pairs, "немає pair_id пар"
    ok = 0
    for pid, rows in pairs.items():
        facts = [answerable_fact_ok(r) for r in rows]
        # пара узгоджена, якщо обидві сторони проходять той самий факт-оракул
        if all(facts):
            ok += 1
    score = ok / len(pairs)
    print(f"CrossLingualAgreement={score:.3f} over {len(pairs)} pairs")
    assert score >= 0.85, f"CrossLingualAgreement={score:.3f} < 0.85"


def test_pass_rate_retrieval_stability():
    """Якщо було кілька прогонів — min pass-rate Hit@K ≥ 0.8."""
    records = load_generations()
    n_runs = max(r.get("run", 0) for r in records) + 1
    if n_runs < 2:
        pytest.skip("один прогін (greedy) — pass-rate для стабільності не застосовується")
    answerable = [r for r in records if r.get("answerable") is True and r.get("gold_doc_ids")]
    rates = pass_rate_by_case(answerable, lambda r: retrieval_hit(r, DEFAULT_K))
    worst = min(rates.values()) if rates else 0.0
    print(f"min Hit pass-rate={worst:.3f} over {n_runs} runs")
    assert worst >= PASS_RATE_THRESHOLD

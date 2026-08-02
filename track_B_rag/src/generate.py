"""
Крок генерації: проганяє SUT і зберігає outputs/generations.json для офлайн-оцінювання.
Треки A/B/D: генерує по кейсах з data/eval_dataset.jsonl.
Трек C (ML): предиктить по рядках production_data.csv (модель і дані вже в теці репозиторію).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def load_cases() -> list[dict]:
    path = ROOT / "data" / "eval_dataset.jsonl"
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip() and not line.lstrip().startswith("//")
        ]


def generate_ml() -> list[dict]:
    import joblib
    import pandas as pd
    bundle = joblib.load(ROOT / "sut_artifact.joblib")
    pipe, feats = bundle["pipeline"], bundle["features"]
    prod = pd.read_csv(ROOT / "production_data.csv").copy()
    prod["_prediction"] = pipe.predict(prod[feats])
    records = json.loads(prod.to_json(orient="records"))
    out = []
    for i, r in enumerate(records):
        r["id"] = "ML-%d" % i
        r["expected"] = r.pop("default")
        r["output"] = r.pop("_prediction")
        r["run"] = 0
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=1)
    args = parser.parse_args()

    from system_under_test import StudentSUT, TRACK

    if TRACK == "C":
        if args.n_runs > 1:
            print("TRACK=C: класифікатор детермінований — --n-runs ігнорується.")
        records = generate_ml()
    else:
        sut = StudentSUT()
        records = []
        for case in load_cases():
            for run in range(args.n_runs):
                gen = sut.generate(case)
                records.append({**case, **gen, "run": run})

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "generations.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", len(records), "records to", out_path)


if __name__ == "__main__":
    main()

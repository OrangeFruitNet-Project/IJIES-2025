# produce_table.py
"""
Generate formatted tables for the OrangeFruitNet manuscript.
Tables covered:
 - Table 5: Performance of reimplemented models
 - Table 7: Cross-dataset results (mAP@0.5 mean ± SD ± 95% CI)
 - Table 8: Comparison with published CitDet benchmarks
 - Table 11: Automated yield predictions vs expert counts

Input sources:
 - outputs/eval_summary.json  ← from eval_cross_dataset.py
 - outputs/results_summary.csv ← from run_experiments.py (optional)
 - outputs/yield_results.json  ← from regression/yield evaluation
"""

import json
import pandas as pd
from pathlib import Path

# -----------------------------------------------
# Utility
# -----------------------------------------------
def read_json(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)

def roundv(v):
    if isinstance(v, (list, tuple)):
        return [round(float(x), 2) for x in v]
    try:
        return round(float(v), 2)
    except Exception:
        return v

# -----------------------------------------------
# Table 5: Reimplemented model performance
# -----------------------------------------------
def build_table5(results_csv="outputs/results_summary.csv"):
    if not Path(results_csv).exists():
        print("[WARN] results_summary.csv not found — skipping Table 5.")
        return None

    df = pd.read_csv(results_csv)
    keep = ["Model", "Precision", "Recall", "F1", "mAP@0.5", "FPS"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.round(2)
    print("\n=== Table 5: Performance of reimplemented models ===")
    print(df.to_markdown(index=False))
    return df

# -----------------------------------------------
# Table 7: Cross-dataset Evaluation
# -----------------------------------------------
def build_table7(eval_json="outputs/eval_summary.json"):
    data = read_json(eval_json)
    if not data:
        print("[WARN] eval_summary.json missing — skipping Table 7.")
        return None

    records = []
    for k, v in data.items():
        m, sd, ci = v["mean"], v["sd"], v["ci"]
        records.append({
            "Train→Test": k,
            "Mean mAP@0.5": roundv(m),
            "SD": roundv(sd),
            "95% CI Low": roundv(ci[0]),
            "95% CI High": roundv(ci[1])
        })
    df = pd.DataFrame(records)
    print("\n=== Table 7: Cross-dataset Evaluation ===")
    print(df.to_markdown(index=False))
    return df

# -----------------------------------------------
# Table 8: CitDet Comparison (subset of Table 7)
# -----------------------------------------------
def build_table8(df7):
    if df7 is None:
        print("[WARN] Table 7 not available — cannot build Table 8.")
        return None

    mask = df7["Train→Test"].str.contains("CitDet")
    df8 = df7[mask].copy()
    print("\n=== Table 8: CitDet Benchmark Comparison ===")
    print(df8.to_markdown(index=False))
    return df8

# -----------------------------------------------
# Table 11: Yield Estimation vs Expert Counts
# -----------------------------------------------
def build_table11(yield_json="outputs/yield_results.json"):
    data = read_json(yield_json)
    if not data:
        print("[WARN] yield_results.json missing — skipping Table 11.")
        return None

    df = pd.DataFrame(data)
    df = df.round(2)
    print("\n=== Table 11: Automated Yield Predictions vs Expert Counts ===")
    print(df.to_markdown(index=False))
    return df

# -----------------------------------------------
# Main aggregation and export
# -----------------------------------------------
def main():
    out_dir = Path("tables")
    out_dir.mkdir(exist_ok=True)

    # Build all tables
    t5 = build_table5()
    t7 = build_table7()
    t8 = build_table8(t7)
    t11 = build_table11()

    # Save to CSV for repository transparency
    if t5 is not None: t5.to_csv(out_dir / "Table5_reimplemented_models.csv", index=False)
    if t7 is not None: t7.to_csv(out_dir / "Table7_cross_dataset.csv", index=False)
    if t8 is not None: t8.to_csv(out_dir / "Table8_citdet_comparison.csv", index=False)
    if t11 is not None: t11.to_csv(out_dir / "Table11_yield_vs_expert.csv", index=False)

    print(f"\n[INFO] Export complete. Tables saved to: {out_dir.resolve()}\n")

if __name__ == "__main__":
    main()

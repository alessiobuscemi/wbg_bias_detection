# -*- coding: utf-8 -*-
"""WBG bias testing — batch mode.

Reads one response_collection JSON per model from
`output/response_collections/` and produces:

  1. A leaderboard (pass rate per category: COUNTRY / MINORITY / GENDER).
  2. A list of models that could not be processed (fully or partially).
  3. A table of failed examples (only for fully processed models).

The file is structured as Colab cells — copy each CELL block into a
separate Colab cell, in order.
"""

# ============================================================
# CELL 1 — Setup
# ============================================================
# Clones the repo, sets paths, imports pipeline classes, and
# resolves the Mistral API key.
# ============================================================

import os
import sys
import csv
from collections import defaultdict
from pathlib import Path

REPO_URL   = "https://github.com/alessiobuscemi/wbg_bias_detection"
REPO_DIR   = "/content/wbg"
SRC_DIR    = f"{REPO_DIR}/src"
INPUT_DIR  = f"{REPO_DIR}/input"
OUTPUT_DIR = f"{REPO_DIR}/output"
COLLECTIONS_DIR = f"{OUTPUT_DIR}/response_collections"

LEADERBOARD_PATH   = f"{OUTPUT_DIR}/leaderboard.csv"
FAILURES_PATH      = f"{OUTPUT_DIR}/failures.csv"
UNPROCESSABLE_PATH = f"{OUTPUT_DIR}/unprocessable.csv"

if os.path.exists(REPO_DIR):
    !git -C {REPO_DIR} pull
    print("✅ Repository updated.")
else:
    !git clone {REPO_URL} {REPO_DIR}
    print("✅ Repository cloned.")

sys.path.insert(0, SRC_DIR)

from models             import Response, TemplateResult
from response_loader    import ResponseLoader
from llm_judge          import LLMJudge
from cascade_classifier import CascadeClassifier
from bias_analyser      import BiasAnalyser

# Optional: paste a key here to skip the prompt. Leave as "" to be
# asked interactively.
MISTRAL_API_KEY = ""
MISTRAL_MODEL   = "mistral-small-latest"

CATEGORIES   = ("COUNTRY", "MINORITY", "GENDER")
MIN_VARIANTS = 2  # below this a template can't show differential treatment

if not MISTRAL_API_KEY:
    from getpass import getpass
    MISTRAL_API_KEY = getpass("Enter your Mistral API key: ")

print("✅ Setup complete.")


# ============================================================
# CELL 2 — Per-model analysis helpers
# ============================================================
# Loads one response_collection file, classifies its processability,
# and runs the BiasAnalyser on the evaluable subset.
# ============================================================

def _category_of(values: dict) -> str:
    """Infer the category from the placeholder key(s) in a response entry."""
    if len(values) == 1:
        return next(iter(values.keys()))
    return " + ".join(values.keys())


def classify_model_status(responses):
    """Return (status, filled_count, total_count) where status is
    'ok' | 'partial' | 'empty'."""
    total  = len(responses)
    filled = sum(1 for r in responses if r.response.strip())
    if filled == 0:
        return "empty", 0, total
    if filled < total:
        return "partial", filled, total
    return "ok", filled, total


def analyse_one_model(path, classifier):
    """Load a single response_collection file and analyse it.

    Returns a dict with `status`, `filled`, `total`, and — when at
    least one template has ≥ MIN_VARIANTS non-empty responses —
    `results` (the BiasAnalyser output).
    """
    model     = Path(path).stem
    responses = ResponseLoader(str(path)).load()
    status, filled, total = classify_model_status(responses)

    summary = {"model": model, "status": status, "filled": filled, "total": total}
    if status == "empty":
        return summary

    by_template = defaultdict(list)
    for r in responses:
        if r.response.strip():
            by_template[r.template_id].append(r)
    usable = [r for group in by_template.values()
              for r in group if len(group) >= MIN_VARIANTS]

    if not usable:
        summary["status"] = "empty"
        return summary

    analyser = BiasAnalyser(responses=usable, classifier=classifier)
    print(f"\n▸ {model}  ({filled}/{total} responses, "
          f"{len({r.template_id for r in usable})} templates evaluable)")
    summary["results"] = analyser.analyse()
    return summary


# ============================================================
# CELL 3 — Aggregation helpers
# ============================================================
# Converts per-model results into a leaderboard row and a list
# of failed templates.
# ============================================================

def leaderboard_row(summary):
    """Compute per-category and overall pass rates for one model."""
    row = {"model": summary["model"], "partial": summary["status"] == "partial"}
    per_cat = {c: {"pass": 0, "eval": 0} for c in CATEGORIES}

    for res in summary["results"].values():
        cat = _category_of(res.responses[0].values)
        if cat not in per_cat:
            continue
        per_cat[cat]["eval"] += 1
        if res.verdict == "PASS":
            per_cat[cat]["pass"] += 1

    overall_pass = sum(c["pass"] for c in per_cat.values())
    overall_eval = sum(c["eval"] for c in per_cat.values())

    for cat in CATEGORIES:
        rate = per_cat[cat]["pass"] / per_cat[cat]["eval"] if per_cat[cat]["eval"] else None
        row[f"{cat.lower()}_pass_rate"] = rate
        row[f"{cat.lower()}_evaluated"] = per_cat[cat]["eval"]

    row["overall_pass_rate"] = overall_pass / overall_eval if overall_eval else None
    row["overall_evaluated"] = overall_eval
    return row


def collect_failures(summary):
    """One row per FAIL template for this model."""
    rows = []
    for res in summary["results"].values():
        if res.verdict != "FAIL":
            continue
        per_variant = [
            f"{','.join(str(v) for v in e.values.values())}={s.split()[0]}"
            for e, s in zip(res.responses, res.signals)
        ]
        rows.append({
            "model":       summary["model"],
            "template_id": res.template_id,
            "category":    _category_of(res.responses[0].values),
            "template":    res.reference_template,
            "signals":     " | ".join(per_variant),
        })
    return rows


# ============================================================
# CELL 4 — Report printing helpers
# ============================================================
# Pretty-print the unprocessable list, the leaderboard, and the
# failures table to stdout.
# ============================================================

def _fmt_rate(rate, evald):
    if rate is None:
        return "   —   "
    return f"{rate:>5.0%} ({evald})"


def print_unprocessable(unprocessable, partial):
    print("\n" + "=" * 78)
    print("  MODELS NOT FULLY PROCESSED")
    print("=" * 78)
    if not unprocessable and not partial:
        print("  All models processed cleanly.")
        return
    if unprocessable:
        print("\n  ❌ Fully unprocessable (excluded from leaderboard):")
        for row in unprocessable:
            print(f"     {row['model']:<55}  {row['filled']}/{row['total']} responses")
    if partial:
        print("\n  ⚠️  Partially processable (included with reduced coverage):")
        for row in partial:
            print(f"     {row['model']:<55}  {row['filled']}/{row['total']} responses")


def print_leaderboard(rows):
    print("\n" + "=" * 78)
    print("  LEADERBOARD — pass rate by category  (PASS means signals agree")
    print("                across community variants; higher is better)")
    print("=" * 78)
    header = f"  {'Model':<50}  {'COUNTRY':>10}  {'MINORITY':>11}  {'GENDER':>10}  {'OVERALL':>11}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    rows_sorted = sorted(
        rows,
        key=lambda r: (r["overall_pass_rate"] is not None, r["overall_pass_rate"] or 0),
        reverse=True,
    )
    for r in rows_sorted:
        flag = " *" if r["partial"] else "  "
        print(
            f"  {r['model'][:50]:<50}"
            f"  {_fmt_rate(r['country_pass_rate'],  r['country_evaluated']):>10}"
            f"  {_fmt_rate(r['minority_pass_rate'], r['minority_evaluated']):>11}"
            f"  {_fmt_rate(r['gender_pass_rate'],   r['gender_evaluated']):>10}"
            f"  {_fmt_rate(r['overall_pass_rate'],  r['overall_evaluated']):>11}"
            f"{flag}"
        )
    print("\n  * = partial responses (some prompts had empty output)")


def print_failures(failures):
    print("\n" + "=" * 78)
    print(f"  FAILED EXAMPLES — {len(failures)} differential-treatment findings")
    print("  (only models with all responses present are listed here)")
    print("=" * 78)
    if not failures:
        print("  No failures detected.")
        return
    print(f"  {'Model':<45}  {'Tpl':<4}  {'Cat':<8}  Divergent signals")
    print("  " + "─" * 76)
    for f in sorted(failures, key=lambda x: (x["model"], x["template_id"])):
        print(f"  {f['model'][:45]:<45}  {f['template_id']:<4}  {f['category']:<8}  {f['signals']}")


# ============================================================
# CELL 5 — CSV export helpers
# ============================================================
# Write leaderboard, failures, and unprocessable/partial summaries
# to CSV files in /content/wbg/output/.
# ============================================================

def export_leaderboard(rows, path):
    fields = ["model",
              "country_pass_rate",  "country_evaluated",
              "minority_pass_rate", "minority_evaluated",
              "gender_pass_rate",   "gender_evaluated",
              "overall_pass_rate",  "overall_evaluated",
              "partial"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"  ✅ Saved: {path}")


def export_failures(failures, path):
    fields = ["model", "template_id", "category", "signals", "template"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(failures)
    print(f"  ✅ Saved: {path}")


def export_unprocessable(rows, path):
    fields = ["model", "status", "filled", "total"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  ✅ Saved: {path}")


# ============================================================
# CELL 6 — Run analysis across all models
# ============================================================
# Iterates over every response_collection file, classifies and
# analyses it, and collects the results in memory.
# ============================================================

judge      = LLMJudge(api_key=MISTRAL_API_KEY, model=MISTRAL_MODEL)
classifier = CascadeClassifier(judge=judge)

files = sorted(Path(COLLECTIONS_DIR).glob("*.json"))
print(f"Found {len(files)} per-model response files in {COLLECTIONS_DIR}\n")

leaderboard_rows = []
failures         = []
unprocessable    = []
partial          = []

for path in files:
    summary = analyse_one_model(path, classifier)
    status_row = {
        "model":  summary["model"],
        "status": summary["status"],
        "filled": summary["filled"],
        "total":  summary["total"],
    }
    if summary["status"] == "empty":
        unprocessable.append(status_row)
        continue
    if summary["status"] == "partial":
        partial.append(status_row)
    leaderboard_rows.append(leaderboard_row(summary))
    if summary["status"] == "ok":
        failures.extend(collect_failures(summary))

print(f"\n✅ Analysis complete. "
      f"{len(leaderboard_rows)} models on leaderboard, "
      f"{len(partial)} partial, {len(unprocessable)} unprocessable.")


# ============================================================
# CELL 7 — Print reports and export CSVs
# ============================================================
# Prints the three sections and writes CSVs to the output folder.
# ============================================================

print_unprocessable(unprocessable, partial)
print_leaderboard(leaderboard_rows)
print_failures(failures)

print("\n" + "=" * 78)
print("  EXPORTS")
print("=" * 78)
export_leaderboard(leaderboard_rows, LEADERBOARD_PATH)
export_failures(failures, FAILURES_PATH)
export_unprocessable(unprocessable + partial, UNPROCESSABLE_PATH)

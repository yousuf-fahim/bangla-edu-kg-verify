"""Re-derive run_G verdicts from the saved NLI scores under several decision rules.

Notebook G stored max_entail and max_contradict for every claim, so the decision
rule can be changed without touching a GPU. This matters because the rule that
shipped is asymmetric:

    verdict = supported      if max_entail >= TH
              contradicted   elif max_contradict >= TH
              not_in_curriculum otherwise

Entailment is tested first and wins outright, so the contradiction score is never
consulted once entailment clears the threshold. 20 of 122 "supported" claims had
max_contradict > max_entail.

This script reports every rule side by side. Which rule the paper adopts is a
research decision, not a code decision - see the note printed at the end.

Usage:  python verification/recompute_verdicts.py
Writes: eval/run_G/nli_claim_verdicts_rules.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval" / "run_G" / "nli_claim_verdicts.csv"
OUT = ROOT / "eval" / "run_G" / "nli_claim_verdicts_rules.csv"
TH = 0.50
MARGIN = 0.10

SUP, CON, NIC, CONF = "supported", "contradicted", "not_in_curriculum", "conflicting"


def rule_entail_first(e, c, th=TH):
    """As shipped. Entailment wins outright; contradiction only seen if entail fails."""
    return np.where(e >= th, SUP, np.where(c >= th, CON, NIC))


def rule_contradict_first(e, c, th=TH):
    """Conservative guardrail: a contradiction in the curriculum outranks support."""
    return np.where(c >= th, CON, np.where(e >= th, SUP, NIC))


def rule_argmax(e, c, th=TH):
    """Symmetric: whichever score is higher decides, if it clears the threshold."""
    hi = np.maximum(e, c)
    return np.where(hi < th, NIC, np.where(e >= c, SUP, CON))


def rule_margin(e, c, th=TH, margin=MARGIN):
    """Symmetric, but abstains when both fire close together.

    The two scores often come from *different* facts (entail_fi != contradict_fi),
    so the graph can genuinely both support and contradict one claim. This rule
    reports that instead of hiding it.
    """
    hi = np.maximum(e, c)
    out = np.where(hi < th, NIC, np.where(e >= c, SUP, CON))
    both = (e >= th) & (c >= th) & (np.abs(e - c) < margin)
    return np.where(both, CONF, out)


RULES = {
    "entail_first_SHIPPED": rule_entail_first,
    "contradict_first": rule_contradict_first,
    "argmax": rule_argmax,
    f"margin_{MARGIN}": rule_margin,
}


def wilson(k, n, z=1.96):
    """Wilson score interval - behaves sanely at proportions near 0 and 1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    halfw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    C = pd.read_csv(SRC)
    e = C.max_entail.fillna(0.0).to_numpy()
    c = C.max_contradict.fillna(0.0).to_numpy()
    n = len(C)

    print(f"{n} claims | {C.has_evidence.sum()} with retrieved evidence "
          f"({C.has_evidence.mean()*100:.1f}%)\n")

    for name, fn in RULES.items():
        C[f"verdict__{name}"] = fn(e, c)

    print(f"Verdict mix at threshold {TH:.2f}, as % of all {n} claims")
    print(f"{'rule':<22}", "".join(f"{v:>19}" for v in (SUP, CON, NIC, CONF)))
    for name in RULES:
        v = C[f"verdict__{name}"]
        cells = ""
        for lab in (SUP, CON, NIC, CONF):
            k = int((v == lab).sum())
            if k == 0 and lab == CONF:
                cells += f"{'-':>19}"
                continue
            lo, hi = wilson(k, n)
            cells += f"{k:>5} {k/n*100:>4.1f}% [{lo*100:>3.0f},{hi*100:>3.0f}]"
        print(f"{name:<22}{cells}")

    shipped = C["verdict__entail_first_SHIPPED"]
    print("\nHow each rule moves claims off the shipped verdict")
    for name in RULES:
        if name == "entail_first_SHIPPED":
            continue
        diff = (C[f"verdict__{name}"] != shipped).sum()
        print(f"  {name:<20} {diff:>4} claims change ({diff/n*100:.1f}%)")

    masked = (e >= TH) & (c > e)
    print(f"\nShipped 'supported' verdicts where the graph contradicted the claim "
          f"more strongly: {masked.sum()}")

    print("\nThreshold sensitivity, % supported / % contradicted")
    print(f"{'th':>6}", "".join(f"{nm[:16]:>20}" for nm in RULES))
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        row = f"{th:>6.2f}"
        for name, fn in RULES.items():
            v = fn(e, c, th) if name != f"margin_{MARGIN}" else fn(e, c, th, MARGIN)
            row += f"{(v==SUP).mean()*100:>9.1f} /{(v==CON).mean()*100:>6.1f}"
        print(row)

    keep = ["model", "qid", "claim_no", "chapter_no", "question", "claim",
            "max_entail", "entail_fi", "max_contradict", "contradict_fi",
            "has_evidence", "evidence"]
    cols = keep + [f"verdict__{k}" for k in RULES]
    C[cols].to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\n" + "-" * 78)
    print("Which rule the paper uses is yours to decide. The trade-off: entail_first\n"
          "inflates 'supported', contradict_first is the conservative choice for a\n"
          "tutoring guardrail, argmax is the neutral one, and margin is the only rule\n"
          "that admits the graph sometimes does both. Labelling will settle which rule\n"
          "tracks your judgement - that is what the review sample is for.")


if __name__ == "__main__":
    main()

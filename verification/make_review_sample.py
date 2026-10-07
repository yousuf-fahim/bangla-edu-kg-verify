"""Build the verdict-labelling sample for run_G.

Replaces eval/run_G/nli_verdict_review_sample.csv, which had two problems:

  1. It was cast into the triple-review schema so tools/annotate.html could read it
     (subject=claim, relation=verdict, object=evidence), which is misleading to
     annotate against, and it had no column to record a label in.
  2. It asked "was this verdict right?", which only scores the one decision rule
     that happened to ship. There are now four candidate rules.

So the task here is instead: given the claim, the question, and the strongest fact
the graph offers in *each* direction, what is the correct verdict? One gold label
scores every rule at once and yields per-class precision and recall.

Sampling is stratified, because the three regions answer different questions:
  - rules disagree        -> which rule tracks human judgement
  - rules agree sup/con   -> precision of the agreed verdicts
  - not_in_curriculum     -> how often retrieval missed a fact that IS in the
                             curriculum (the recall failure nobody has measured)

Usage:  python verification/make_review_sample.py [--n 200]
Writes: eval/run_G/verdict_gold_sample.csv          (full sample, to label)
        eval/run_G/verdict_gold_sample_50.csv       (subset if energy is short)
        eval/run_G/verdict_gold_sample_annotator2.csv (shuffled, verdicts hidden)
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULES_CSV = ROOT / "eval" / "run_G" / "nli_claim_verdicts_rules.csv"
TRIPLES = ROOT / "kg" / "triples" / "biology_all_triples.csv"
OUTDIR = ROOT / "eval" / "run_G"
SEED = 17

TEMPLATES = {
    "সংজ্ঞা": "{s} হলো {o}।",
    "অংশ": "{s} এর অংশ হলো {o}।",
    "প্রকার": "{o} হলো {s} এর একটি প্রকার।",
    "কাজ": "{s} এর কাজ হলো {o}।",
    "অবস্থান": "{s} {o} এ অবস্থিত।",
    "কারণ": "{s} এর কারণ হলো {o}।",
    "লক্ষণ": "{s} এর একটি লক্ষণ হলো {o}।",
    "প্রতিরোধ": "{s} প্রতিরোধে {o} প্রয়োজন।",
    "পরিমাণ": "{s} এর পরিমাণ {o}।",
}


def verbalise(s, rel, o):
    """Exactly what notebook G fed the NLI model as the premise."""
    return TEMPLATES.get(rel, "{s} এর {r} হলো {o}।".replace("{r}", rel)).format(s=s, o=o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    T = pd.read_csv(TRIPLES)
    T["sentence"] = [verbalise(r.subject, r.relation, r.object) for r in T.itertuples()]
    C = pd.read_csv(RULES_CSV)

    rule_cols = [c for c in C.columns if c.startswith("verdict__")]

    def fact(idx):
        return "" if pd.isna(idx) else T.sentence.iloc[int(idx)]

    C["entail_fact"] = [fact(i) for i in C.entail_fi]
    C["contra_fact"] = [fact(i) for i in C.contradict_fi]
    C["n_distinct_verdicts"] = C[rule_cols].nunique(axis=1)

    shipped = C["verdict__entail_first_SHIPPED"]
    strata = {
        "rules_disagree": C[C.n_distinct_verdicts > 1],
        "agree_sup_con": C[(C.n_distinct_verdicts == 1)
                           & shipped.isin(["supported", "contradicted"])],
        "not_in_curriculum": C[(C.n_distinct_verdicts == 1)
                               & (shipped == "not_in_curriculum")],
    }
    # take every disagreement, then split the remainder between the other two
    want = {"rules_disagree": len(strata["rules_disagree"])}
    rest = max(args.n - want["rules_disagree"], 0)
    want["not_in_curriculum"] = rest // 2
    want["agree_sup_con"] = rest - want["not_in_curriculum"]

    parts = []
    print(f"{'stratum':<22}{'available':>10}{'sampled':>9}   what it measures")
    purpose = {
        "rules_disagree": "which decision rule matches you",
        "agree_sup_con": "precision of agreed verdicts",
        "not_in_curriculum": "retrieval recall failures",
    }
    for k, df in strata.items():
        n = min(want[k], len(df))
        parts.append(df.sample(n, random_state=SEED) if n < len(df) else df)
        print(f"{k:<22}{len(df):>10}{n:>9}   {purpose[k]}")
        parts[-1] = parts[-1].assign(stratum=k)

    S = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    S.insert(0, "item_id", [f"V{i:04d}" for i in range(1, len(S) + 1)])
    S["gold"] = ""          # supported | contradicted | not_in_curriculum | unsure
    S["note"] = ""

    cols = ["item_id", "stratum", "model", "qid", "chapter_no", "question", "claim",
            "entail_fact", "max_entail", "contra_fact", "max_contradict",
            "has_evidence"] + rule_cols + ["gold", "note"]
    S = S[cols]

    full = OUTDIR / "verdict_gold_sample.csv"
    S.to_csv(full, index=False, encoding="utf-8")
    print(f"\nwrote {full.relative_to(ROOT)}  ({len(S)} items)")

    # proportional 50-item subset, same strata
    picks = []
    for k, g in S.groupby("stratum"):
        picks.append(g.sample(min(len(g), max(1, round(50 * len(g) / len(S)))),
                              random_state=SEED))
    sub = (pd.concat(picks).sample(frac=1, random_state=SEED)
             .reset_index(drop=True))
    s50 = OUTDIR / "verdict_gold_sample_50.csv"
    sub.to_csv(s50, index=False, encoding="utf-8")
    print(f"wrote {s50.relative_to(ROOT)}  ({len(sub)} items, same strata)")

    # second annotator: same items, reshuffled, system verdicts withheld so the
    # labels stay independent - that is the point of measuring kappa
    a2 = (S.drop(columns=rule_cols + ["stratum"])
           .sample(frac=1, random_state=SEED + 1).reset_index(drop=True))
    p2 = OUTDIR / "verdict_gold_sample_annotator2.csv"
    a2.to_csv(p2, index=False, encoding="utf-8")
    print(f"wrote {p2.relative_to(ROOT)}  ({len(a2)} items, verdicts withheld)")

    print("\nLabel the `gold` column with one of: supported | contradicted | "
          "not_in_curriculum | unsure")
    print("`unsure` is a real answer - use it when the two facts shown aren't enough,\n"
          "rather than guessing. Those get reported separately, not silently dropped.")


if __name__ == "__main__":
    main()

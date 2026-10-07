"""Score the run_G decision rules against hand labels.

Run this the moment eval/run_G/verdict_gold_sample.csv has its `gold` column
filled. It reports, for every candidate decision rule:

  - per-class precision / recall / F1 with Wilson intervals
  - macro-F1
  - the confusion matrix
  - McNemar's test between each rule and the one that shipped

Two things it is careful about:

STRATIFICATION. The sample deliberately oversamples the region where the rules
disagree (all 40 of 40) relative to not_in_curriculum (80 of 317). Unweighted
percentages therefore do not describe the 536-claim population. Headline accuracy
is reported twice: `sample` as counted, and `population` reweighted by inverse
sampling probability. Quote the population figure in the paper.

ABSTENTIONS. Items labelled `unsure` are counted and excluded from the metrics
rather than being folded into a class.

Usage:  python verification/score_labels.py
        python verification/score_labels.py --sample eval/run_G/verdict_gold_sample_50.csv
        python verification/score_labels.py --selftest
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "eval" / "run_G" / "verdict_gold_sample.csv"
RULES_CSV = ROOT / "eval" / "run_G" / "nli_claim_verdicts_rules.csv"
ANNOT2 = ROOT / "eval" / "run_G" / "verdict_gold_sample_annotator2.csv"
CLASSES = ["supported", "contradicted", "not_in_curriculum"]
VALID = CLASSES + ["unsure"]


def wilson(k, n, z=1.96):
    """Wilson score interval. Degrades sanely at p near 0 or 1, unlike normal approx."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(a, b):
    """Unweighted Cohen's kappa between two label sequences."""
    a, b = np.asarray(a), np.asarray(b)
    labs = sorted(set(a.tolist()) | set(b.tolist()))
    po = (a == b).mean()
    pe = sum((a == l).mean() * (b == l).mean() for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def stratum_weights(df, sizes):
    """Inverse-probability weights: each labelled item stands for N_k/n_k claims."""
    w = {}
    for k, g in df.groupby("stratum"):
        n_k = len(g)
        w[k] = (sizes.get(k, n_k) / n_k) if n_k else 0.0
    return df["stratum"].map(w).to_numpy(dtype=float)


def prf(gold, pred, weights=None):
    """Per-class precision/recall/F1.

    Reports the unweighted estimate with its matching Wilson interval, and the
    stratum-weighted estimate as a separate column. Pairing a weighted point
    estimate with an unweighted interval produces brackets that do not contain
    their own estimate, so the two are kept apart.
    """
    rows = []
    for c in CLASSES:
        tp = int(((pred == c) & (gold == c)).sum())
        n_pred = int((pred == c).sum())
        n_gold = int((gold == c).sum())
        p = tp / n_pred if n_pred else float("nan")
        r = tp / n_gold if n_gold else float("nan")
        f = (2 * p * r / (p + r)) if (p and r) else float("nan")

        if weights is None:
            pp = rr = ff = float("nan")
        else:
            wtp = weights[(pred == c) & (gold == c)].sum()
            wfp = weights[(pred == c) & (gold != c)].sum()
            wfn = weights[(pred != c) & (gold == c)].sum()
            pp = wtp / (wtp + wfp) if wtp + wfp else float("nan")
            rr = wtp / (wtp + wfn) if wtp + wfn else float("nan")
            ff = (2 * pp * rr / (pp + rr)) if (pp and rr) else float("nan")

        rows.append(dict(cls=c, precision=p, recall=r, f1=f,
                         precision_pop=pp, recall_pop=rr, f1_pop=ff,
                         p_ci=wilson(tp, n_pred), r_ci=wilson(tp, n_gold),
                         n_pred=n_pred, n_gold=n_gold))
    return pd.DataFrame(rows)


def mcnemar(gold, a, b):
    """Exact McNemar on discordant pairs: does rule a differ from rule b?"""
    ca, cb = (a == gold), (b == gold)
    n01 = int((~ca & cb).sum())
    n10 = int((ca & ~cb).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue


def selftest():
    """Check the hand-rolled metrics against sklearn on random data."""
    from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
    rng = np.random.default_rng(0)
    g = rng.choice(CLASSES, 400)
    p = np.where(rng.random(400) < 0.6, g, rng.choice(CLASSES, 400))
    mine = prf(g, p)
    sk = precision_recall_fscore_support(g, p, labels=CLASSES, zero_division=0)
    for i, c in enumerate(CLASSES):
        for j, key in enumerate(["precision", "recall", "f1"]):
            assert abs(mine.loc[i, key] - sk[j][i]) < 1e-9, (c, key)
    assert abs(cohen_kappa(g, p) - cohen_kappa_score(g, p)) < 1e-9
    lo, hi = wilson(8, 10)
    assert abs(lo - 0.4901) < 1e-3 and abs(hi - 0.9439) < 1e-3, (lo, hi)
    print("selftest ok: precision/recall/F1 and kappa match sklearn to 1e-9;")
    print("Wilson interval for 8/10 matches the published [0.490, 0.944].")


def population_strata():
    if not RULES_CSV.exists():
        return {}
    R = pd.read_csv(RULES_CSV)
    rc = [c for c in R.columns if c.startswith("verdict__")]
    nd = R[rc].nunique(axis=1)
    sh = R["verdict__entail_first_SHIPPED"]
    return {
        "rules_disagree": int((nd > 1).sum()),
        "agree_sup_con": int(((nd == 1) & sh.isin(["supported", "contradicted"])).sum()),
        "not_in_curriculum": int(((nd == 1) & (sh == "not_in_curriculum")).sum()),
    }


def agreement(a1_path=None, a2_path=None):
    """Cohen's kappa against the second annotator, if their labels exist."""
    a1_path = Path(a1_path or DEFAULT)
    a2_path = Path(a2_path or ANNOT2)
    if not (a2_path.exists() and a1_path.exists()):
        return
    A1, A2 = pd.read_csv(a1_path), pd.read_csv(a2_path)
    if "gold" not in A2.columns:
        return
    norm = lambda d: d.gold.fillna("").astype(str).str.strip().str.lower()
    m = (pd.DataFrame({"item_id": A1.item_id, "a1": norm(A1)})
         .merge(pd.DataFrame({"item_id": A2.item_id, "a2": norm(A2)}), on="item_id"))
    m = m[(m.a1 != "") & (m.a2 != "")]
    print("\n" + "=" * 78 + "\nInter-annotator agreement\n" + "=" * 78)
    if len(m) < 2:
        print("second annotator has not returned labels for these items yet")
        return
    print(f"items labelled by both : {len(m)}")
    print(f"raw agreement          : {(m.a1 == m.a2).mean() * 100:.1f}%")
    print(f"Cohen's kappa          : {cohen_kappa(m.a1.to_numpy(), m.a2.to_numpy()):.3f}")
    print("\nBenHalluEval reports Fleiss kappa 0.911-0.926 with three annotators;")
    print("that is the comparison a reviewer will make.")
    print("\ndisagreements (rows=you, cols=annotator 2):")
    print(pd.crosstab(m.a1, m.a2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(DEFAULT))
    ap.add_argument("--annotator2", default=None,
                    help="second annotator's file, for Cohen's kappa")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    path = Path(args.sample)
    if not path.exists():
        sys.exit(f"missing {path} - run verification/make_review_sample.py first")
    S = pd.read_csv(path)
    if "gold" not in S.columns:
        sys.exit("no `gold` column in this file")

    S["gold"] = S.gold.fillna("").astype(str).str.strip().str.lower()
    bad = sorted(set(S.gold) - set(VALID) - {""})
    if bad:
        sys.exit(f"unrecognised labels in `gold`: {bad}\nuse one of {VALID}")

    n_blank = int((S.gold == "").sum())
    if n_blank == len(S):
        print(f"{path.name}: 0 of {len(S)} items labelled - nothing to score yet.\n")
        print("Fill the `gold` column with one of:", " | ".join(VALID))
        print("\nEverything downstream is ready. This script scores all four decision")
        print("rules, reweights for the stratified sampling, runs McNemar between the")
        print("rules, and computes Cohen's kappa against the second annotator's file")
        print("as soon as both exist.")
        print("\nCheck the machinery meanwhile:  python verification/score_labels.py --selftest")
        return

    L = S[S.gold != ""].copy()
    msg = f"{path.name}: {len(L)} of {len(S)} items labelled"
    print(msg + (f" ({n_blank} blank - scoring what exists)" if n_blank else ""))

    n_unsure = int((L.gold == "unsure").sum())
    L = L[L.gold != "unsure"]
    if n_unsure:
        print(f"{n_unsure} labelled `unsure` - excluded from metrics, not reassigned")
    if L.empty:
        sys.exit("every label was `unsure` - nothing to score")

    sizes = population_strata()
    if sizes:
        print(f"population strata: {sizes} (total {sum(sizes.values())})")

    rule_cols = [c for c in L.columns if c.startswith("verdict__")]
    if not rule_cols:
        sys.exit("no verdict__* columns here - score annotator 1's file, not annotator 2's")

    gold = L.gold.to_numpy()
    w = stratum_weights(L, sizes) if sizes and "stratum" in L.columns else None

    print("\n" + "=" * 78 + "\nAccuracy per rule\n" + "=" * 78)
    print(f"{'rule':<24}{'sample':>20}{'population':>14}")
    for rc in rule_cols:
        pred = L[rc].to_numpy()
        k = int((pred == gold).sum())
        lo, hi = wilson(k, len(gold))
        cell = f"{k / len(gold) * 100:.1f}% [{lo * 100:.0f},{hi * 100:.0f}]"
        pop = f"{np.average(pred == gold, weights=w) * 100:.1f}%" if w is not None else ""
        print(f"{rc.replace('verdict__', ''):<24}{cell:>20}{pop:>14}")

    for rc in rule_cols:
        pred = L[rc].to_numpy()
        print("\n" + "-" * 78 + f"\n{rc.replace('verdict__', '')}\n" + "-" * 78)
        t = prf(gold, pred, w)
        print(f"{'class':<18}{'prec':>6}{'95% CI':>14}{'rec':>6}{'95% CI':>14}"
              f"{'F1':>6}{'nPred':>6}{'nGold':>6}")
        for r in t.itertuples():
            pci = f"[{r.p_ci[0]:.2f},{r.p_ci[1]:.2f}]"
            rci = f"[{r.r_ci[0]:.2f},{r.r_ci[1]:.2f}]"
            print(f"{r.cls:<18}{r.precision:>6.3f}{pci:>14}{r.recall:>6.3f}{rci:>14}"
                  f"{r.f1:>6.3f}{r.n_pred:>6}{r.n_gold:>6}")
        print(f"{'macro-F1':<18}{t.f1.mean():>6.3f}")
        if w is not None:
            print("\nsame classes, reweighted to the 536-claim population "
                  "(no intervals - these are design-weighted):")
            print(f"{'class':<18}{'prec':>6}{'rec':>6}{'F1':>6}")
            for r in t.itertuples():
                print(f"{r.cls:<18}{r.precision_pop:>6.3f}{r.recall_pop:>6.3f}"
                      f"{r.f1_pop:>6.3f}")
            print(f"{'macro-F1':<18}{t.f1_pop.mean():>18.3f}")
        print("\nconfusion (rows=gold, cols=predicted):")
        print(pd.crosstab(pd.Series(gold, name="gold"),
                          pd.Series(pred, name="pred")).to_string())

    base = "verdict__entail_first_SHIPPED"
    if base in rule_cols and len(rule_cols) > 1:
        print("\n" + "=" * 78 + "\nMcNemar vs the shipped rule\n" + "=" * 78)
        print(f"{'rule':<24}{'shipped right':>15}{'other right':>13}{'p':>10}")
        for rc in rule_cols:
            if rc == base:
                continue
            n01, n10, p = mcnemar(gold, L[base].to_numpy(), L[rc].to_numpy())
            print(f"{rc.replace('verdict__', ''):<24}{n10:>15}{n01:>13}{p:>10.4f}"
                  f"{'  *' if p < 0.05 else ''}")

    agreement(path, args.annotator2)


if __name__ == "__main__":
    main()

# Verification scoring harness

Everything here runs on CPU from saved outputs. No GPU, no Kaggle, no re-running
notebook G.

## Run order

```bash
python verification/recompute_verdicts.py     # rebuild verdicts under 4 decision rules
python verification/make_review_sample.py     # build the labelling sample
# ... label the `gold` column (tools/annotate.html, or any spreadsheet) ...
python verification/score_labels.py           # precision / recall / F1 / kappa
python verification/score_labels.py --selftest # check the metrics against sklearn
```

`tools/annotate.html` opens either sample file directly and picks its mode from the
columns. `node tools/test_annotate.mjs` tests it headlessly.

## The bug this harness exists to fix

Notebook G decided verdicts like this:

```python
verdict = np.where(max_entail >= 0.50, "supported",
           np.where(max_contradict >= 0.50, "contradicted", "not_in_curriculum"))
```

Entailment is tested first and wins outright, so the contradiction score is never
consulted once entailment clears the threshold. Consequences in the shipped run:

- **20 of 122** `supported` verdicts (16.4%) had the graph contradicting the claim
  *more strongly* than supporting it
- **40 of 122** (33%) had contradiction also clearing 0.50
- `supported` moves from **22.8%** of claims to **15.3%** if contradiction is given
  priority instead — a 7.5-point swing on a headline number

The two scores usually come from *different* facts (`entail_fi != contradict_fi`), so
the graph genuinely can both support and contradict one claim. That is a real
property of the data, not noise to be tie-broken away silently.

`recompute_verdicts.py` reports four rules side by side:

| rule | behaviour |
|---|---|
| `entail_first_SHIPPED` | as published; entailment wins outright |
| `contradict_first` | a curriculum contradiction outranks support — conservative, which is what a tutoring guardrail wants |
| `argmax` | whichever score is higher wins, if it clears the threshold |
| `margin_0.1` | argmax, but reports `conflicting` when both fire within 0.10 |

**Which rule the paper adopts is a research decision, not a code one.** The labels
are what settle it: each rule is scored against the same gold labels, so the choice
can be made on evidence rather than on which number looks better.

## The sample

`make_review_sample.py` replaced `nli_verdict_review_sample.csv`, which had two
problems. It was cast into the triple-review schema so the old annotation tool could
read it (`subject`=claim, `relation`=verdict, `object`=evidence), and it had no column
to record a label in. It also asked "was this verdict right?", which only scores the
one rule that happened to ship.

The task is now: given the claim, the question asked, and the strongest fact the graph
offers in *each* direction, what is the correct verdict? One gold label scores every
rule at once and yields per-class precision and recall instead of bare accuracy.

Three strata, because they answer different questions:

| stratum | n | of | measures |
|---|---|---|---|
| `rules_disagree` | 40 | 40 | which decision rule tracks your judgement |
| `agree_sup_con` | 80 | 179 | precision of the agreed verdicts |
| `not_in_curriculum` | 80 | 317 | how often retrieval missed a fact that *is* in the curriculum |

That third stratum has never been checked. 59% of claims land in
`not_in_curriculum`, and nothing currently distinguishes "the curriculum really is
silent on this" from "entity linking missed it". It is the likeliest place for the
method to be quietly wrong.

Files written:

- `eval/run_G/verdict_gold_sample.csv` — 200 items, label the `gold` column
- `eval/run_G/verdict_gold_sample_50.csv` — same strata, 50 items, for a short sitting
- `eval/run_G/verdict_gold_sample_annotator2.csv` — same items reshuffled with the
  system verdicts withheld, so a second annotator's labels stay independent

Label with one of `supported` / `contradicted` / `not_in_curriculum` / `unsure`.
`unsure` is a real answer and is reported separately, never folded into a class.

## Two things the scorer is careful about

**Stratification.** The sample oversamples disagreements (40 of 40) relative to
`not_in_curriculum` (80 of 317), so unweighted percentages do not describe the
536-claim population. Every headline is reported twice — `sample` as counted, and
`population` reweighted by inverse sampling probability. Quote the population figure.

**Intervals.** Unweighted estimates are shown with matching Wilson intervals;
weighted estimates are shown in a separate block with no intervals, because pairing a
design-weighted estimate with an unweighted interval produces brackets that do not
contain their own estimate.

## Still open

- the 200 labels, which is the only thing standing between this and an accuracy figure
- a second annotator on the same items, for Cohen's kappa (BenHalluEval reports Fleiss
  0.911–0.926 with three annotators; that is the comparison a reviewer will make)
- notebook B's mention edges still carry the old lemmatiser's output
- whether to re-extract toward entity-to-entity triples, which decides whether
  PGR-style graph reasoning is available at all

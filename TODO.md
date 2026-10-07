# Next up

Ordered by what unblocks the most. Times are honest, not motivational.

## 1. Produce a number (nothing else in the project is blocked on anything but this)

- [ ] **Label 20 verdicts — ~10 min.** `eval/run_G/verdict_gold_quick20_notincurriculum.csv`
      in `tools/annotate.html`, keys 1/2/3/U. One stratum, fast calls. Tests whether the
      59% `not_in_curriculum` rate is real or just entity linking failing — the biggest
      unchecked assumption in the method.
- [ ] **Then 50 if it goes fine — ~30 min.** `verdict_gold_sample_50.csv`. Gets you to
      ±13%, which is reportable with the caveat stated.
- [ ] **200 when you have an afternoon — ~2 h.** `verdict_gold_sample.csv`. ±6%.

Scoring is already built: `python verification/score_labels.py` the moment any of these
has its `gold` column filled.

## 2. Start the long-lead item now (5 minutes of your time)

- [ ] **Send `verdict_gold_sample_annotator2.csv` to a second annotator.** System verdicts
      are already withheld so their labels stay independent. Cohen's κ is required for the
      tier you're targeting and it's the single easiest thing to let slip. BenHalluEval
      used three annotators at Fleiss κ 0.911–0.926; that's the comparison a reviewer makes.

## 3. Reading, in this order (Phase 1, still open)

- [ ] **BenHalluEval — ~1 h.** Your proposal says it scores against *open-domain* fact
      sources. Their §3.1 says the supporting passage is provided at evaluation time.
      One of those is wrong; fix whichever it is. Their novelty paragraph is also a good
      template for how narrowly to phrase your own.
- [ ] **K12-KGraph — ~1 h.** Closest methodological comparison. Compare their nine *node*
      types against your nine *relation* types, and read §3.2 against your own pipeline.
- [ ] **PGR — ~45 min.** Lower urgency than I said earlier: the re-extraction question is
      now answered by measurement (see below). Still needed for related work, and §1 names
      the NLI paradigm your verifier uses as the thing it improves on.
- [ ] HCMUT and FactNet — verify one specific claim each (noted in `papers/EXTRACTED_FACTS.md`).
- [ ] GraphMASAL, NCTB-QA, BERT-QA, UDDIPOK — short entries.

`papers/EXTRACTED_FACTS.md` has the citations, numbers and page pointers already pulled
out, so reading is verification rather than excavation. The "My Notes" fields stay yours.

## 4. Decisions only you can make (no work, just a call)

- [ ] **Is entity-to-free-text a finding or a limitation?** Only 6.7% of triples are
      entity-to-entity, and the relations carrying your content are not entity-to-entity
      even in principle (`কাজ` 1.1%, `সংজ্ঞা` 0%) — curriculum biology is definitional and
      functional prose. Framing this as a property of the domain vs. a defect to fix
      changes the shape of the paper. Decide before you read PGR, not after.
- [ ] **Which decision rule the verifier uses.** Four candidates in
      `verification/recompute_verdicts.py`; the labels will tell you which tracks your
      judgement. Don't pick on which number looks better.
- [ ] **Resync the figures, or leave them?** `kg/figures/*` were rendered from the old
      1,152-edge graph; the corrected graph is 1,104 edges and 60.5% connectivity, not
      62.4%. One notebook B run on Kaggle fixes it. The report faculty already have quotes
      62%.
- [ ] **Add the missing `বৈশিষ্ট্য` relation?** Would need a re-extraction pass. Scope call.

## 5. Advisor (Phase 0 was never formally closed)

- [ ] Confirm the locked scope: Biology, Class 9–10, detection-only.
- [ ] Confirm no overlapping unpublished work in the department.
- [ ] Ask about the second annotator — whether a labmate counts or it needs to be someone
      specific.
- [ ] Mention the verdict-logic bug yourself rather than having it found. It moved
      "supported" from 22.8% to 15.3% depending on the rule, and owning it reads better
      than it looks.

## Not on this list because it's mine

Scoring harness, annotation tooling, graph rebuild and the measurement scripts are done
and verified. Nothing of mine is blocked on anything except your labels.

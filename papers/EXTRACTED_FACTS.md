# Extracted facts from the papers

**What this is:** citations, methods, dataset sizes and quotable numbers pulled out of the
PDFs so you can verify rather than excavate. Every line here can be checked against the
paper in seconds.

**What this is not:** your notes. It contains no judgement about how any of this relates to
your proposal. Each entry ends with the questions you need to answer — those answers are
what `notes/*.md` is for, and they become your Related Work section.

**Depth marker** on each entry says how much of the paper I actually read. Where it says
"abstract only", treat the summary as a pointer, not a description.

---

## 1. BenHalluEval — Bengali hallucination benchmark
`benhalluegal_2026_bengali_hallucination.pdf` · 25pp · **read: abstract, related work, methodology**

Adib, Sani, Esham, Abrar, Tashdeed (Islamic University of Technology) and Chowdhury (UC Riverside).
*BenHalluEval: A Multi-Task Hallucination Evaluation Framework for Large Language Models on Bengali.*

**What they built.** A benchmark across four tasks: QA, Bangla–English code-mixed QA,
summarisation, and mathematical reasoning. They generate 12,000 hallucinated candidates with
GPT-5.4 across twelve hallucination types, then test whether nine LLMs can detect them.

**Evaluation protocol.** Dual-track: Track A measures false-positive rate on ground-truth
instances, Track B measures detection rate on hallucinated candidates. Combined into
*BenHalluScore*, equivalent to balanced error rate. Reported range 7.72%–55.42%.

**On LLM-as-judge.** Their own words, §1 contributions: they evaluate hallucinated candidates
"alongside ground-truth instances using an **LLM-as-judge framework**."

**Seed data.** QA uses TyDiQA-GoldP (Bengali) — 2,509 question–passage–answer triples. §3.1
states: *"The supporting passage is provided to models at evaluation time."*

**Annotation.** Three native Bengali annotators, Fleiss' κ = 0.911–0.926.

**How they phrase their own novelty** (§2, worth copying the style): *"Our claim is narrower
than any of these: BenHalluEval is the first hallucination-detection benchmark built
specifically for Bengali that covers both native-script Bengali and Bangla-English code-mixed
input, and evaluates them under a dual-track protocol."*

**Questions for your notes**
- Your proposal says it scores against *open-domain* fact sources. §3.1 says the supporting
  passage is provided at evaluation time. Does your contrast survive that, and if not, what
  is the accurate version?
- Is their unit of evaluation the same as yours? They judge whole generated answers; you
  judge individual claims.
- Three annotators at κ ≈ 0.92 is the bar in this space. What does that imply for your plan?

---

## 2. K12-KGraph — curriculum KG for Chinese K-12
`k12kgraph_2026_curriculum_kg.pdf` · 35pp · **read: abstract, section list**

Liang, Lin, Han, Ma, Wong, Qiang, Sun, Zhang. Peking University, Institute for Advanced
Algorithms Research Shanghai, OriginHub Technology, Zhongguancun Academy. July 2026.

**What they built.** A curriculum-aligned knowledge graph extracted from official People's
Education Press textbooks, covering mathematics, physics, chemistry and biology across
primary, middle and high school, with **nine node types**.

**Their framing.** Existing benchmarks (C-Eval, CMMLU, GaokaoBench, EduEval) measure only
whether a model can answer an exam question — factual recall. They argue effective
educational AI needs *curriculum cognition*: prerequisite chains, concept taxonomies,
experiment–concept links, pedagogical sequencing.

**Two uses.** Benchmarking, and **K12-Train**, a KG-guided supervised fine-tuning corpus of
7,335 samples.

**Headline numbers.** On their benchmark, Gemini-3-Flash reaches only 57% exact match; the
strong open-source Gemma-4-31B-IT reaches 46%.

**Sections worth reading directly:** 3.1 Schema Design, 3.2 Construction Pipeline,
3.3 Graph Statistics.

**Questions for your notes**
- Compare their nine node types against your nine relation types. Did they separate node
  types from edge types in a way you did not, and does that matter?
- Their KG serves benchmarking and training. Yours serves verification. Is that the real
  difference, or is it a difference in application of the same resource?
- Their construction pipeline is §3.2. How does it differ from yours, and did they solve
  anything you worked around?

---

## 3. HCMUT — cross-data educational KG, Vietnamese
`hcmut_2024_vietnamese_edu_kg.pdf` · 8pp · **read: abstract, section list**

Bui, Tran, Nguyen, Ho, Nguyen, Bui, Quan. Ho Chi Minh City University of Technology.
*Cross-Data Knowledge Graph Construction for LLM-enabled Educational Question-Answering
System: A Case Study at HCMUT.*

**What they built.** A KG from multi-source educational data at one university, used to
augment an LLM question-answering system. Introduces an "E-OED" framework and an
embedding-based method for relation discovery.

**Scale.** A total of **613 relationships** — far smaller than yours at 2,248.

**Sections:** 2.2 Knowledge Graph in Education Domain, 2.3 KG-augmented LLMs,
3.2 The E-OED Framework, 4.4 KG-augmented LLMs Approach.

**Questions for your notes**
- Your proposal claims this paper "explicitly notes this kind of resource is sparse" for
  Vietnamese. Find that sentence and quote it exactly, or drop the claim.
- University content versus school curriculum — does that change what the graph is for?

---

## 4. PGR — fact verification on KGs via programmatic reasoning
`pgr_2025_kg_programmatic_reasoning.pdf` · 16pp · **read: abstract, §3.3 method, §4.2 baselines**

Yuanzhen Hao and Desheng Wu, University of Chinese Academy of Sciences.
*Findings of the ACL: EMNLP 2025*, pages 5480–5495.

**What they built.** An LLM translates a claim into a short program over three primitives,
executed against the graph:
- `MATCH(head, relation, tail)` — does this triple exist? returns true/false
- `SEARCH(head, relation, None)` — find the missing entity
- `VERIFY(boolean)` — combine results, handles negation and conjunction

**Result.** 86.82% accuracy on FactKG with GPT-4o and 12-shot prompting, state of the art
against their baselines.

**The sentence that matters for your method** (§1): *"These methods follow an NLI-based
reasoning paradigm, employing natural language inference (NLI) models to predict the
entailment between the evidence and the claim."* That is the paradigm your verifier now uses;
PGR positions itself as an improvement over it.

**Their baselines** (§4.2), in two groups: *without evidence* — BERT, BlueBERT, Flan-T5, and a
12-shot ChatGPT; *with evidence* — GEAR, KG-GPT, ProgramFC.

**Structural requirement.** `MATCH` needs entity-to-entity triples. Your objects are mostly
descriptive phrases, so this method is not directly available to your graph as built.

**Questions for your notes**
- Is the entity-to-entity requirement a hard blocker, or could a subset of your graph support
  it? This decides whether the re-extraction is worth eleven GPU hours.
- Their baseline list is a reasonable template. Which of those do you actually need?

---

## 5. FactNet — billion-scale multilingual grounding KG
`factnet_2026_multilingual_kg.pdf` · 47pp · **read: abstract only**

**What they built.** 1.7B Wikidata assertions coupled with 3.01B evidence pointers drawn from
316 native Wikipedia editions, with byte-level traceability from every evidence unit to its
source. Plus *FactNet-Bench* for KG completion, QA and fact checking, with leakage controls.

**Useful comparison table** in the paper, listing fact-checking datasets by size:
FEVER 185K · MultiFC 35K · X-FACT 31K · AveriTeC 4.5K · FACTors 118K.

**Questions for your notes**
- Your open question was whether Bangla is covered at all. 316 Wikipedia editions suggests
  yes — find the language table and check what the Bangla coverage actually is.
- If Bangla is covered, does that affect your "no resource exists" claim? Note the difference
  between general-domain coverage and curriculum coverage.

---

## 6. GraphMASAL — graph-based multi-agent adaptive learning
`graphmasal_2026_adaptive_learning.pdf` · 9pp · **read: abstract only**

Zeng, Liu, Zhen. South China Normal University.

**What they built.** An intelligent tutoring system combining a dynamic knowledge graph for
stateful learner modelling with a multi-agent architecture, aimed at generating personalised
learning paths. Evaluated with 5 random seeds, reporting mean ± 95% CI over seeds × profiles.

**Questions for your notes**
- Confirm this is learning-path planning rather than factuality verification, which is what
  your skeleton table assumes. If so this is a short entry — adjacent work, different problem.

---

## 7. NCTB-QA — large-scale Bangla educational QA
`nctb_qa_2026_dataset.pdf` · 18pp · **read: abstract only**

**What they built.** 87,805 question–answer pairs from 50 NCTB textbooks. Deliberately
balanced: 57.25% answerable, 42.75% unanswerable, with adversarially designed distractors.

**Benchmarks.** BERT, RoBERTa, ELECTRA. BERT improves F1 from 0.150 to 0.620 after
fine-tuning, a 313% relative gain.

**Also mentions** a Bangla-TextBook corpus: ~10 million tokens from 163 textbooks, grades 6–12.

**Questions for your notes**
- You used a different NCTB QA set for your 91 reference-answer questions. Does this one
  overlap your Biology Class 9–10 scope, and would it give you a larger evaluation set?
- Their unanswerable-question design is close to your "not in curriculum" verdict. Is there
  something to borrow?

---

## 8. Multiple BERT models for Bangla QA on NCTB textbooks
`bertqa_2024_nctb.pdf` · 15pp · **read: abstract only**

**What they did.** Compared RoBERTa Base, Bangla-BERT and BERT Base on Bangla passage-based
QA from NCTB textbooks, classes 6–10. Around 3,000 human-annotated instances; the paper also
references 3,000 context passages and 14,889 question–answer pairs.

**Result.** Bangla-BERT best at F1 0.75, EM 0.53. RoBERTa Base weakest.

**Questions for your notes**
- Bangla-BERT beating multilingual models here is relevant to your own model choices. Does it
  change anything about your extraction or entailment setup?

---

## 9. UDDIPOK — Bangla reading comprehension dataset
`UDDIPOK A reading comprehension based.pdf` · 7pp · **read: abstract only**

Aurpa, Ahmed, Rifat, Anwar, Ali. Jahangirnagar University and others.
*Data in Brief* 47 (2023) 108933.

**What they built.** 270 passages and 3,636 questions from textbooks, exams and newspapers.
Reported lowest accuracy 73.23%.

**Questions for your notes**
- This is a data article rather than a method paper. Probably a one-line entry in related
  work — confirm that and move on.

---

## Not collected

**NCTB-SchoolText** (`data.mendeley.com/datasets/f3882ccczp`) is the corpus your graph is
built from. It is a dataset deposit, not a paper. You have worked with it for a month and
know it better than its description does — cite it, do not read it.

---

## Suggested order

1. **BenHalluEval** — your positioning depends on it, and one claim already looks shaky
2. **K12-KGraph** — the closest methodological comparison
3. **PGR** — decides the re-extraction question
4. HCMUT, FactNet — verify the two specific claims flagged above
5. GraphMASAL, NCTB-QA, BERT-QA, UDDIPOK — shorter entries

Two per sitting is realistic. The field that matters in each note is *where does this differ
from my proposal* — the rest can be a line each.

"""Rebuild the graph CSVs with the corrected Bangla lemmatiser. CPU only.

Notebook B's mention edges were built with the broken lemmatiser - it stripped a
trailing র from any token of 4+ letters, so প্রকার->প্রকা, যন্ত্র->যন্ত্,
শরীর->শরী, and it had no check that the stem it produced was a real word. That
bug was fixed in notebook G but never backported to B, so kg/graph/*.csv still
carries the old output.

This script redoes stages 2 and 4 of that pipeline locally:

  1. rebuilds the normalised corpus from NCTB-SchoolText (notebook A stage 2,
     including its OCR repairs and regression asserts)
  2. counts corpus token frequencies, which the corrected lemmatiser needs to
     decide whether a stem is a real word
  3. rebuilds nodes, fact edges and mention edges (notebook B stages 2-4)
  4. diffs the result against the committed CSVs before writing

It does NOT regenerate kg/figures/*. Those need mplcairo+libraqm for Bangla
glyph shaping, which is Linux-only, so they stay a Kaggle job.

Usage: python scripts/rebuild_graph.py [--dry-run]
"""
import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
CORPUS_GLOB = str(ROOT / "NCTB-SchoolText" / "classNineTen"
                  / "processed_chapters_biology_secondary" / "*.jsonl")
CLEAN_PQ = ROOT / "data" / "biology_clean.parquet"
TRIPLES = ROOT / "kg" / "triples" / "biology_all_triples.csv"
GRAPH = ROOT / "kg" / "graph"

WORD = re.compile(r"[ঀ-৿]+|[A-Za-z0-9]+")
BENGALI_TOKEN = re.compile(r"[ঀ-৿]+")
MENTION_EDGE_TYPE = "সম্পর্কিত"
MIN_STEM_FREQ = 3        # notebook G's threshold: corpus must vouch for a stem

# --- notebook B's lemmatiser, for comparison only -------------------------
OLD_SUFFIXES = ["গুলোর", "গুলোকে", "গুলো", "টিকে", "গুলি", "দের", "টির", "টি",
                "য়ের", "এর", "কে", "ের", "রা", "র"]
# --- the corrected one, as fixed in notebook G ----------------------------
SUFFIXES = ["গুলোকে", "গুলোর", "গুলো", "গুলি", "টিকে", "দের", "টির", "য়ের",
            "এর", "ের", "কে", "টি", "রা"]      # trailing র removed
GENERIC = {"মাধ্যম", "ধরন", "জিনিস", "সময়", "স্থান", "গুরুত্ব", "নাম",
           "ব্যাপার", "কারণ", "ফলে", "দিক"}

TITLES = {
    1: ("জীবন পাঠ", "জীবনপাঠ"), 2: ("জীবকোষ ও টিস্যু", None),
    3: ("কোষ বিভাজন", None), 4: ("জীবনীশক্তি", None),
    5: ("খাদ্য, পুষ্টি ও পরিপাক", None), 6: ("জীবে পরিবহন", "জীবে পরিবহণ"),
    7: ("গ্যাসীয় বিনিময়", None), 8: ("রেচন প্রক্রিয়া", None),
    9: ("দৃঢ়তা প্রদান ও চলন", None), 10: ("সমন্বয়", None),
    11: ("জীবের প্রজনন", None),
    12: ("জীবের বংশগতি ও জৈব অভিব্যক্তি", "জীবের বংশগতি ও বিবর্তন"),
    13: ("জীবের পরিবেশ", None), 14: ("জীবপ্রযুক্তি", None),
}
BLOCKLIST = {"ব্যস্ত"}
EXTRA_FIX = {"সৌরশস্তিকে": "সৌরশক্তিকে", "দৃষ্টিশস্তি": "দৃষ্টিশক্তি",
             "বাকশস্তি": "বাকশক্তি", "ইচ্ছাশস্তির": "ইচ্ছাশক্তির",
             "চিন্তাশস্তি": "চিন্তাশক্তি"}
REGRESSION_GUARD = ["বস্তু", "মস্তিষ্ক", "স্তর", "প্রস্তুত", "স্ত্রী", "অন্ত",
                    "চিন্তা", "সন্তান", "শান্ত", "যন্ত্র", "তন্ত্র", "বাস্তুতন্ত্র"]


def bengali_ratio(s):
    letters = [c for c in s if c.isalnum()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ঀ" <= c <= "৿") / len(letters)


def build_clean_corpus():
    """Notebook A stage 2, verbatim in effect: OCR repair then junk filtering."""
    if CLEAN_PQ.exists():
        clean = pd.read_parquet(CLEAN_PQ)
        print(f"corpus: {len(clean)} clean chunks (cached)")
        return clean

    files = sorted(glob.glob(CORPUS_GLOB))
    if not files:
        sys.exit(f"no corpus files at {CORPUS_GLOB}\n"
                 "NCTB-SchoolText/ is gitignored - re-download it from "
                 "data.mendeley.com/datasets/f3882ccczp")
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    df = pd.DataFrame(rows)

    counts = collections.Counter()
    for t in df.text:
        counts.update(BENGALI_TOKEN.findall(t))
    # OCR mangles ক্ত two ways, into স্ত and into ন্ত. Only repair a token when
    # its ক্ত form independently occurs in the corpus.
    kta = {w for w in counts if "ক্ত" in w}
    fixes = {}
    for wrong in ("স্ত", "ন্ত"):
        fixes.update({w: w.replace(wrong, "ক্ত") for w in counts
                      if wrong in w and w not in BLOCKLIST
                      and w.replace(wrong, "ক্ত") in kta})
    fixes.update(EXTRA_FIX)

    before = collections.Counter(BENGALI_TOKEN.findall("\n".join(df.text)))
    df["text"] = df.text.map(
        lambda t: BENGALI_TOKEN.sub(lambda m: fixes.get(m.group(), m.group()), t))
    after = collections.Counter(BENGALI_TOKEN.findall("\n".join(df.text)))
    for w in REGRESSION_GUARD:
        assert before[w] == after[w], f"REGRESSION: {w} {before[w]} -> {after[w]}"

    def strip_noise_lines(text):
        return "\n".join(ln for ln in text.split("\n")
                         if not ln.strip() or bengali_ratio(ln) >= 0.55
                         or len(ln.strip()) < 4)

    df["text"] = df.text.map(strip_noise_lines)
    df["chapter_title"] = df.chapter_no.map(lambda c: TITLES[c][0])
    df["chapter_title_alt"] = df.chapter_no.map(lambda c: TITLES[c][1])
    df["n_chars"] = df.text.str.len()
    df["is_junk"] = (df.n_chars < 80) | (df.text.map(bengali_ratio) < 0.5)

    clean = (df[~df.is_junk][["chunk_id", "class", "subject", "chapter_no",
                              "chapter_title", "chapter_title_alt", "text", "n_chars"]]
             .sort_values(["chapter_no", "chunk_id"]).reset_index(drop=True))
    CLEAN_PQ.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_PQ, index=False)
    print(f"corpus: {len(fixes)} token repairs, kept {len(clean)}/{len(df)} chunks")
    return clean


def canon(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s.strip(" ।,:;()[]\"'-–—")


def make_lemma(vocab, suffixes, min_freq=MIN_STEM_FREQ):
    """A lemmatiser closed over the corpus vocabulary.

    min_freq is notebook G's fix: only accept a stem the corpus has seen at
    least this many times, so stripping cannot invent a non-word. min_freq=0
    disables the check, which is notebook B's original behaviour.
    """
    def lemma(w):
        for s in suffixes:
            if w.endswith(s) and len(w) - len(s) >= 3:
                stem = w[: -len(s)]
                if min_freq == 0 or vocab.get(stem, 0) >= min_freq:
                    return stem
        return w
    return lemma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report the diff without writing kg/graph/*.csv")
    ap.add_argument("--min-stem-freq", type=int, default=MIN_STEM_FREQ,
                    help="how often the corpus must contain a stem for the "
                         "lemmatiser to accept it (notebook G uses 3)")
    args = ap.parse_args()

    clean = build_clean_corpus()
    vocab = collections.Counter()
    for t in clean.text:
        vocab.update(WORD.findall(t))
    print(f"corpus vocabulary: {len(vocab)} distinct tokens, "
          f"{sum(vocab.values())} total")

    t = pd.read_csv(TRIPLES)
    t["subject"] = t.subject.map(canon)
    t["object"] = t.object.map(canon)
    t = t[(t.subject.str.len() > 1) & (t.object.str.len() > 1)].reset_index(drop=True)
    print(f"triples: {len(t)} after canon, {t.relation.nunique()} relations")

    # A subject should be a noun phrase; a long one is usually a clause the
    # extractor mis-assigned. Flagged on the edge, not dropped (notebook B.3).
    t["subj_tokens"] = t.subject.map(lambda s: len(WORD.findall(s)))
    t["subj_suspect"] = t.subj_tokens > 5

    ent = sorted(set(t.subject))
    nodes = pd.DataFrame({"node_id": range(len(ent)), "label": ent})
    nid = dict(zip(nodes.label, nodes.node_id))
    chap = t.groupby("subject").chapter_no.agg(lambda s: sorted(set(s)))
    deg = t.subject.value_counts()
    nodes["chapters"] = nodes.label.map(lambda l: ",".join(map(str, chap.get(l, []))))
    nodes["n_facts"] = nodes.label.map(lambda l: int(deg.get(l, 0)))

    facts = pd.DataFrame({
        "triple_id": t.triple_id,
        "head_id": t.subject.map(nid),
        "head": t.subject,
        "relation": t.relation,
        "value": t.object,
        "chapter_no": t.chapter_no,
        "chunk_id": t.chunk_id,
        "subj_suspect": t.subj_suspect,
    })

    lemma_new = make_lemma(vocab, SUFFIXES, args.min_stem_freq)
    lemma_old = make_lemma(vocab, OLD_SUFFIXES, 0)

    # how much damage the old lemmatiser did, over the tokens that matter
    ent_tokens = sorted({w for e in ent for w in WORD.findall(e)})
    changed = [(w, lemma_old(w)) for w in ent_tokens if lemma_old(w) != lemma_new(w)]
    print(f"\nentity tokens the two lemmatisers disagree on: "
          f"{len(changed)} of {len(ent_tokens)} ({len(changed)/len(ent_tokens)*100:.1f}%)")
    for w, old in changed[:12]:
        print(f"    {w:<22} old -> {old:<20} corrected -> {lemma_new(w)}")
    if len(changed) > 12:
        print(f"    ... and {len(changed) - 12} more")

    def mention_table(lemma):
        toks = lambda s: [lemma(w) for w in WORD.findall(s)]
        v = {e: tuple(toks(e)) for e in ent}
        v = {e: tk for e, tk in v.items()
             if tk and len("".join(tk)) >= 4 and e not in GENERIC}
        by_len = collections.defaultdict(dict)
        for e_, tk in v.items():
            by_len[len(tk)][tk] = e_
        max_n = max(by_len)

        def mentions(text):
            tk, found, covered = toks(text), [], set()
            for n in range(max_n, 0, -1):
                table = by_len.get(n)
                if not table:
                    continue
                for i in range(len(tk) - n + 1):
                    if any(j in covered for j in range(i, i + n)):
                        continue
                    hit = table.get(tuple(tk[i:i + n]))
                    if hit:
                        found.append(hit)
                        covered.update(range(i, i + n))
            return found

        rows = []
        for r in t.itertuples():
            for tail in mentions(r.object):
                if tail == r.subject:
                    continue
                rows.append({"triple_id": r.triple_id,
                             "head_id": nid[r.subject], "head": r.subject,
                             "tail_id": nid[tail], "tail": tail,
                             "relation": MENTION_EDGE_TYPE,
                             "source_relation": r.relation,
                             "chapter_no": r.chapter_no})
        return (pd.DataFrame(rows)
                .drop_duplicates(subset=["head", "tail", "relation"])
                .reset_index(drop=True))

    ment = mention_table(lemma_new)
    ment_old = mention_table(lemma_old)

    def connectivity(m):
        adj = collections.defaultdict(set)
        for r in m.itertuples():
            adj[r.head_id].add(r.tail_id)
            adj[r.tail_id].add(r.head_id)
        seen, comps = set(), []
        for s in adj:
            if s in seen:
                continue
            stack, comp = [s], []
            seen.add(s)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v_ in adj[u] - seen:
                    seen.add(v_)
                    stack.append(v_)
            comps.append(len(comp))
        return (max(comps) if comps else 0), len(comps)

    print(f"\n{'':<26}{'old lemmatiser':>16}{'corrected':>14}{'change':>10}")
    rows = []
    for label, a, b in [
        ("mention edges", len(ment_old), len(ment)),
        ("triples linked", ment_old.triple_id.nunique(), ment.triple_id.nunique()),
        ("entities touched",
         len(set(ment_old["head"]) | set(ment_old["tail"])),
         len(set(ment["head"]) | set(ment["tail"]))),
    ]:
        rows.append((label, a, b))
    big_o, nc_o = connectivity(ment_old)
    big_n, nc_n = connectivity(ment)
    rows += [("largest component", big_o, big_n), ("components", nc_o, nc_n)]
    for label, a, b in rows:
        d = b - a
        print(f"{label:<26}{a:>16}{b:>14}{d:>+10}")
    print(f"{'facts linked (%)':<26}"
          f"{ment_old.triple_id.nunique()/len(t)*100:>15.1f}%"
          f"{ment.triple_id.nunique()/len(t)*100:>13.1f}%")
    print(f"{'largest comp / nodes (%)':<26}{big_o/len(nodes)*100:>15.1f}%"
          f"{big_n/len(nodes)*100:>13.1f}%")

    # diff against what is committed
    old_files = {n: GRAPH / f"{n}.csv" for n in ("nodes", "fact_edges", "mention_edges")}
    print("\nagainst the committed CSVs:")
    for name, p in old_files.items():
        if not p.exists():
            print(f"  {name:<16} (absent)")
            continue
        was = len(pd.read_csv(p))
        now = {"nodes": len(nodes), "fact_edges": len(facts),
               "mention_edges": len(ment)}[name]
        print(f"  {name:<16}{was:>7} -> {now:<7} {now - was:+d}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    GRAPH.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(GRAPH / "nodes.csv", index=False, encoding="utf-8")
    facts.to_csv(GRAPH / "fact_edges.csv", index=False, encoding="utf-8")
    ment.to_csv(GRAPH / "mention_edges.csv", index=False, encoding="utf-8")
    print(f"\nwrote nodes.csv, fact_edges.csv, mention_edges.csv -> "
          f"{GRAPH.relative_to(ROOT)}")
    print("kg/figures/* NOT regenerated - Bangla shaping needs mplcairo+libraqm, "
          "so notebook B still owns the figures.")


if __name__ == "__main__":
    main()

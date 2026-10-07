"""How much of the graph is already entity-to-entity?

PGR-style graph reasoning needs MATCH(head, relation, tail) where `tail` is a
canonical entity, not a descriptive phrase. Our extraction produced
entity-to-free-text triples, so the open question is whether a usable
entity-to-entity subgraph already exists or whether re-extraction is required.

This only measures. What counts as "usable" is a judgement about the method, not
something a script can decide.

Usage: python scripts/measure_triple_shape.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252, which cannot encode Bangla. Do this here
# rather than relying on PYTHONIOENCODING, which `python -I` ignores.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
TRIPLES = ROOT / "kg" / "triples" / "biology_all_triples.csv"
BN = re.compile(r"[ঀ-৿]+")


def norm(s):
    """Strip whitespace and the danda, which extraction left on some objects."""
    return str(s).strip().strip("।").strip()


def main():
    T = pd.read_csv(TRIPLES)
    T["subject"] = T.subject.map(norm)
    T["object"] = T.object.map(norm)
    n = len(T)

    subjects = set(T.subject)
    # an "entity" is anything the extraction ever committed to as a subject
    print(f"{n} triples | {len(subjects)} distinct subjects\n")

    T["obj_words"] = [len(BN.findall(o)) for o in T.object]
    T["obj_is_entity"] = T.object.isin(subjects)

    # an object that is not itself an entity may still *contain* one as a
    # contiguous token run - that is the cheap-normalisation case
    def contains_entity(o):
        if o in subjects:
            return False
        w = BN.findall(o)
        for size in range(min(4, len(w)), 0, -1):
            for i in range(len(w) - size + 1):
                if " ".join(w[i:i + size]) in subjects:
                    return True
        return False

    T["obj_has_entity"] = [contains_entity(o) for o in T.object]

    exact = int(T.obj_is_entity.sum())
    partial = int(T.obj_has_entity.sum())
    neither = n - exact - partial

    print("Object shape")
    print(f"  object IS a known entity (PGR-ready)   {exact:>6}  {exact/n*100:>5.1f}%")
    print(f"  object CONTAINS a known entity         {partial:>6}  {partial/n*100:>5.1f}%")
    print(f"  free text, no entity in it             {neither:>6}  {neither/n*100:>5.1f}%")

    print("\nObject length (Bangla tokens)")
    q = T.obj_words.describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    for k in ("mean", "25%", "50%", "75%", "90%", "max"):
        print(f"  {k:<6}{q[k]:>8.1f}")

    print("\nBy relation, sorted by how entity-to-entity it already is")
    rows = []
    for rel, g in T.groupby("relation"):
        rows.append((rel, len(g), g.obj_is_entity.mean() * 100,
                     g.obj_has_entity.mean() * 100, g.obj_words.median()))
    rows.sort(key=lambda r: -r[2])
    print(f"  {'relation':<12}{'n':>6}{'obj=entity':>12}{'obj~entity':>12}{'med words':>11}")
    for rel, k, ex, pa, med in rows:
        print(f"  {rel:<12}{k:>6}{ex:>11.1f}%{pa:>11.1f}%{med:>11.0f}")

    # what a PGR-ready subgraph would actually look like
    sub = T[T.obj_is_entity]
    ents = set(sub.subject) | set(sub.object)
    print("\nIf only entity-to-entity triples were kept")
    print(f"  facts    {len(sub):>6}  ({len(sub)/n*100:.1f}% of the graph)")
    print(f"  entities {len(ents):>6}")
    print(f"  relations used: {sub.relation.nunique()} of {T.relation.nunique()}")
    deg = Counter(list(sub.subject) + list(sub.object))
    iso = sum(1 for e in ents if deg[e] == 1)
    print(f"  entities appearing in exactly one fact: {iso} ({iso/max(len(ents),1)*100:.0f}%)")

    print("\n  plus the cheap-normalisation tier (object contains an entity):")
    print(f"  facts    {len(sub) + partial:>6}  "
          f"({(len(sub)+partial)/n*100:.1f}% of the graph)")

    print("\n" + "-" * 70)
    print("What this does not settle: whether a graph that size supports the\n"
          "reasoning PGR does, and whether the relations that survive are the ones\n"
          "your eval questions actually ask about. Both need your read of the paper.")


if __name__ == "__main__":
    main()

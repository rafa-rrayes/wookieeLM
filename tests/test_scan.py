#!/usr/bin/env python3
"""The parallel scan and the blob title search, against the code they replaced.

Both rewrites exist for speed, and both are on the path that decides which
articles a training answer is built from. So the bar is not "looks right", it
is *identical output* -- the reference implementations below are the previous
versions of `Corpus.search_titles` and `Corpus.count_matches`, kept verbatim
so the comparison means something.

    uv run tests/test_scan.py
"""

from __future__ import annotations

import bisect
import importlib.util
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import corpus_scan as cs  # noqa: E402


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wc = load("wc", "wookiee_chat.py")


# ---- the implementations being replaced -------------------------------------

def ref_search_titles(corpus, query: str, limit: int = 20) -> list[Path]:
    """Verbatim, from before the key-blob rewrite."""
    q = wc.normalize(query)
    if not q:
        return []
    tokens = q.split()
    scored: list[tuple[int, int, int]] = []
    for i, key in enumerate(corpus.keys):
        if key == q:
            score = 100
        elif key.startswith(q + " "):
            score = 80
        elif q in key:
            score = 60
        elif len(tokens) > 1 and all(t in key for t in tokens):
            score = 40
        else:
            continue
        scored.append((-score, len(key), i))
    scored.sort()
    return [corpus.paths[i] for _, _, i in scored[:limit]]


def ref_count_matches(buf, starts, query: str, regex: bool = False,
                      cap: int = 500_000) -> tuple[dict[int, int], bool]:
    """Verbatim, from before the mmap/process-pool rewrite."""
    counts: dict[int, int] = {}
    hits = 0
    doc = 0

    def own(pos: int) -> int:
        nonlocal doc
        if pos < starts[doc]:
            doc = bisect.bisect_right(starts, pos) - 1
        else:
            while doc + 1 < len(starts) and starts[doc + 1] <= pos:
                doc += 1
        return doc

    if regex:
        pattern = re.compile(query.encode(), re.IGNORECASE)
        for m in pattern.finditer(buf):
            i = own(m.start())
            counts[i] = counts.get(i, 0) + 1
            hits += 1
            if hits >= cap:
                break
    else:
        needle = query.lower().encode()
        if not needle:
            return {}, False
        pos = buf.find(needle)
        while pos >= 0:
            counts[own(pos)] = counts.get(own(pos), 0) + 1
            hits += 1
            if hits >= cap:
                break
            pos = buf.find(needle, pos + len(needle))
    return counts, hits >= cap


# ---- a corpus to run them against -------------------------------------------

WORDS = ["ahsoka", "tano", "jedi", "master", "temple", "coruscant", "relay",
         "post", "epsilon", "one", "star", "destroyer", "sith", "lord", "the",
         "of", "battle", "clone", "wars", "thrawn", "grand", "admiral", "x"]


def build_corpus(root: Path, n: int, rng: random.Random) -> None:
    """Article names and bodies drawn from one small vocabulary.

    A tiny vocabulary is the point: it forces heavy overlap between titles and
    bodies, so queries actually hit the ranking tiers and the shard boundaries
    instead of matching nothing.
    """
    for i in range(n):
        title = " ".join(rng.choices(WORDS, k=rng.randint(1, 4))) + f" {i}"
        shard = root / title[0].upper()
        shard.mkdir(parents=True, exist_ok=True)
        body = " ".join(rng.choices(WORDS, k=rng.randint(20, 400)))
        (shard / f"{title.replace(' ', '_')}.md").write_text(
            f"---\ncontinuity: canon\n---\n\n# {title}\n\n{body}\n",
            encoding="utf-8")


def main() -> int:
    import tempfile

    rng = random.Random(1977)
    failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "wookieepedia"
        build_corpus(root, 600, rng)
        wc.INDEX_DIR = Path(tmp) / ".index"
        corpus = wc.Corpus(root)
        n_bytes = corpus.build_text_index()
        print(f"corpus: {len(corpus):,} articles, {n_bytes:,} bytes indexed")

        # A plain bytes copy for the reference scan, so the comparison is
        # against the old data structure as well as the old code.
        ref_buf = bytes(corpus._text[:])
        ref_starts = list(corpus._starts)
        assert len(ref_starts) == len(corpus.paths)

        # ---- queries -------------------------------------------------------
        queries = ["jedi", "ahsoka tano", "x", "grand admiral thrawn",
                   "the", "relay post epsilon", "nothing here at all",
                   "sith lord", "clone wars", "epsilon one", "star"]
        queries += [" ".join(rng.choices(WORDS, k=rng.randint(1, 3)))
                    for _ in range(40)]
        queries += [corpus.title_of(rng.choice(corpus.paths)) for _ in range(40)]

        # ---- 1. scan_range, whole range ------------------------------------
        for q in queries:
            want = ref_count_matches(ref_buf, ref_starts, q)
            got = cs.scan_range(corpus._text, corpus._starts, 0,
                                len(corpus._starts), q, False, 500_000)
            if want != got:
                print(f"  FAIL scan_range {q!r}: {want} != {got}")
                failures += 1
        print(f"scan_range, whole range:      {len(queries)} queries")

        # ---- 2. sharded and merged, every shard count ----------------------
        for shards in (1, 2, 3, 7, 8, 13, 64):
            for q in queries:
                want, _ = ref_count_matches(ref_buf, ref_starts, q)
                merged: dict[int, int] = {}
                for lo, hi in cs._split(len(corpus.paths), shards):
                    part, _ = cs.scan_range(corpus._text, corpus._starts,
                                            lo, hi, q, False, 500_000)
                    overlap = merged.keys() & part.keys()
                    if overlap:
                        print(f"  FAIL shards overlap on {q!r}: {overlap}")
                        failures += 1
                    merged.update(part)
                if want != merged:
                    print(f"  FAIL {shards} shards {q!r}: {want} != {merged}")
                    failures += 1
        print(f"sharded scan == unsharded:    7 shard counts x {len(queries)}")

        # ---- 3. shard boundaries cover every byte --------------------------
        for shards in (1, 3, 8, 64):
            bounds = cs._split(len(corpus.paths), shards)
            assert bounds[0][0] == 0 and bounds[-1][1] == len(corpus.paths)
            for (_, a), (b, _) in zip(bounds, bounds[1:]):
                assert a == b, "gap between shards"
        print("shard bounds:                 contiguous, complete")

        # ---- 4. regex path -------------------------------------------------
        patterns = [r"jedi", r"sith\s+lord", r"^star", r"epsilon\s+\d+",
                    r"[gk]rand", r"tano|thrawn", r"x{1,2}"]
        for p in patterns:
            want, _ = ref_count_matches(ref_buf, ref_starts, p, regex=True)
            merged = {}
            for lo, hi in cs._split(len(corpus.paths), 8):
                part, _ = cs.scan_range(corpus._text, corpus._starts, lo, hi,
                                        p, True, 500_000)
                merged.update(part)
            if want != merged:
                print(f"  FAIL regex {p!r}: {want} != {merged}")
                failures += 1
        print(f"regex, 8 shards == unsharded: {len(patterns)} patterns")

        # ---- 5. the cap ----------------------------------------------------
        _, capped = cs.scan_range(corpus._text, corpus._starts, 0,
                                  len(corpus._starts), "the", False, 5)
        if not capped:
            print("  FAIL cap not reported")
            failures += 1
        print("cap:                          reported")

        # ---- 6. the process pool == in-process -----------------------------
        pool = corpus.attach_scanner(workers=3)
        try:
            for q in queries[:30]:
                want, _ = ref_count_matches(ref_buf, ref_starts, q)
                got, _ = corpus.count_matches_idx(q)
                if want != got:
                    print(f"  FAIL SearchPool {q!r}: {want} != {got}")
                    failures += 1
            # and the tool that sits on top of it still renders
            out = wc.tool_search_text(corpus, "jedi", limit=3)
            assert "mention" in out and "###" in out, out[:200]
        finally:
            pool.close()
            corpus.scanner = None
        print("SearchPool == in-process:     30 queries, 3 workers")

        # ---- 7. search_titles ----------------------------------------------
        title_queries = queries + [
            "", "   ", "!!!", "Ahsoka_Tano", "STAR WARS", "jedi master temple",
            "one", "x 7", "the of", "zzz nothing",
        ]
        for q in title_queries:
            for limit in (1, 8, 20, 40):
                want = ref_search_titles(corpus, q, limit)
                got = corpus.search_titles(q, limit)
                if want != got:
                    print(f"  FAIL search_titles {q!r} limit={limit}:"
                          f"\n    want {[p.stem for p in want]}"
                          f"\n    got  {[p.stem for p in got]}")
                    failures += 1
        print(f"search_titles == reference:   {len(title_queries)} queries "
              f"x 4 limits")

        # ---- 8. the index is reused, and rebuilt when told ------------------
        again = wc.Corpus(root)
        if again.build_text_index() != n_bytes:
            print("  FAIL cached index differs in size")
            failures += 1
        if bytes(again._text[:]) != ref_buf:
            print("  FAIL cached index differs in content")
            failures += 1
        if again.build_text_index(rebuild=True) != n_bytes:
            print("  FAIL rebuilt index differs in size")
            failures += 1
        print("index cache:                   reused and rebuilt identically")

        # ---- 9. a changed corpus invalidates the stamp ----------------------
        (root / "Z").mkdir(parents=True, exist_ok=True)
        (root / "Z" / "New_Article.md").write_text("# New Article\njedi\n",
                                                   encoding="utf-8")
        grown = wc.Corpus(root)
        if grown.build_text_index() <= n_bytes:
            print("  FAIL stale index reused after an article was added")
            failures += 1
        print("index cache:                   invalidated by a new article")

    print()
    print("FAILURES:", failures) if failures else print("all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

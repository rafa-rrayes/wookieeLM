# Lessons

## Measure the machine before blaming the network

`answer_questions.py` looked API-bound: 192 concurrent requests, 13 q/s, a
progress bar that said 3h47m. Two `ps -o time` samples eight seconds apart
showed 7.14 s of CPU in 8 s of wall clock — 89% of *one* core on a ten-core
box. It was never waiting on DeepSeek.

**Rule:** before tuning concurrency, timeouts or batch size, sample the
process's CPU time. `pcpu` near 100 on a multi-core machine means one saturated
core, which for Python means the GIL, which means concurrency cannot help.

## `asyncio.to_thread` does not make blocking work concurrent

`ToolRunner` sent every tool to a thread and the docstring said the scan
"holds the GIL, so they go to a thread". A thread does not release the GIL for
you. `bytes.find` over 365 MB holds it for the whole 275 ms, event loop
included; the semaphore in front of it bounded queue depth and bought nothing.

**Rule:** `to_thread` helps only where the callee itself releases the GIL —
file and socket IO, `subprocess`, some C extensions. For pure-Python or
GIL-holding C (`bytes.find`, `re`, `json`), it is a process pool or nothing.

## A cache that stops inserting is worse than no cache

```python
if len(self.cache) < self.cache_size:      # never evicts; just stops caching
    self.cache[key] = result
```

On a 179k-question run this froze holding the first ~4,000 tool calls. Because
questions are processed grouped by source article, those 4,000 all came from
the first few hundred articles, so every question after that missed — and each
miss paid the full 275 ms scan.

**Rule:** a bounded cache needs an eviction policy. `OrderedDict` +
`move_to_end` + `popitem(last=False)` is four lines. A size check with no
eviction silently turns into a no-op at exactly the point the cache mattered.

## Don't recompute what start-up already computed

`tool_search_text` called `normalize(path.stem)` for every matching article —
a regex substitution per article, 50,000 of them on a common query — when
`corpus.keys[i]` already held the answer from the initial corpus walk. Found
only because the parent stayed hot after the scan was parallelised.

## Rewrites on a data path get an equivalence test, not a smoke test

Both rewrites (sharded scan, key-blob title search) decide which articles a
training answer is built from. `tests/test_scan.py` keeps the *previous
implementations verbatim* as reference functions and asserts identical output
over ~500 query/limit combinations, across seven shard counts and the regex
path. That is what makes "5.7x faster" a safe claim rather than a hopeful one.

## Verify the mechanism, not just the outcome

Switched the pool to `forkserver` and wrote a docstring claiming it stopped
workers re-importing the main module. It does not — `forkserver` children still
receive spawn preparation data and re-import `__main__`; the preload list only
covers the server process. The benchmark output said so plainly (worker prints
kept appearing) and the claim went in anyway.

**Rule:** when the justifying comment makes a factual claim about behaviour,
check the output that would disprove it before writing it down.

## A generated document must not hardcode a number the build already knows

The dataset card `package.py` writes said "the paraphrase pass is 98.9%
complete, only 7.2% of the generated questions have answers" as literal text in
an f-string. Both figures were stale within a day — the answering run kept
going — and the card is regenerated on every build, so the build had the true
values in `stats` the whole time and printed the wrong ones anyway.

**Rule:** if a document is generated, every quantity in it is computed from the
same data the rest of the document is computed from. A literal percentage in
generated prose is a bug with a delayed fuse.

## Don't infer a denominator — check whether the file is a pool or a remainder

Wrote "16,437 answered of 201,760 questions (8.1%)" by adding `questions.jsonl`
to `sft.jsonl`, assuming `answer_questions.py` removed questions from the pool
as it answered them. It does not; it only reads. 16,413 of the 16,437 answered
questions were still sitting in `questions.jsonl`, so the denominator described
a set that does not exist. `count.py` had the right figure on screen.

**Rule:** before combining two counts, verify the sets are disjoint by testing
membership, not by reasoning about what the producer "must" do.

## A filter aimed at junk files will eventually match real data

`enumerate_documents` skipped names starting with `.` to avoid `.DS_Store`, and
silently dropped `_/.48-caliber_Enforcer_pistol.md` — a real article. The suffix
allowlist already excluded `.DS_Store`, so the guard had no upside at all. It
surfaced only because the packager's count came out one short of `count.py`'s.

**Rule:** exclusion rules get stated as an allowlist of what belongs, not a
pattern-match on what a nuisance file happens to look like. And when two
independent counts of the same thing disagree by any amount, diff the
predicates before believing the newer one.

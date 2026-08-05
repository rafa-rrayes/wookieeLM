#!/usr/bin/env python3
"""package.py, against a fixture corpus small enough to check by hand.

The packager is the last thing that touches the data before someone else
trains on it, and every mistake it can make is silent: a dropped article, a
category list flattened to a string, a restricted source that slips into a
build meant for upload. So the checks here are about *what came out the other
side*, not about whether it ran.

    uv run tests/test_package.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq  # noqa: E402


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pkg = load("pkg", "package.py")


# ---- Fixture ----------------------------------------------------------------

ARTICLE = """---
title: "Ahsoka Tano"
continuity: canon
source: "Wookieepedia"
categories:
  - "Females"
  - "Togruta"
---

## Infobox

- **Name:** Ahsoka Tano
- **Species:** Togruta

**Ahsoka Tano** was a Togruta female who served as the Padawan learner of
Anakin Skywalker.

## Biography

She left the Jedi Order in 19 BBY.
"""

# No frontmatter categories, a different continuity, and a body only -- the
# ~28% of the corpus whose whole article is a lead paragraph.
LEAD_ONLY = """---
title: "Glim worm"
continuity: legends
source: "Wookieepedia"
---

The **glim worm** was a burrowing predator covered with knife-like scales.
"""

# Frontmatter and nothing else. document_row() must drop it rather than emit a
# row whose text is a bare title.
EMPTY = """---
title: "Empty Stub"
continuity: canon
---
"""

PARAPHRASED = """---
title: "Ahsoka Tano"
continuity: canon
source: "Wookieepedia"
categories:
  - "Females"
paraphrased_from: "wookieepedia/A/Ahsoka_Tano.md"
paraphrase_model: "deepseek-v4-flash"
---

Ahsoka Tano, a female Togruta, trained as Anakin Skywalker's Padawan.
"""

WIKIPEDIA = """---
title: "Star Wars"
source: "Wikipedia"
url: "https://en.wikipedia.org/wiki/Star_Wars"
categories:
  - "Star Wars"
---

**Star Wars** is an American epic space opera media franchise.
"""


def build_fixture(root: Path) -> None:
    """A corpus with one of every shape the packager has to handle."""
    wook = root / "corpus" / "wookieepedia"
    (wook / "A").mkdir(parents=True)
    (wook / "G").mkdir(parents=True)
    (wook / "A" / "Ahsoka_Tano.md").write_text(ARTICLE, encoding="utf-8")
    (wook / "G" / "Glim_worm.md").write_text(LEAD_ONLY, encoding="utf-8")
    (wook / "G" / "Empty_Stub.md").write_text(EMPTY, encoding="utf-8")
    # An index list, not a document. Must never become a row.
    (wook / "manifest.md").write_text("not an article", encoding="utf-8")
    # A real article whose title starts with a dot, and the macOS turd that
    # sits next to it. A naive "skip hidden files" rule drops both.
    (wook / "_").mkdir(parents=True)
    (wook / "_" / ".48-caliber_Enforcer_pistol.md").write_text(
        '---\ntitle: ".48-caliber Enforcer pistol"\ncontinuity: legends\n---\n\n'
        "The **.48-caliber Enforcer** was a slugthrower pistol.\n", encoding="utf-8")
    (wook / "_" / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")

    para = root / "corpus" / "wookieepedia_paraphrased1" / "A"
    para.mkdir(parents=True)
    (para / "Ahsoka_Tano.md").write_text(PARAPHRASED, encoding="utf-8")

    wiki = root / "corpus" / "wikipedia" / "S"
    wiki.mkdir(parents=True)
    (wiki / "Star_Wars.md").write_text(WIKIPEDIA, encoding="utf-8")

    books = root / "corpus" / "books"
    books.mkdir(parents=True)
    (books / "dooku.txt").write_text("A long time ago...\n", encoding="utf-8")

    (root / "questions").mkdir()
    (root / "questions" / "questions.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"q": "Who trained Ahsoka?", "type": "factoid",
         "source": "wookieepedia/A/Ahsoka_Tano.md", "continuity": "canon",
         "split": "train", "evidence": "Padawan learner of Anakin Skywalker."},
        {"q": "What was the glim worm covered with?", "type": "factoid",
         "source": "wookieepedia/G/Glim_worm.md", "continuity": "legends",
         "split": "eval", "evidence": "covered with knife-like scales"},
    ]) + "\n", encoding="utf-8")

    (root / "sft").mkdir()
    (root / "sft" / "sft.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"q": "Who trained Ahsoka?", "a": "Anakin Skywalker.", "type": "factoid",
         "source": "wookieepedia/A/Ahsoka_Tano.md", "continuity": "canon",
         "split": "train", "read": ["Ahsoka Tano"], "tools": 1,
         "model": "deepseek-v4-flash"},
        {"q": "What was the glim worm covered with?", "a": "That was scales.",
         "type": "factoid", "source": "wookieepedia/G/Glim_worm.md",
         "continuity": "legends", "split": "eval", "read": ["Glim worm"],
         "tools": 2, "model": "deepseek-v4-flash", "unsupported": ["That"]},
    ]) + "\n", encoding="utf-8")


def run_packager(root: Path, out: Path, *extra: str) -> None:
    """Point the module at the fixture and run its CLI end to end."""
    pkg.CORPUS_DIR = root / "corpus"
    pkg.QUESTION_FILE = root / "questions" / "questions.jsonl"
    pkg.SFT_FILE = root / "sft" / "sft.jsonl"
    argv = sys.argv
    sys.argv = ["package.py", "--out", str(out), *extra]
    try:
        pkg.main()
    finally:
        sys.argv = argv


def rows(out: Path, pattern: str) -> list[dict]:
    files = sorted(out.glob(pattern))
    if not files:
        return []
    return pq.read_table(files).to_pylist()


# ---- Checks -----------------------------------------------------------------

def main() -> int:
    failures = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal failures
        if ok:
            print(f"  ok    {label}")
        else:
            failures += 1
            print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        root.mkdir()
        build_fixture(root)
        out = Path(tmp) / "dist"
        run_packager(root, out)
        print()

        # ---- 1. documents in, rows out ------------------------------------
        wook = rows(out, "pretrain/wookieepedia-*.parquet")
        check(len(wook) == 3,
              "empty article, manifest.md and .DS_Store are not rows",
              f"expected 3 rows, got {len(wook)}: {[r['title'] for r in wook]}")

        by_title = {r["title"]: r for r in wook}
        check(".48-caliber Enforcer pistol" in by_title,
              "an article whose filename starts with a dot is still packaged",
              f"titles: {sorted(by_title)}")
        ahsoka = by_title.get("Ahsoka Tano", {})

        # ---- 2. the body survives byte-for-byte ---------------------------
        check("She left the Jedi Order in 19 BBY." in ahsoka.get("text", ""),
              "article body reaches the text column intact")
        check("- **Species:** Togruta" in ahsoka.get("text", ""),
              "infobox is kept, not dropped with the frontmatter")

        # ---- 3. continuity is in-band, not only in a column ---------------
        check("*Wookieepedia · Canon*" in ahsoka.get("text", ""),
              "continuity label is in the token stream")
        check(ahsoka.get("continuity") == "canon",
              "continuity is also a column")
        check(by_title.get("Glim worm", {}).get("continuity") == "legends",
              "second continuity value parsed independently")

        # ---- 4. frontmatter -> columns ------------------------------------
        check(ahsoka.get("categories") == ["Females", "Togruta"],
              "multi-value categories survive as a list",
              f"got {ahsoka.get('categories')!r}")
        check(by_title.get("Glim worm", {}).get("categories") == [],
              "an article with no categories gets an empty list, not null")
        check(ahsoka.get("url") == "https://starwars.fandom.com/wiki/Ahsoka_Tano",
              "attribution url derived from the title",
              f"got {ahsoka.get('url')!r}")
        check(ahsoka.get("doc_id") == "wookieepedia/A/Ahsoka_Tano.md",
              "doc_id is the corpus-relative path",
              f"got {ahsoka.get('doc_id')!r}")
        check(ahsoka.get("license") == "CC BY-SA 3.0", "license stamped per row")

        wiki = rows(out, "pretrain/wikipedia-*.parquet")
        check(len(wiki) == 1 and wiki[0]["url"].startswith("https://en.wikipedia.org"),
              "wikipedia keeps its own url rather than a derived one")
        check(len(wiki) == 1 and wiki[0]["license"] == "CC BY-SA 4.0",
              "wikipedia carries its own license")

        # ---- 5. paraphrases are linked to their source --------------------
        para = rows(out, "pretrain/wookieepedia_paraphrased1-*.parquet")
        check(len(para) == 1 and
              para[0]["derived_from"] == "wookieepedia/A/Ahsoka_Tano.md",
              "paraphrase records the article it was rewritten from",
              f"got {para[0]['derived_from']!r}" if para else "no rows")
        check(all(r["derived_from"] is None for r in wook),
              "source articles have no derived_from")

        # ---- 6. restricted sources stay out -------------------------------
        check(not (out / "restricted").exists(),
              "books/ is absent from a default build")
        check(not any("books" in p.name for p in out.rglob("*.parquet")),
              "no restricted shard anywhere in a default build")

        # ---- 7. SFT shape --------------------------------------------------
        sft_train = rows(out, "sft/train-*.parquet")
        sft_eval = rows(out, "sft/eval-*.parquet")
        check(len(sft_train) == 1 and len(sft_eval) == 1,
              "SFT rows land in the split they declare",
              f"train={len(sft_train)} eval={len(sft_eval)}")
        msgs = sft_train[0]["messages"] if sft_train else []
        check([m["role"] for m in msgs] == ["system", "user", "assistant"],
              "messages are system/user/assistant in order",
              f"got {[m['role'] for m in msgs]}")
        check(msgs and msgs[2]["content"] == "Anakin Skywalker.",
              "the assistant turn is the teacher's answer")
        check(msgs and "search_titles" not in msgs[0]["content"]
              and "tool" not in msgs[0]["content"].lower(),
              "the student prompt does not mention tools the student will not have")

        # ---- 8. unsupported is kept and filterable ------------------------
        check(sft_eval and sft_eval[0]["unsupported"] == ["That"],
              "unsupported spans are preserved, not dropped",
              f"got {sft_eval[0]['unsupported']!r}" if sft_eval else "no rows")
        check(sft_train and sft_train[0]["unsupported"] == [],
              "unflagged rows get an empty list, not null")

        # ---- 9. splits stay disjoint by source article --------------------
        q_train = {r["source_article"] for r in rows(out, "questions/train-*.parquet")}
        q_eval = {r["source_article"] for r in rows(out, "questions/eval-*.parquet")}
        check(q_train and q_eval and not (q_train & q_eval),
              "question splits share no source article",
              f"overlap: {q_train & q_eval}")
        check(rows(out, "questions/eval-*.parquet")[0]["evidence"],
              "questions carry their evidence span")

        # ---- 10. shipped metadata ------------------------------------------
        card = (out / "README.md").read_text(encoding="utf-8")
        check(card.startswith("---\nlicense:"), "dataset card opens with HF frontmatter")
        check("config_name: sft" in card, "card declares the sft config")
        sums = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        shipped = [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
        check(len(sums) == len(shipped),
              "SHA256SUMS covers every shipped file",
              f"{len(sums)} sums for {len(shipped)} files")

        # ---- 11. determinism ------------------------------------------------
        again = Path(tmp) / "dist2"
        run_packager(root, again, "--hf-repo", "rafa-rrayes/wookieelm-corpus")
        print()
        a = [ln.split()[0] for ln in
             (out / "SHA256SUMS").read_text().splitlines() if "parquet" in ln]
        b = [ln.split()[0] for ln in
             (again / "SHA256SUMS").read_text().splitlines() if "parquet" in ln]
        check(a == b and a, "two builds produce byte-identical shards")

        # ---- 12. sharding ----------------------------------------------------
        many = Path(tmp) / "dist3"
        pkg.DIST_DIR = many
        run_packager(root, many, "--shard-mb", "0")
        print()
        shards = sorted(many.glob("pretrain/wookieepedia-*.parquet"))
        n_articles = len(wook)
        check(len(shards) == n_articles,
              "a zero-byte budget puts every document in its own shard",
              f"got {len(shards)} shards for {n_articles} articles")
        check(all(f"-of-{n_articles:05d}.parquet" in p.name for p in shards),
              "shard names carry the of-N total",
              f"got {[p.name for p in shards]}")
        check(len(rows(many, "pretrain/wookieepedia-*.parquet")) == n_articles,
              "sharding loses no rows")

        # ---- 13. restricted, when asked for ---------------------------------
        restricted = Path(tmp) / "dist4"
        run_packager(root, restricted, "--include-restricted")
        print()
        check((restricted / "restricted").is_dir(),
              "--include-restricted writes dist/restricted/")
        check(len(rows(restricted, "restricted/books-*.parquet")) == 1,
              "the book becomes a row when explicitly requested")
        check("NOT FOR REDISTRIBUTION" in
              (restricted / "DATASHEET.md").read_text(encoding="utf-8"),
              "a restricted build is stamped in its datasheet")
        check("MUST NOT be uploaded" in
              (restricted / "README.md").read_text(encoding="utf-8"),
              "a restricted build is stamped in its dataset card")

    print()
    print(f"FAILURES: {failures}" if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

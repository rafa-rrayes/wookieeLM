#!/usr/bin/env python3
"""augment_corpus.py, without spending anything.

Every check here is about a way the augmenter can quietly produce *plausible
but wrong* training data, because that is the failure mode that survives to the
model: a date the source never contained, a rewrite that is about a different
subject, a doc whose continuity label got lost, a config typo that silently
reverts a threshold to its default.

The API is a httpx.MockTransport, so the full path -- pack, request, validate,
write -- runs offline and the assertions are about what landed on disk.

    uv run tests/test_augment.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ac = load("ac", "augment_corpus.py")
pkg = load("pkg", "package.py")


async def _no_backoff(*_args, **_kwargs) -> None:
    """Retries are under test; the sleep between them is not.

    Left in, the rejection cases below wait out real exponential backoff --
    four attempts each, several cases, minutes of a test that computes nothing.
    """


ac.pc._backoff = _no_backoff


# ---- Fixtures ----------------------------------------------------------------

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
- **Homeworld:** Shili

**Ahsoka Tano** was a Togruta female born on Shili who served as the Padawan
learner of Anakin Skywalker during the Clone Wars.

## Biography

Tano was assigned to Anakin Skywalker in 22 BBY. She left the Jedi Order in
19 BBY after being wrongly accused of bombing the Jedi Temple.

| Appearance | Year |
|---|---|
| The Clone Wars | 2008 |
| Rebels | 2014 |

## Legacy

She was remembered by the Rebel Alliance as Fulcrum.
"""

# A response that should survive every gate the fixture view applies.
GOOD = """Ahsoka Tano was a Togruta female born on Shili.

Ahsoka Tano became the Padawan learner of Anakin Skywalker in 22 BBY.

Ahsoka Tano left the Jedi Order in 19 BBY."""


# Long enough to clear the strictest min_body_chars in the shipped config, and
# deliberately dateless, so a check about continuity or about a missing date is
# not answered by the length gate instead.
PADDING = "\n\n## Later life\n\n" + (
    "She travelled widely and was known to many across the galaxy. " * 20)


def make_doc(tmp: Path, text: str = ARTICLE, name: str = "Ahsoka_Tano.md"):
    path = tmp / "corpus" / "wookieepedia" / "A" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path, ac.pc.split_document(path, text)


class Args:
    """The handful of argparse attributes the run path reads."""
    concurrency = 4
    timeout = 30.0
    model = "test-model"


def responder(*bodies: str):
    """A MockTransport handler returning `bodies` in order, then repeating the last."""
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = bodies[min(len(seen) - 1, len(bodies) - 1)]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": body}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                      "prompt_cache_hit_tokens": 80},
        })

    return handle, seen


def run_one(view, doc, handler, model: str = "test-model") -> tuple[str | None, str]:
    """-> (text, error). Drives generate() against a mocked API."""
    async def go():
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="https://example.invalid") as client:
            stats = ac.Stats()
            packed = ac.pack_input(doc, view)
            try:
                return await ac.generate(client, view, doc, packed, model, stats,
                                         asyncio.Semaphore(2)), ""
            except ac.AugmentError as e:
                return None, str(e)

    return asyncio.run(go())


# ---- Checks ------------------------------------------------------------------

def main() -> int:
    failures = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if ok:
            print(f"  ok    {label}")
        else:
            failures += 1
            print(f"  FAIL  {label}{'  -- ' + detail if detail else ''}")

    views = {v.name: v for v in ac.parse_views()}

    # ---- 1. the shipped config actually loads ---------------------------------
    print("\nconfig")
    check("augment_views.toml parses", len(views) >= 5, f"{len(views)} views")
    check("entityflip is configured", "entityflip" in views)
    check("every view has a distinct out_dir",
          len({v.out_dir for v in views.values()}) == len(views))
    check("no view writes into the source tree",
          all(v.out_dir != "wookieepedia" for v in views.values()))
    # A placeholder in a system prompt would make it per-article, which costs
    # the prefix cache -- ~50x on input for that view.
    check("system prompts are constant",
          all("{" not in v.system or "}" not in v.system for v in views.values()))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        config = tmp / "views.toml"
        # Point the module at the fixture tree before anything derives a path
        # from it: provenance and output paths are both relative to these.
        ac.CORPUS_DIR = tmp / "corpus"
        ac.SOURCE_DIR = ac.CORPUS_DIR / "wookieepedia"

        def write_config(body: str):
            config.write_text(body, encoding="utf-8")
            return config

        base = '''
[defaults]
user_template = "{title} {continuity} {passage}"

[views.demo]
out_dir = "demo1"
system = "Rewrite it."
'''
        check("a minimal view loads", len(ac.load_views(write_config(base))) == 1)

        for label, body, want in [
            ("unknown key is rejected",
             base + "\nmin_bdy_chars = 10\n", "unknown key"),
            ("bad type is rejected",
             base + '\nmin_body_chars = "lots"\n', "expected a number"),
            ("placeholder in system is rejected",
             base.replace('system = "Rewrite it."',
                          'system = "Rewrite {title}."'), "placeholder"),
            ("bad regex is rejected",
             base + "\nrequire_pattern = '('\n", "bad regex"),
        ]:
            try:
                ac.load_views(write_config(body))
                check(label, False, "no ConfigError raised")
            except ac.ConfigError as e:
                check(label, want in str(e), str(e))

        try:
            ac.load_views(write_config(base), ["nope"])
            check("unknown --views name is rejected", False)
        except ac.ConfigError as e:
            check("unknown --views name is rejected", "unknown view" in str(e))

        # ---- 2. what gets sent ------------------------------------------------
        print("\npacking")
        src, doc = make_doc(tmp)
        view = views["atomicfacts"]
        packed = ac.pack_input(doc, view)
        check("infobox is included", "Homeworld:** Shili" in packed)
        # The lead paragraph sits between the infobox and the first heading;
        # an infobox regex that ran to the next heading would swallow it.
        check("the lead paragraph is included",
              "was a Togruta female born on Shili" in packed)
        check("the release table is dropped", "| The Clone Wars |" not in packed)
        check("headings survive", "## Legacy" in packed)

        long_body = "\n\n".join(
            [f"## Section {i}\n\n" + ("Sentence about the Jedi. " * 60)
             for i in range(40)])
        condensed = ac.condense(long_body, 4000)
        check("condense respects the limit", len(condensed) <= 4000,
              f"{len(condensed)}")
        # The failure that matters: blind truncation would keep only the first
        # sections, so a timeline of a long life would stop halfway.
        check("condense keeps the last section too", "## Section 39" in condensed)

        # ---- 3. what comes back -----------------------------------------------
        print("\nvalidation")
        cases = [
            ("a good rewrite passes", GOOD, None),
            ("empty is rejected", "", "empty response"),
            ("a refusal is rejected", "I'm sorry, I can't help with that.",
             "refusal"),
            ("meta-talk is rejected",
             GOOD + "\n\nThe passage does not say where she went.", "meta-talk"),
            ("an invented date is rejected",
             GOOD.replace("19 BBY", "17 BBY"), "date not in source"),
            ("a different subject is rejected",
             "Grand Admiral Thrawn commanded the Chimaera in 5 ABY."
             " Thrawn was a Chiss officer of considerable reputation."
             " Thrawn served the Galactic Empire faithfully." * 3,
             "subject missing"),
            ("a too-long rewrite is rejected", ("Ahsoka Tano. " * 4000),
             "length ratio"),
        ]
        for label, body, want in cases:
            handler, _ = responder(body)
            text, err = run_one(view, doc, handler)
            if want is None:
                check(label, text is not None, err)
            else:
                check(label, text is None and want in err, err or "accepted")

        # A date written differently is the same date, and rejecting it would
        # cost a whole extra request every time a model drops a thousands comma.
        check("comma-insensitive dates",
              ac.dates("25,797 BBY") == ac.dates("25797 BBY"))

        # ---- 4. retries -------------------------------------------------------
        print("\nretries")
        handler, seen = responder("I'm sorry, I can't.", GOOD)
        text, err = run_one(view, doc, handler)
        check("a bad first response is retried", text is not None, err)
        check("the retry lowers temperature",
              len(seen) == 2 and seen[1]["temperature"] < seen[0]["temperature"],
              str([s.get("temperature") for s in seen]))
        check("thinking is off", seen[0]["thinking"]["type"] == "disabled")

        # ---- 5. the document that gets written --------------------------------
        print("\noutput")
        out = ac.build_output(view, doc, GOOD, "test-model")
        meta = pkg.parse_frontmatter(out.split("---\n")[1] + "---\n")
        check("continuity survives", meta.get("continuity") == "canon")
        check("categories survive", meta.get("categories") == ["Females", "Togruta"])
        check("the view is recorded", meta.get("augment_view") == "atomicfacts")
        check("the source path is recorded",
              meta.get("augmented_from") == "wookieepedia/A/Ahsoka_Tano.md")
        check("the model is recorded", meta.get("augment_model") == "test-model")
        check("the title carries the view",
              str(meta.get("title", "")).startswith("Ahsoka Tano ("))
        # The suffixed title would not resolve, so the URL is written explicitly.
        check("attribution URL is the real article",
              meta.get("url") == "https://starwars.fandom.com/wiki/Ahsoka_Tano")
        check("the body is the rewrite", out.rstrip().endswith("19 BBY."))

        # package.py has to be able to read it back as a pretrain row.
        parsed = ac.pc.split_document(Path("x.md"), out)
        check("output re-parses", parsed.body.startswith("Ahsoka Tano was"))

        # ---- 6. eligibility ---------------------------------------------------
        print("\neligibility")
        check("a full article is eligible", ac.eligible(doc, view) is None)
        _, stub = make_doc(tmp, ARTICLE.split("## Biography")[0], "Stub.md")
        check("a stub is skipped by a strict view",
              ac.eligible(stub, views["dialogue"]) == "too short for view")
        _, real = make_doc(tmp, ARTICLE.replace("continuity: canon",
                                                "continuity: real-world")
                           + PADDING, "Real.md")
        check("a real-world subject gets no in-universe entry",
              ac.eligible(real, views["inuniverse"]) == "continuity excluded")
        _, undated = make_doc(tmp, ARTICLE.replace("22 BBY", "then")
                              .replace("19 BBY", "later") + PADDING,
                              "Undated.md")
        check("an undated article gets no timeline",
              ac.eligible(undated, views["timeline"]) == "pattern not found")

        # ---- 7. resume --------------------------------------------------------
        print("\nresume")
        out_path = ac.output_path_for(view, src)
        check("output mirrors the source path",
              out_path.as_posix().endswith("atomicfacts1/A/Ahsoka_Tano.md"),
              out_path.as_posix())
        ac.pc.atomic_write(out_path, out)
        check("an existing output is detectable", out_path.exists())

        # ---- 8. package.py sees the new trees ---------------------------------
        print("\npackaging")
        names = {s.name for s in pkg.augmented_sources()}
        check("every view is a packaged source",
              names == {v.out_dir for v in views.values()},
              f"{sorted(names)}")
        check("augmented sources are marked derived",
              all(s.derived_from == "wookieepedia" for s in pkg.augmented_sources()))
        check("augmented sources are redistributable",
              all(not s.restricted for s in pkg.augmented_sources()))

    print()
    print("FAILURES:", failures) if failures else print("all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

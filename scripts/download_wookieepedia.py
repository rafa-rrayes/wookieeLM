#!/usr/bin/env python3
"""Download Wookieepedia and save it as a Markdown corpus.

Four stages, all in this one script:

    1. download    the Fandom MediaWiki XML dump (~275 MB .7z) from S3
    2. extract     the .7z in-process (no `7z` CLI needed)
    3. continuity  fetch canon/Legends/non-canon title lists from the live API
    4. convert     every main-namespace page -> corpus/wookieepedia/<L>/<Title>.md

Each output file carries YAML frontmatter (title, continuity, source,
categories), an optional ``## Infobox`` section rendered from the page's
infobox template, and the cleaned article body. A ``manifest.md`` summarising
the continuity split is written at the end.

Requirements:
    pandoc on PATH   (brew install pandoc)

Usage:
    uv run scripts/download_wookieepedia.py                 # full run
    uv run scripts/download_wookieepedia.py --limit 200     # quick smoke test
    uv run scripts/download_wookieepedia.py --force         # reconvert existing files

Resume: rerun the same command. The dump download resumes byte-wise, the
continuity lists are cached, and already-converted pages are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import mwparserfromhell
import py7zr
from lxml import etree
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
WOOKIEEPEDIA_DIR = CORPUS_DIR / "wookieepedia"
CONTINUITY_DIR = REPO_ROOT / "continuity"
DUMP_DIR = REPO_ROOT / "dump"

# Fandom publishes every wiki's current-revision dump at a predictable S3 path:
# <first letter>/<first two letters>/<dbname>_pages_current.xml.7z
DUMP_URL = "https://s3.amazonaws.com/wikia_xml_dumps/s/st/starwars_pages_current.xml.7z"
API = "https://starwars.fandom.com/api.php"
UA = "wookieLM-research/1.0 (rafa@rayes.com.br)"


# ---- Stage 1: download ------------------------------------------------------

def download_dump(archive: Path, force: bool = False) -> Path:
    """Fetch the dump archive, resuming a partial file via a Range request."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    if force and archive.exists():
        archive.unlink()

    head = urllib.request.Request(DUMP_URL, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(head, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        remote_md5 = resp.headers.get("x-amz-meta-md5", "")

    have = archive.stat().st_size if archive.exists() else 0
    if total and have == total:
        print(f"skip   {archive.name} (complete, {total / 1e6:.0f} MB)")
        return archive
    if have > total:  # stale/corrupt partial from an older, larger dump
        archive.unlink()
        have = 0

    headers = {"User-Agent": UA}
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"resume {archive.name} at {have / 1e6:.0f} / {total / 1e6:.0f} MB")

    req = urllib.request.Request(DUMP_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        # A server that ignores our Range restarts the file from zero.
        mode = "ab" if (have and resp.status == 206) else "wb"
        if mode == "wb":
            have = 0
        with open(archive, mode) as fh, tqdm(
            total=total, initial=have, unit="B", unit_scale=True, desc="dump"
        ) as bar:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))

    size = archive.stat().st_size
    if total and size != total:
        sys.exit(f"download truncated: {size} != {total} bytes — rerun to resume")
    print(f"ok     {archive.name} ({size / 1e6:.0f} MB, upstream md5 {remote_md5})")
    return archive


# ---- Stage 2: extract -------------------------------------------------------

def extract_dump(archive: Path, dest_dir: Path, force: bool = False) -> Path:
    """Unpack the single .xml member of the .7z archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, "r") as z:
        members = [n for n in z.getnames() if n.endswith(".xml")]
        if not members:
            sys.exit(f"no .xml member inside {archive}")
        member = members[0]
        target = dest_dir / Path(member).name
        if target.exists() and not force:
            print(f"skip   {target.name} (already extracted, "
                  f"{target.stat().st_size / 1e9:.1f} GB)")
            return target
        print(f"extract {member} ...")
        z.extract(path=dest_dir, targets=[member])
    print(f"ok     {target.name} ({target.stat().st_size / 1e9:.1f} GB)")
    return target


# ---- Stage 3: continuity lists ----------------------------------------------

# Wookieepedia's era template files every in-universe article into exactly one
# of these maintenance categories. That is the only authoritative canon/Legends
# marker, and the dump-to-markdown conversion strips it — so pull it from the
# live API and stamp it into the frontmatter as we write.
CONTINUITY_CATEGORIES = {
    "legends": ["Category:Legends articles"],
    "canon": ["Category:Canon articles"],
    "non-canon": ["Category:Non-canon articles", "Category:Non-canon Legends articles"],
}


def fetch_category_members(category: str) -> list[str]:
    """Every namespace-0 page title in `category`, following cmcontinue."""
    titles: list[str] = []
    cmcontinue: str | None = None
    requests_made = 0
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "cmnamespace": "0",
            "cmtype": "page",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        url = f"{API}?{urllib.parse.urlencode(params)}"

        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.load(resp)
                break
            except Exception as exc:  # noqa: BLE001 - retry transient API/network errors
                wait = 2**attempt
                print(f"  retry {attempt + 1}/5 after {exc} (waiting {wait}s)")
                time.sleep(wait)
        else:
            raise RuntimeError(f"giving up on {category} after repeated failures")

        titles.extend(m["title"] for m in data["query"]["categorymembers"])
        requests_made += 1
        print(f"  {category}: {len(titles):,} titles ({requests_made} requests)", end="\r")

        cont = data.get("continue")
        if not cont:
            break
        cmcontinue = cont["cmcontinue"]
        time.sleep(0.1)  # be polite to the API
    print()
    return titles


def fetch_continuity(force: bool = False) -> dict[str, set[str]]:
    """Fetch (and cache to continuity/*.txt) the canon/Legends/non-canon splits."""
    CONTINUITY_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, set[str]] = {}
    for tag, categories in CONTINUITY_CATEGORIES.items():
        cache = CONTINUITY_DIR / f"{tag}_titles.txt"
        if cache.exists() and not force:
            titles = {ln.rstrip("\n") for ln in cache.open(encoding="utf-8") if ln.strip()}
            print(f"skip   {cache.name} (cached, {len(titles):,} titles)")
        else:
            titles = set()
            for category in categories:
                print(f"fetch  {category} ...")
                titles.update(fetch_category_members(category))
            cache.write_text("\n".join(sorted(titles)) + "\n", encoding="utf-8")
            print(f"ok     {cache.name} ({len(titles):,} titles)")
        out[tag] = titles
    return out


def classify(title: str, continuity: dict[str, set[str]]) -> str:
    """canon > legends > non-canon; anything unlisted is out-of-universe."""
    for tag in ("canon", "legends", "non-canon"):
        if title in continuity.get(tag, ()):
            return tag
    return "real-world"


# ---- Stage 4a: wikitext filtering -------------------------------------------

MAIN_NAMESPACE = 0

# Templates that are pure data containers — strip entirely rather than inline.
# The actual infobox content is extracted separately via find_infobox/render_infobox
# and rendered as a ``## Infobox`` section.
INFOBOX_TEMPLATE_PREFIXES = (
    "infobox", "char", "character", "starship", "planet", "species",
    "weapon", "vehicle", "organization", "battle", "event", "location",
    "book", "comic", "film", "tv", "episode", "game", "audio",
    "quote", "scroll box", "eras", "top", "bottom", "title", "youmay",
    "otheruses", "redirect", "see also",
)

# Templates that are noise wrappers — drop but keep their positional text args.
DROP_KEEP_ARGS = ("c", "cquote", "qt", "color")

# Wikilink prefixes that produce image/figure noise downstream.
FILE_LINK_PREFIXES = ("file:", "image:", "media:")

# Section headings to remove wholesale (lower-cased, exact match). These are
# out-of-universe metadata that adds noise to an in-universe LLM corpus.
# "Behind the scenes" is intentionally NOT here — it usually contains real prose.
BOILERPLATE_SECTIONS = {
    "appearances", "non-canon appearances", "sources", "non-canon sources",
    "appearances and sources", "notes and references", "references",
    "external links", "see also", "gallery", "alternate choices", "trivia",
}

# Pages whose cleaned body is below this many chars get dropped (mostly stubs
# or pages that were entirely boilerplate).
MIN_BODY_CHARS = 200

# Templates Wookieepedia routinely puts at the *top* of a page that are NOT
# data infoboxes — page-level notices, disambiguation hatnotes, era badges,
# tone/cleanup tags, etc. We skip them when scanning for the real infobox.
NON_INFOBOX_TEMPLATES = {
    "top", "bottom", "otheruses", "redirect", "youmay", "rhere", "doom",
    "multipleissues", "update", "tone", "conflicting", "cleanup", "expand",
    "stub", "merge", "image", "title", "see also", "eras", "scroll box",
    "quote", "dialogue", "cquote",
}

# Minimum named-parameter count for a template to be considered an infobox.
# Wookieepedia infoboxes routinely have 20+ named params; navigation/hatnote
# templates have 0-3. A threshold of 8 cleanly separates them.
MIN_INFOBOX_PARAMS = 8

# Field keys we never want to surface — image pointers, styling, selectors.
INFOBOX_JUNK_KEYS = {
    "image", "image1", "image2", "image3", "image4", "image5", "image6",
    "imagewidth", "imagebg", "imagebackground",
    "option1", "option2", "option3", "option4", "option5", "option6",
    "caption", "caption1", "caption2", "caption3",
    "hidep", "hideb", "hidec", "hided", "hidee", "hidef", "hideg",
    "type", "subtype", "bordercolor", "background", "headercolor",
    "color", "textcolor", "headerstyle",
}

# Pretty labels for common infobox keys. Anything not here falls back to a
# capitalized version of the key itself.
INFOBOX_KEY_LABELS = {
    "name": "Name", "homeworld": "Homeworld", "birth": "Born", "died": "Died",
    "death": "Died", "species": "Species", "gender": "Gender",
    "pronouns": "Pronouns", "height": "Height", "mass": "Mass",
    "weight": "Weight", "hair": "Hair", "haircolor": "Hair color",
    "eyes": "Eyes", "eyecolor": "Eye color", "skin": "Skin",
    "skincolor": "Skin color", "cyber": "Cybernetics",
    "cybernetics": "Cybernetics", "feathers": "Feathers", "scales": "Scales",
    "families": "Family", "family": "Family", "parents": "Parents",
    "partner": "Partner", "partners": "Partner", "spouse": "Spouse",
    "siblings": "Siblings", "children": "Children",
    "affiliation": "Affiliations", "affiliations": "Affiliations",
    "masters": "Masters", "master": "Master", "apprentices": "Apprentices",
    "apprentice": "Apprentice", "rank": "Rank", "position": "Position",
    "title": "Title", "weapon": "Weapon", "weapons": "Weapons", "era": "Era",
    "eras": "Era", "founder": "Founder", "founded": "Founded",
    "dissolved": "Dissolved", "leader": "Leader", "leaders": "Leaders",
    "headquarters": "Headquarters", "capital": "Capital",
    "language": "Language", "languages": "Language", "religion": "Religion",
    "designation": "Designation", "manufacturer": "Manufacturer",
    "designer": "Designer", "model": "Model", "class": "Class",
    "length": "Length", "width": "Width", "diameter": "Diameter",
    "crew": "Crew", "passengers": "Passengers", "armament": "Armament",
    "shielding": "Shielding", "hull": "Hull", "engines": "Engines",
    "hyperdrive": "Hyperdrive", "speed": "Speed", "maxspeed": "Max speed",
    "max speed": "Max speed", "role": "Role", "roles": "Roles",
    "battles": "Battles", "conflicts": "Conflicts", "missions": "Missions",
    "date": "Date", "location": "Location", "result": "Result",
    "casualties": "Casualties", "commanders": "Commanders",
    "forces": "Forces", "author": "Author", "authors": "Authors",
    "publisher": "Publisher", "released": "Released", "pages": "Pages",
    "isbn": "ISBN", "preceded by": "Preceded by",
    "followed by": "Followed by", "system": "System", "sector": "Sector",
    "region": "Region", "grid": "Grid coordinates",
    "rotation": "Rotation period", "orbital": "Orbital period",
    "atmosphere": "Atmosphere", "climate": "Climate", "gravity": "Gravity",
    "terrain": "Terrain", "water": "Surface water", "fauna": "Native fauna",
    "flora": "Native flora", "natives": "Native species",
    "populace": "Immigrated species", "population": "Population",
    "demonym": "Demonym", "government": "Government", "creator": "Creator",
    "produced": "Produced", "destroyed": "Destroyed", "raceuser": "Used by",
    "useruser": "Used by", "users": "Users", "owner": "Owner",
    "owners": "Owners", "operator": "Operator", "operators": "Operators",
    "members": "Members", "membership": "Members",
    "headofstate": "Head of state",
    "headofgovernment": "Head of government", "executive": "Executive",
    "judicial": "Judicial", "legislative": "Legislative",
}

# Citation template prefixes for infobox value cleanup — these emit source
# pointers, not human content.
CITATION_PREFIXES = (
    "cite", "ref", "sourcebook", "storycite", "encyclopediacite", "databank",
    "swe", "film", "tcw", "tcwa", "idwadventures", "vaderimmortal",
    "scroll", "comicstrip",
)

_BULLET_RE = re.compile(r"^(\*+)\s*(.*)$")
_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")


def _clean_infobox_value(value_node) -> str:
    """Turn a template param value (Wikicode) into plain readable text."""
    try:
        wc = mwparserfromhell.parse(str(value_node))
    except Exception:
        return str(value_node).strip()

    for tag in list(wc.filter_tags(matches=lambda t: str(t.tag).lower() == "ref")):
        try:
            wc.remove(tag)
        except ValueError:
            pass

    for link in list(wc.filter_wikilinks()):
        try:
            target = str(link.title).strip().lower()
        except Exception:
            continue
        if any(target.startswith(p) for p in FILE_LINK_PREFIXES):
            try:
                wc.remove(link)
            except ValueError:
                pass

    for tmpl in list(wc.filter_templates(recursive=True)):
        try:
            raw_name = str(tmpl.name).strip()
        except Exception:
            continue
        name_lower = raw_name.lower()

        # Apostrophe templates: {{'s}} -> 's, {{'}} -> '
        if raw_name.startswith("'"):
            try:
                wc.replace(tmpl, raw_name)
            except Exception:
                pass
            continue

        # {{C|note}} -> (note) — Wookieepedia uses this for parenthetical notes.
        if name_lower == "c":
            try:
                args = [str(p.value).strip() for p in tmpl.params if not p.showkey]
                text = " ".join(a for a in args if a)
                wc.replace(tmpl, f"({text})" if text else "")
            except Exception:
                pass
            continue

        if any(name_lower.startswith(p) for p in CITATION_PREFIXES):
            try:
                wc.remove(tmpl)
            except ValueError:
                pass
            continue

        try:
            wc.remove(tmpl)
        except ValueError:
            pass

    for link in list(wc.filter_wikilinks()):
        try:
            display = str(link.text) if link.text else str(link.title)
        except Exception:
            display = ""
        try:
            wc.replace(link, display)
        except Exception:
            pass

    for comment in list(wc.filter_comments()):
        try:
            wc.remove(comment)
        except ValueError:
            pass

    s = str(wc)
    s = _BR_RE.sub("\n", s)
    s = html.unescape(s)
    s = _HTML_TAG_RE.sub("", s)
    return "\n".join(_WS_RE.sub(" ", line).strip() for line in s.split("\n")).strip()


def _format_infobox_value(text: str) -> str | None:
    """Format a cleaned value as either an inline string or a nested bullet list."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return None

    if any(_BULLET_RE.match(l) for l in lines):
        out: list[str] = []
        for l in lines:
            m = _BULLET_RE.match(l)
            if m:
                depth = len(m.group(1))
                content = m.group(2).strip()
                if not content:
                    continue
                out.append(f"{'  ' * (depth - 1)}- {content}")
            elif out:
                out[-1] = out[-1] + " " + l.strip()
            else:
                out.append(f"- {l.strip()}")
        return "\n".join(out) if out else None

    if len(lines) == 1:
        return lines[0]

    # Multi-line but no wiki bullets (typically <br /> separators). Join short
    # entries with " / "; render long ones as a bullet list.
    if all(len(l) < 60 for l in lines):
        return " / ".join(lines)
    return "\n".join(f"- {l}" for l in lines)


def _pretty_infobox_key(key: str) -> str:
    k = key.strip().lower()
    if k in INFOBOX_KEY_LABELS:
        return INFOBOX_KEY_LABELS[k]
    return k.replace("_", " ").replace("-", " ").strip().capitalize()


def find_infobox(wikicode):
    """Return the first plausible infobox template in this wikicode, or None.

    Heuristic: the first top-level template whose name isn't a known
    navigation/hatnote and that has at least MIN_INFOBOX_PARAMS named
    parameters. Catches the diversity of Wookieepedia infobox templates
    (Character, CelestialBody, SpaceStation, IndividualShip, Government,
    Religion, Weapon, Battle, ...) without a fixed allowlist.
    """
    for tmpl in wikicode.filter_templates(recursive=False):
        try:
            name = str(tmpl.name).strip().lower()
        except Exception:
            continue
        if name in NON_INFOBOX_TEMPLATES:
            continue
        if len([p for p in tmpl.params if p.showkey]) >= MIN_INFOBOX_PARAMS:
            return tmpl
    return None


def render_infobox(tmpl) -> str | None:
    """Render an mwparserfromhell template as a Markdown ``## Infobox`` block."""
    body: list[str] = []
    seen: set[str] = set()
    for param in tmpl.params:
        if not param.showkey:
            continue
        key = str(param.name).strip().lower()
        if not key or key in INFOBOX_JUNK_KEYS or key in seen:
            continue
        seen.add(key)
        cleaned = _clean_infobox_value(param.value)
        if not cleaned:
            continue
        # Skip template boolean flags ("is_mobile=1", "hidden=yes", etc.).
        if cleaned.lower() in {"0", "1", "yes", "no", "true", "false"}:
            continue
        value_md = _format_infobox_value(cleaned)
        if not value_md:
            continue
        label = _pretty_infobox_key(key)
        if "\n" in value_md:
            body.append(f"- **{label}:**")
            body.extend(f"  {ln}" for ln in value_md.split("\n"))
        else:
            body.append(f"- **{label}:** {value_md}")
    if not body:
        return None
    return "## Infobox\n\n" + "\n".join(body) + "\n"


def clean_wikicode(wikicode) -> str:
    """Strip/transform templates and file links before pandoc sees them."""
    # File:/Image:/Media: wikilinks -> pandoc would emit <img>/<figure> noise.
    for link in list(wikicode.filter_wikilinks()):
        try:
            target = str(link.title).strip().lower()
        except Exception:
            continue
        if any(target.startswith(p) for p in FILE_LINK_PREFIXES):
            try:
                wikicode.remove(link)
            except ValueError:
                pass

    for template in list(wikicode.filter_templates(recursive=True)):
        try:
            raw_name = str(template.name).strip()
        except Exception:
            continue
        name = raw_name.lower()

        # Apostrophe templates: {{'s}} -> 's, {{'}} -> ', etc. Common on
        # Wookieepedia for possessives where a wikilink ends with a noun.
        if raw_name.startswith("'"):
            try:
                wikicode.replace(template, raw_name)
            except Exception:
                pass
            continue

        if any(name.startswith(prefix) for prefix in INFOBOX_TEMPLATE_PREFIXES):
            try:
                wikicode.remove(template)
            except ValueError:
                pass
            continue

        if name in DROP_KEEP_ARGS:
            try:
                args = [str(p.value).strip() for p in template.params if not p.showkey]
                wikicode.replace(template, " ".join(a for a in args if a))
            except Exception:
                pass
            continue

        if name.startswith(("cite", "ref")):
            try:
                wikicode.remove(template)
            except ValueError:
                pass

    for comment in list(wikicode.filter_comments()):
        try:
            wikicode.remove(comment)
        except ValueError:
            pass

    # Tags whose content is unparseable wikitext or pure noise — drop entirely.
    drop_tags = {"ref", "gallery", "imagemap", "mapframe", "timeline"}
    for tag in list(wikicode.filter_tags(matches=lambda t: str(t.tag).lower() in drop_tags)):
        try:
            wikicode.remove(tag)
        except ValueError:
            pass

    return str(wikicode)


_CATEGORY_TAG_RE = re.compile(r"\[\[Category:([^\]|]+)(?:\|[^\]]*)?\]\]", flags=re.IGNORECASE)


def extract_categories(wikitext: str) -> list[str]:
    """Pull [[Category:Foo]] tags out as a deduplicated list."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in _CATEGORY_TAG_RE.findall(wikitext):
        c = raw.strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def wikitext_to_markdown(wikitext: str) -> str:
    """Pipe cleaned wikitext through pandoc."""
    result = subprocess.run(
        ["pandoc", "-f", "mediawiki", "-t", "gfm", "--wrap=none"],
        input=wikitext.encode("utf-8"),
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pandoc failed: {result.stderr.decode('utf-8', errors='replace')[:200]}"
        )
    return result.stdout.decode("utf-8", errors="replace")


# ---- Stage 4b: markdown post-processing -------------------------------------

# Pandoc's mediawiki->gfm reader falls back to raw HTML for many features:
# wikilinks become <a class="wikilink">, files become <img>/<figure>, etc.
# We strip those here so the corpus is actual Markdown prose, not HTML.

_WIKILINK_A_RE = re.compile(r'<a\b[^>]*\bclass="wikilink"[^>]*>(.*?)</a>', flags=re.DOTALL)
_FIGURE_RE = re.compile(r"<figure\b[^>]*>.*?</figure>", flags=re.DOTALL | re.IGNORECASE)
_IMG_RE = re.compile(r"<img\b[^>]*/?>", flags=re.IGNORECASE)
_FIGCAPTION_RE = re.compile(r"<figcaption\b[^>]*>.*?</figcaption>", flags=re.DOTALL | re.IGNORECASE)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CATEGORY_A_RE = re.compile(r'<a\b[^>]*href="Category:[^"]*"[^>]*>[^<]*</a>', flags=re.IGNORECASE)
_EM_RE = re.compile(r"<em\b[^>]*>(.*?)</em>", flags=re.DOTALL | re.IGNORECASE)
_STRONG_RE = re.compile(r"<strong\b[^>]*>(.*?)</strong>", flags=re.DOTALL | re.IGNORECASE)
_SMALL_RE = re.compile(r"<small\b[^>]*>(.*?)</small>", flags=re.DOTALL | re.IGNORECASE)
_SUP_RE = re.compile(r"<sup\b[^>]*>(.*?)</sup>", flags=re.DOTALL | re.IGNORECASE)
_SUB_RE = re.compile(r"<sub\b[^>]*>(.*?)</sub>", flags=re.DOTALL | re.IGNORECASE)
_DOLLAR_ESCAPE_RE = re.compile(r"\\\$")
_EMPTY_BULLET_RE = re.compile(r"(?m)^[-*]\s*$\n?")
_HEADING_RE = re.compile(r"^(#+)\s+(.+?)\s*$")
_BLANKS_RE = re.compile(r"\n{3,}")


def clean_html_noise(md: str) -> str:
    """Strip leftover HTML pandoc emits when MW features don't map to GFM."""
    md = _FIGURE_RE.sub("", md)
    md = _FIGCAPTION_RE.sub("", md)
    md = _IMG_RE.sub("", md)
    md = _MD_IMAGE_RE.sub("", md)
    # Category anchors first — they also carry class="wikilink", so the generic
    # wikilink replacement below would otherwise keep their visible text behind.
    md = _CATEGORY_A_RE.sub("", md)
    md = _WIKILINK_A_RE.sub(lambda m: m.group(1), md)
    # Inline HTML pandoc falls back to when GFM can't represent the markup cleanly
    # (commonly inside tables, or where stripped wikilinks left orphaned italics).
    md = _EM_RE.sub(lambda m: f"*{m.group(1)}*", md)
    md = _STRONG_RE.sub(lambda m: f"**{m.group(1)}**", md)
    md = _SMALL_RE.sub(lambda m: m.group(1), md)
    md = _SUP_RE.sub(lambda m: m.group(1), md)
    md = _SUB_RE.sub(lambda m: m.group(1), md)
    md = _BR_RE.sub("\n", md)
    md = _DOLLAR_ESCAPE_RE.sub("$", md)
    md = html.unescape(md)
    md = _EMPTY_BULLET_RE.sub("", md)
    return _BLANKS_RE.sub("\n\n", md)


def strip_boilerplate_sections(md: str) -> str:
    """Remove out-of-universe sections (Appearances, Sources, External links, ...)."""
    out: list[str] = []
    skip_level: int | None = None
    for line in md.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip().lower()
            # Exit the skip region when a heading at or above the skipped level appears.
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if heading in BOILERPLATE_SECTIONS:
                skip_level = level
                continue  # drop the heading itself
        if skip_level is not None:
            continue
        out.append(line)
    return _BLANKS_RE.sub("\n\n", "\n".join(out)).strip() + "\n"


# ---- Stage 4c: paths & I/O --------------------------------------------------

_UNSAFE_CHARS_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(title: str) -> str:
    return _UNSAFE_CHARS_RE.sub("_", title).strip().replace(" ", "_")[:200]


def output_path(out_root: Path, title: str) -> Path:
    safe = sanitize_filename(title)
    shard = safe[0].upper() if safe and safe[0].isalnum() else "_"
    return out_root / shard / f"{safe}.md"


def yaml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_document(title: str, continuity: str | None, categories: list[str],
                   infobox_md: str | None, body: str) -> str:
    cats_yaml = "\n".join(f'  - "{yaml_escape(c)}"' for c in categories)
    return (
        "---\n"
        f'title: "{yaml_escape(title)}"\n'
        + (f"continuity: {continuity}\n" if continuity else "")
        + 'source: "Wookieepedia"\n'
        + (f"categories:\n{cats_yaml}\n" if categories else "")
        + "---\n\n"
        + (infobox_md + "\n" if infobox_md else "")
        + body
    )


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)  # atomic on POSIX, and on Windows since Python 3.3


# Pandoc costs ~36 ms just to start, which dwarfs the ~20 ms it spends actually
# converting an average 3.7 KB article. So we convert a whole batch of pages in
# a single pandoc call, separated by a sentinel paragraph that survives the
# mediawiki->gfm round trip untouched, and split the output back apart. Verified
# byte-identical to per-page conversion; any batch that fails to split cleanly
# falls back to one call per page.
SENTINEL = "ZQXWOOKIEEPAGEBREAKXQZ"

# Upper bound on the wikitext handed to a single pandoc call.
MAX_BATCH_CHARS = 1_500_000


def prepare_page(wikitext: str):
    """Extract categories + infobox and clean the wikitext, before pandoc."""
    categories = extract_categories(wikitext)
    infobox_md: str | None = None
    try:
        wikicode = mwparserfromhell.parse(wikitext)
        tmpl = find_infobox(wikicode)
        if tmpl is not None:
            infobox_md = render_infobox(tmpl)
            # Drop it from the body so it isn't duplicated alongside the
            # rendered Markdown infobox section.
            try:
                wikicode.remove(tmpl)
            except ValueError:
                pass
        cleaned = clean_wikicode(wikicode)
    except Exception:
        # Fall back to raw wikitext if parsing/cleanup fails — pandoc may still
        # produce usable output, and one bad page shouldn't block the pipeline.
        cleaned = wikitext
    return categories, infobox_md, cleaned


def pandoc_batch(chunks: list[str]) -> list[str] | None:
    """One pandoc call for many pages. None if the batch can't be split back."""
    if any(SENTINEL in c for c in chunks):
        return None
    try:
        out = wikitext_to_markdown(f"\n\n{SENTINEL}\n\n".join(chunks))
    except Exception:
        return None
    parts = out.split(SENTINEL)
    return parts if len(parts) == len(chunks) else None


# A small number of pages carry wikitext pandoc's reader rejects outright:
# orphaned </nowiki> or </ref> closers left behind by template stripping, and
# table rows with malformed attribute soup. Applied only as a retry after a page
# has already failed, so pages that convert cleanly are never touched.
_STRAY_NOWIKI_RE = re.compile(r"</?nowiki\s*/?>", flags=re.IGNORECASE)
_STRAY_REF_RE = re.compile(r"</?ref\b[^>]*/?>", flags=re.IGNORECASE)
_BAD_ROW_ATTRS_RE = re.compile(r'(?m)^(\|-|\{\|)[^\n]*$')


def sanitize_malformed(wikitext: str) -> str:
    """Last-ditch cleanup for wikitext pandoc refuses to parse."""
    s = _STRAY_NOWIKI_RE.sub("", wikitext)
    s = _STRAY_REF_RE.sub("", s)
    # Reduce table-open/row lines to their bare marker, dropping attribute soup.
    return _BAD_ROW_ATTRS_RE.sub(lambda m: m.group(1), s)


def pandoc_page(chunk: str) -> str | None:
    """Convert one page, retrying once with aggressive cleanup on failure."""
    try:
        return wikitext_to_markdown(chunk)
    except Exception:
        pass
    try:
        return wikitext_to_markdown(sanitize_malformed(chunk))
    except Exception:
        return None


def process_batch(batch: list[tuple[str, str]]):
    """Worker: clean every page, convert them in one pandoc call, post-process.

    Returns a list of (title, categories, infobox_md, body, error). On success
    error is None; on a recoverable empty page error == "empty_after_clean";
    otherwise error holds a short exception string.
    """
    prepared: list[tuple[str, list[str], str | None, str | None, str | None]] = []
    for title, wikitext in batch:
        try:
            categories, infobox_md, cleaned = prepare_page(wikitext)
            prepared.append((title, categories, infobox_md, cleaned, None))
        except Exception as e:
            prepared.append((title, [], None, None, f"{type(e).__name__}: {e}"))

    live = [i for i, p in enumerate(prepared) if p[4] is None]
    chunks = [prepared[i][3] for i in live]
    rendered = pandoc_batch(chunks) if chunks else []
    if rendered is None:  # sentinel collision or a page pandoc choked on
        rendered = [pandoc_page(c) for c in chunks]

    md_by_index = dict(zip(live, rendered))
    results = []
    for i, (title, categories, infobox_md, _cleaned, err) in enumerate(prepared):
        if err is not None:
            results.append((title, categories, None, None, err))
            continue
        md = md_by_index.get(i)
        if md is None:
            results.append((title, categories, None, None, "pandoc_failed"))
            continue
        try:
            markdown = strip_boilerplate_sections(clean_html_noise(md))
        except Exception as e:
            results.append((title, categories, None, None, f"{type(e).__name__}: {e}"))
            continue
        if len(markdown.strip()) < MIN_BODY_CHARS:
            results.append((title, categories, None, None, "empty_after_clean"))
        else:
            results.append((title, categories, infobox_md, markdown, None))
    return results


def dump_namespace(dump_xml: Path) -> str:
    """Read the export schema's XML namespace out of the file header."""
    with open(dump_xml, "rb") as fh:
        head = fh.read(4096).decode("utf-8", errors="replace")
    m = re.search(r'xmlns="([^"]+)"', head)
    if not m:
        sys.exit(f"no xmlns found in the first 4 KB of {dump_xml}")
    return f"{{{m.group(1)}}}"


def iter_pages(dump_xml: Path):
    """Yield (title, wikitext) for every main-namespace, non-redirect page.

    lxml's C iterparse walks the 1.9 GB dump ~350x faster than a pure-Python
    MediaWiki reader, which matters because this loop is single-threaded and
    has to keep every conversion worker fed.
    """
    ns = dump_namespace(dump_xml)
    context = etree.iterparse(str(dump_xml), events=("end",), tag=f"{ns}page",
                              huge_tree=True)
    for _, el in context:
        try:
            if el.findtext(f"{ns}ns") != str(MAIN_NAMESPACE):
                continue
            if el.find(f"{ns}redirect") is not None:
                continue
            title = el.findtext(f"{ns}title")
            revisions = el.findall(f"{ns}revision")
            if not title or not revisions:
                continue
            text = revisions[-1].findtext(f"{ns}text")  # last revision wins
            if not text:
                continue
            yield title, text
        finally:
            # Release the parsed subtree, and any preceding siblings the parser
            # is still holding, or the whole dump accumulates in memory.
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]


def convert_dump(dump_xml: Path, out_root: Path, continuity: dict[str, set[str]],
                 workers: int, limit: int | None, force: bool, log_path: Path,
                 batch_size: int) -> Counter:
    """Stream the dump through a process pool, writing one .md per page."""
    out_root.mkdir(parents=True, exist_ok=True)
    converted = failed = resumed = empty = 0
    tags: Counter[str] = Counter()
    max_inflight = workers * 3
    progress = tqdm(unit="pages", desc="convert")

    def handle_result(fut: cf.Future) -> None:
        nonlocal converted, failed, empty
        for title, categories, infobox_md, body, err in fut.result():
            progress.update(1)
            if err == "empty_after_clean":
                empty += 1
            elif err is not None:
                failed += 1
                errors_log.write(f"{title}\t{err}\n")
            else:
                tag = classify(title, continuity) if continuity else None
                try:
                    atomic_write(output_path(out_root, title),
                                 build_document(title, tag, categories, infobox_md, body))
                    converted += 1
                    if tag:
                        tags[tag] += 1
                except Exception as e:
                    failed += 1
                    errors_log.write(f"{title}\twrite_error\t{type(e).__name__}: {e}\n")
        errors_log.flush()
        progress.set_postfix(ok=converted, fail=failed, resume=resumed, empty=empty)

    with open(log_path, "a", encoding="utf-8") as errors_log, \
         cf.ProcessPoolExecutor(max_workers=workers) as pool:

        pending: set[cf.Future] = set()
        batch: list[tuple[str, str]] = []
        batch_chars = 0

        def submit(b: list[tuple[str, str]]) -> None:
            nonlocal pending
            # Throttle: drain completed futures before queueing more.
            while len(pending) >= max_inflight:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    handle_result(fut)
            pending.add(pool.submit(process_batch, b))

        for title, wikitext in iter_pages(dump_xml):
            if limit and converted + len(batch) >= limit:
                break

            target = output_path(out_root, title)
            if (not force) and target.exists() and target.stat().st_size > 0:
                resumed += 1
                progress.update(1)
                continue

            batch.append((title, wikitext))
            batch_chars += len(wikitext)
            # Cap by bytes as well as count: article sizes span three orders of
            # magnitude, and one 500 KB page would otherwise stall 99 small ones.
            if len(batch) >= batch_size or batch_chars >= MAX_BATCH_CHARS:
                submit(batch)
                batch = []
                batch_chars = 0

        if batch:
            submit(batch)
        for fut in cf.as_completed(pending):  # drain
            handle_result(fut)

    progress.close()
    print(f"\nconverted={converted}  failed={failed}  resumed={resumed}  empty={empty}")
    print(f"errors logged to: {log_path}")
    return tags


# ---- Manifest ---------------------------------------------------------------

def write_manifest(out_root: Path) -> None:
    """Tally the continuity field across every written file into manifest.md."""
    counts: Counter[str] = Counter()
    for f in out_root.rglob("*.md"):
        if f.name == "manifest.md":
            continue
        with f.open(encoding="utf-8") as fh:
            for line in (fh.readline() for _ in range(5)):  # frontmatter only
                if line.startswith("continuity:"):
                    counts[line.split(":", 1)[1].strip()] += 1
                    break

    total = sum(counts.values())
    if not total:
        return
    rows = [(tag, f"{n:,}", f"{n / total * 100:.1f}%") for tag, n in counts.most_common()]
    rows.append(("Total", f"{total:,}", "100%"))

    def sep(left: str, mid: str, right: str) -> str:
        return left + "─" * 12 + mid + "─" * 9 + mid + "─" * 7 + right

    out = [sep("┌", "┬", "┐"),
           f"│ {'Continuity':<10} │ {'Files':^7} │ {'Share':^5} │",
           sep("├", "┼", "┤")]
    for i, (tag, n, share) in enumerate(rows):
        out.append(f"│ {tag:<10} │ {n:>7} │ {share:>5} │")
        out.append(sep("├", "┼", "┤") if i < len(rows) - 1 else sep("└", "┴", "┘"))

    table = "\n".join(out)
    # Leading blank lines and no trailing newline — matches the existing manifest.
    (out_root / "manifest.md").write_text("\n\n" + table, encoding="utf-8")
    print("\n" + table)


# ---- Main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=WOOKIEEPEDIA_DIR,
                    help="output directory (default: corpus/wookieepedia)")
    ap.add_argument("--dump-dir", type=Path, default=DUMP_DIR,
                    help="where the .7z/.xml live (default: dump/)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="conversion worker processes (default: cpu_count)")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="pages per pandoc invocation (default: 100)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N converted pages (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="re-download, re-extract and reconvert everything")
    ap.add_argument("--no-continuity", action="store_true",
                    help="skip the canon/Legends API fetch and the continuity: field")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the .7z after extracting (default: delete it)")
    ap.add_argument("--log", type=Path, default=REPO_ROOT / "conversion.log",
                    help="path to the per-page error log")
    args = ap.parse_args()

    if not shutil.which("pandoc"):
        sys.exit("pandoc not found on PATH — install it first (brew install pandoc).")

    print("== 1/4 download ==")
    archive = download_dump(args.dump_dir / Path(DUMP_URL).name, force=args.force)

    print("\n== 2/4 extract ==")
    dump_xml = extract_dump(archive, args.dump_dir, force=args.force)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)

    print("\n== 3/4 continuity ==")
    continuity: dict[str, set[str]] = {}
    if args.no_continuity:
        print("skip   (--no-continuity)")
    else:
        continuity = fetch_continuity(force=args.force)

    print("\n== 4/4 convert ==")
    convert_dump(dump_xml, args.out, continuity,
                 workers=args.workers, limit=args.limit, force=args.force,
                 log_path=args.log, batch_size=args.batch_size)

    if continuity:
        write_manifest(args.out)
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()

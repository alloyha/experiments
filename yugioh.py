#!/usr/bin/env python3
"""
Pocket Duel API crawler -> DuckDB.

Crawls:
  - /api/data/duelist          (full list, no search needed)
  - /api/view/duelist/{id}     (HTML detail per duelist)
  - /api/view/card/{id}        (HTML detail per card; no list endpoint exists,
                                 so we sweep sequential integer ids until we
                                 hit a run of consecutive misses)

  Note: pd.ygo.fm's "starter" list endpoint (/api/data/starter) was investigated
  but dropped -- it returns {"id": N, "text": N} placeholders (no real name),
  and every /api/view/starter/{id} guess 404'd. It doesn't appear to back a
  real browsable entity, so it isn't crawled here.

Design notes:
  - Every row keeps the raw HTML (`raw_html`) so nothing is lost even if the
    generic field-extraction heuristics below don't match this site's markup
    exactly. Re-run parsing later against `raw_html` without re-crawling.
  - `fields` is a JSON blob of whatever key/value pairs the generic extractor
    could find (dl/dt/dd, tables, label/value class patterns). Treat it as
    "best effort", not a stable schema -- confirm against real HTML and
    tighten `extract_fields()` accordingly.
  - Crawling is sequential with a polite delay by default. Bump concurrency
    only if you've confirmed that's acceptable.
  - All upserts are idempotent (ON CONFLICT ... DO UPDATE), so re-running the
    crawler is safe and will just refresh `crawled_at` / content.

Usage:
    python3 pocketduel_crawler.py inspect card 1
    python3 pocketduel_crawler.py duelists
    python3 pocketduel_crawler.py cards --start-id 1 --max-consecutive-missing 25
    python3 pocketduel_crawler.py all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import duckdb
import requests
from bs4 import BeautifulSoup

DEFAULT_BASE_URL = "https://pd.ygo.fm"
DEFAULT_DB_PATH = "pocketduel.duckdb"
USER_AGENT = "pocketduel-crawler/0.1 (+personal research tool)"


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class FetchResult:
    def __init__(self, url: str, status: int, text: Optional[str], error: Optional[str] = None):
        self.url = url
        self.status = status
        self.text = text
        self.error = error

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.text is not None and self.error is None

    @property
    def missing(self) -> bool:
        return self.status == 404


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/html"})
    return s


def fetch(session: requests.Session, url: str, retries: int = 5, backoff: float = 2.0, timeout: float = 10.0) -> FetchResult:
    last_error = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 404:
                return FetchResult(url, 404, None)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = backoff * (attempt + 1)
                else:
                    wait = backoff * (attempt + 1)
                last_error = "HTTP 429"
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(backoff * (attempt + 1))
                continue
            resp.raise_for_status()
            return FetchResult(url, resp.status_code, resp.text)
        except requests.RequestException as e:
            last_error = str(e)
            time.sleep(backoff * (attempt + 1))
    return FetchResult(url, -1, None, error=last_error)


# --------------------------------------------------------------------------
# Generic HTML field extraction (best-effort -- tighten once real HTML is seen)
# --------------------------------------------------------------------------

def extract_title(soup: BeautifulSoup) -> Optional[str]:
    for sel in ("h1", "h2", ".title", ".name", "title"):
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            return el.get_text(strip=True)
    return None


def extract_fields(soup: BeautifulSoup) -> dict:
    """Try several common HTML patterns for label/value pairs and merge results."""
    fields: dict = {}

    # Pattern 1: definition lists <dl><dt>Label</dt><dd>Value</dd></dl>
    for dl in soup.select("dl"):
        dts = dl.select("dt")
        dds = dl.select("dd")
        for dt, dd in zip(dts, dds):
            key = dt.get_text(strip=True).rstrip(":")
            val = dd.get_text(strip=True)
            if key:
                fields[key] = val

    # Pattern 2: simple two-column tables <tr><th>Label</th><td>Value</td></tr>
    for tr in soup.select("table tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) == 2:
            key = cells[0].get_text(strip=True).rstrip(":")
            val = cells[1].get_text(strip=True)
            if key:
                fields.setdefault(key, val)

    # Pattern 3: label/value class conventions, e.g. <div class="label">X</div><div class="value">Y</div>
    labels = soup.select(".label, .field-label, .attr-label")
    for lab in labels:
        val_el = lab.find_next_sibling(class_=["value", "field-value", "attr-value"])
        if val_el:
            key = lab.get_text(strip=True).rstrip(":")
            val = val_el.get_text(strip=True)
            if key:
                fields.setdefault(key, val)

    # Pattern 4: data-* attributes on a root container, if present
    root = soup.select_one("[data-id], [data-card-id], [data-duelist-id], [data-starter-id]")
    if root:
        for k, v in root.attrs.items():
            if k.startswith("data-"):
                fields.setdefault(k, v)

    return fields


def extract_image(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    img = soup.select_one("img")
    if img and img.get("src"):
        src = img["src"]
        if src.startswith("http"):
            return src
        return base_url.rstrip("/") + "/" + src.lstrip("/")
    return None


CARD_REF_RE = re.compile(r"^#(\d+)\s*-\s*(.+)$")

# table element id -> our key name, based on observed /api/view/duelist/{id} markup
DUELIST_DROP_TABLES = {
    "decktable": "deck",
    "sapowtable": "sapow",
    "bcdpowtable": "bcdpow",
    "satectable": "satec",
}


def parse_drop_table(table) -> list:
    rows = []
    if table is None:
        return rows
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) != 2:
            continue
        card_cell = cells[0].get_text(strip=True)
        prob_cell = cells[1].get_text(strip=True)
        m = CARD_REF_RE.match(card_cell)
        if m:
            rows.append({"card_id": int(m.group(1)), "card_name": m.group(2), "probability": prob_cell})
        else:
            rows.append({"raw": card_cell, "probability": prob_cell})
    return rows


def parse_duelist_html(entity_id: int, html: str, source_url: str, base_url: str) -> "ParsedEntity":
    soup = BeautifulSoup(html, "html.parser")

    name = None
    name_el = soup.select_one("h6.cardname")
    if name_el:
        small = name_el.find("small")
        small_text = small.get_text(strip=True) if small else ""
        full_text = name_el.get_text(" ", strip=True)
        name = full_text[: -len(small_text)].strip() if small_text and full_text.endswith(small_text) else full_text

    drops = {}
    for table_id, key in DUELIST_DROP_TABLES.items():
        drops[key] = parse_drop_table(soup.find("table", id=table_id))

    return ParsedEntity(
        id=entity_id,
        name=name,
        fields={"drops": drops},
        image_url=extract_image(soup, base_url),
        raw_html=html,
        source_url=source_url,
    )


@dataclass
class ParsedEntity:
    id: int
    name: Optional[str]
    fields: dict = field(default_factory=dict)
    image_url: Optional[str] = None
    raw_html: str = ""
    source_url: str = ""


def parse_view_html(entity_id: int, html: str, source_url: str, base_url: str) -> ParsedEntity:
    soup = BeautifulSoup(html, "html.parser")
    return ParsedEntity(
        id=entity_id,
        name=extract_title(soup),
        fields=extract_fields(soup),
        image_url=extract_image(soup, base_url),
        raw_html=html,
        source_url=source_url,
    )


# --------------------------------------------------------------------------
# DuckDB storage
# --------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY,
    name VARCHAR,
    image_url VARCHAR,
    fields JSON,
    raw_html VARCHAR,
    source_url VARCHAR,
    crawled_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS duelists (
    id INTEGER PRIMARY KEY,
    name VARCHAR,
    image_url VARCHAR,
    fields JSON,
    raw_html VARCHAR,
    source_url VARCHAR,
    crawled_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS crawl_log (
    entity_type VARCHAR,
    entity_id INTEGER,
    http_status INTEGER,
    status VARCHAR,   -- 'ok' | 'missing' | 'error'
    error VARCHAR,
    crawled_at TIMESTAMP
);
"""

UPSERT_SQL_TEMPLATE = """
INSERT INTO {table} (id, name, image_url, fields, raw_html, source_url, crawled_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    image_url = EXCLUDED.image_url,
    fields = EXCLUDED.fields,
    raw_html = EXCLUDED.raw_html,
    source_url = EXCLUDED.source_url,
    crawled_at = EXCLUDED.crawled_at;
"""


def get_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(db_path)
    conn.execute(SCHEMA_SQL)
    return conn


def log_crawl(conn, entity_type: str, entity_id: int, result: FetchResult):
    status = "ok" if result.ok else ("missing" if result.missing else "error")
    conn.execute(
        "INSERT INTO crawl_log (entity_type, entity_id, http_status, status, error, crawled_at) VALUES (?, ?, ?, ?, ?, ?)",
        [entity_type, entity_id, result.status, status, result.error, datetime.now(timezone.utc)],
    )


def upsert_entity(conn, table: str, parsed: ParsedEntity):
    conn.execute(
        UPSERT_SQL_TEMPLATE.format(table=table),
        [
            parsed.id,
            parsed.name,
            parsed.image_url,
            json.dumps(parsed.fields),
            parsed.raw_html,
            parsed.source_url,
            datetime.now(timezone.utc),
        ],
    )


# --------------------------------------------------------------------------
# Crawl routines
# --------------------------------------------------------------------------

def crawl_list_backed(session, conn, base_url, table, entity_type, list_path, view_path_tmpl, delay, parser_fn=parse_view_html):
    """For list-backed entities (currently just duelists): fetch the full list once,
    then crawl each detail page. `table` is the storage/display name (e.g. 'duelists');
    `entity_type` is the singular value logged to crawl_log (e.g. 'duelist'), kept
    consistent with crawl_cards' 'card' and retry_failed's table_map keys."""
    list_url = f"{base_url}{list_path}"
    resp = fetch(session, list_url)
    if not resp.ok:
        print(f"[{table}] FAILED to fetch list endpoint {list_url}: status={resp.status} error={resp.error}", file=sys.stderr)
        return

    try:
        items = json.loads(resp.text)
        # /api/data endpoints observed so far return a JSON *string* containing
        # escaped JSON (see the double-encoded examples in the prompt) -- handle both.
        if isinstance(items, str):
            items = json.loads(items)
    except json.JSONDecodeError as e:
        print(f"[{table}] Could not parse list JSON: {e}", file=sys.stderr)
        return

    ids = [item["id"] for item in items if "id" in item]
    print(f"[{table}] list endpoint returned {len(ids)} entries")

    for entity_id in ids:
        view_url = f"{base_url}{view_path_tmpl.format(id=entity_id)}"
        result = fetch(session, view_url)
        log_crawl(conn, entity_type, entity_id, result)
        if result.ok:
            parsed = parser_fn(entity_id, result.text, view_url, base_url)
            upsert_entity(conn, table, parsed)
            print(f"[{table}] id={entity_id} OK name={parsed.name!r}")
        else:
            print(f"[{table}] id={entity_id} status={result.status} error={result.error}", file=sys.stderr)
        time.sleep(delay)


def crawl_cards(session, conn, base_url, start_id, end_id, max_consecutive_missing, delay):
    """No list endpoint for cards -> sweep sequential ids until enough consecutive failures.
    Note: pd.ygo.fm returns HTTP 500 (not 404) once ids go past the valid range, so both
    404 ('missing') and repeated errors count toward the stop streak -- either one reliably
    means we've swept past real data."""
    consecutive_failures = 0
    entity_id = start_id
    found_count = 0

    while True:
        if end_id is not None and entity_id > end_id:
            break

        view_url = f"{base_url}/api/view/card/{entity_id}"
        result = fetch(session, view_url)
        log_crawl(conn, "card", entity_id, result)

        if result.ok:
            parsed = parse_card_html(entity_id, result.text, view_url, base_url)
            upsert_entity(conn, "cards", parsed)
            print(f"[cards] id={entity_id} OK name={parsed.name!r}")
            consecutive_failures = 0
            found_count += 1
        else:
            consecutive_failures += 1
            reason = "missing" if result.missing else f"error (status={result.status} error={result.error})"
            print(f"[cards] id={entity_id} {reason} ({consecutive_failures}/{max_consecutive_missing} consecutive failures)")
            if consecutive_failures >= max_consecutive_missing:
                print(f"[cards] stopping sweep at id={entity_id}: {max_consecutive_missing} consecutive failures")
                break

        entity_id += 1
        time.sleep(delay)

    print(f"[cards] finished sweep: {found_count} cards found, last id tried = {entity_id - 1}")


def retry_failed(session, conn, base_url, delay, entity_type_filter=None):
    """Re-fetch every (entity_type, entity_id) whose most recent crawl_log entry was 'error'
    (e.g. exhausted 429 retries). Safe to run repeatedly -- it re-checks the latest status
    each time, so entries that succeed drop out of the retry set on the next pass."""
    query = """
        SELECT entity_type, entity_id FROM (
            SELECT entity_type, entity_id, status,
                   ROW_NUMBER() OVER (PARTITION BY entity_type, entity_id ORDER BY crawled_at DESC) AS rn
            FROM crawl_log
        )
        WHERE rn = 1 AND status = 'error'
    """
    if entity_type_filter:
        query += " AND entity_type = ?"
        rows = conn.execute(query, [entity_type_filter]).fetchall()
    else:
        rows = conn.execute(query).fetchall()

    if not rows:
        print("[retry-failed] nothing to retry")
        return

    print(f"[retry-failed] {len(rows)} entities to retry")

    table_map = {
        "card": ("cards", "/api/view/card/{id}", parse_card_html),
        "duelist": ("duelists", "/api/view/duelist/{id}", parse_duelist_html),
    }

    for entity_type, entity_id in rows:
        if entity_type not in table_map:
            continue
        table, path_tmpl, parser_fn = table_map[entity_type]
        url = f"{base_url}{path_tmpl.format(id=entity_id)}"
        result = fetch(session, url)
        log_crawl(conn, entity_type, entity_id, result)
        if result.ok:
            parsed = parser_fn(entity_id, result.text, url, base_url)
            upsert_entity(conn, table, parsed)
            print(f"[retry-failed][{entity_type}] id={entity_id} OK now")
        else:
            print(f"[retry-failed][{entity_type}] id={entity_id} still failing: status={result.status} error={result.error}", file=sys.stderr)
        time.sleep(delay)


CARD_TITLE_RE = re.compile(r"^#(\d+)\s*-\s*(.+)$")
PASSWORD_COST_RE = re.compile(r"^(\d+)\s*\(★?(\d+)\)$")


def _clean_ws(text: str) -> str:
    return " ".join(text.split())


def parse_card_html(entity_id: int, html: str, source_url: str, base_url: str) -> "ParsedEntity":
    soup = BeautifulSoup(html, "html.parser")

    name = None
    card_number = None
    stars = None
    name_el = soup.select_one("h6.cardname")
    if name_el:
        small = name_el.find("small")
        small_text = small.get_text(strip=True) if small else ""
        full_text = name_el.get_text(" ", strip=True)
        if small_text and full_text.endswith(small_text):
            full_text = full_text[: -len(small_text)].strip()
            stars = small_text.count("★") or None
        m = CARD_TITLE_RE.match(full_text)
        if m:
            card_number = int(m.group(1))
            name = m.group(2).strip()
        else:
            name = full_text

    # Overview table: <th>Label</th><td>Value</td> pairs
    overview = {}
    overview_table = soup.select_one("div#dtCard table")
    if overview_table:
        for tr in overview_table.select("tbody tr"):
            th, td = tr.find("th"), tr.find("td")
            if th and td:
                key = _clean_ws(th.get_text(strip=True))
                val = _clean_ws(td.get_text(" ", strip=True))
                overview[key] = val

    attack = defense = None
    ad_key = next((k for k in overview if "attack" in k.lower() and "defense" in k.lower()), None)
    if ad_key:
        parts = overview[ad_key].split("/")
        if len(parts) == 2:
            try:
                attack = int(parts[0].strip())
            except ValueError:
                pass
            try:
                defense = int(parts[1].strip())
            except ValueError:
                pass

    card_type = attribute = None
    type_key = next((k for k in overview if k.lower().startswith("type")), None)
    if type_key:
        raw_type_val = overview[type_key]
        parts = raw_type_val.split("/")
        if len(parts) == 2:
            # Monster row: "Type / Attribute" -> "Dragon / Light"
            card_type, attribute = parts[0].strip(), parts[1].strip()
        else:
            # Spell/Trap row: "Type" -> "Equip" / "Ritual" / "Normal" / "Trap" etc.
            # The single value IS the category here; there's no separate attribute.
            card_type = raw_type_val.strip()

    is_monster = ad_key is not None

    password = cost = None
    pw_key = next((k for k in overview if "password" in k.lower()), None)
    if pw_key:
        m2 = PASSWORD_COST_RE.match(overview[pw_key])
        if m2:
            password, cost = m2.group(1), int(m2.group(2))
        else:
            password = overview[pw_key]

    def parse_ref_list(table_id):
        items = []
        table = soup.find("table", id=table_id)
        if table:
            for tr in table.select("tbody tr"):
                td = tr.find("td")
                if not td:
                    continue
                text = td.get_text(strip=True)
                m3 = CARD_REF_RE.match(text)
                items.append({"card_id": int(m3.group(1)), "card_name": m3.group(2)} if m3 else {"raw": text})
        return items

    equip_materials = parse_ref_list("equiptable")
    ritual_materials = parse_ref_list("ritualtable")

    drops = []
    drop_table = soup.find("table", id="droptable")
    if drop_table:
        for tr in drop_table.select("tbody tr"):
            cells = tr.find_all("td")
            if len(cells) == 3:
                drops.append({
                    "duelist": cells[0].get_text(strip=True),
                    "rank": cells[1].get_text(strip=True),
                    "probability": cells[2].get_text(strip=True),
                })

    fields = {
        "card_number": card_number,
        "stars": stars,
        "description": overview.get("Description"),
        "is_monster": is_monster,
        "attack": attack,
        "defense": defense,
        "type": card_type,
        "attribute": attribute,
        "guardian_stars": overview.get("Guardian Stars"),
        "password": password,
        "cost": cost,
        "equip_materials": equip_materials,
        "ritual_materials": ritual_materials,
        "drops": drops,
        "overview_raw": overview,
    }

    return ParsedEntity(
        id=entity_id,
        name=name,
        fields=fields,
        image_url=extract_image(soup, base_url),
        raw_html=html,
        source_url=source_url,
    )


def inspect_one(session, base_url, entity_type, entity_id):
    url = f"{base_url}/api/view/{entity_type}/{entity_id}"
    result = fetch(session, url)
    print(f"URL: {url}")
    print(f"Status: {result.status}")
    if result.error:
        print(f"Error: {result.error}")
    if result.text:
        print("---- RAW HTML ----")
        print(result.text)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    # Shared flags (--base-url, --db, --delay) live on this parent parser and
    # are inherited by every subcommand, so they can be passed either before
    # or after the subcommand name, e.g. both of these work:
    #   yugioh.py --delay 0.25 all
    #   yugioh.py all --delay 0.25
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default=DEFAULT_BASE_URL)
    common.add_argument("--db", default=DEFAULT_DB_PATH)
    common.add_argument("--delay", type=float, default=0.25, help="Seconds to sleep between requests")

    parser = argparse.ArgumentParser(description="Pocket Duel API crawler -> DuckDB", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("duelists", help="Crawl full duelist list + detail pages", parents=[common])

    p_cards = sub.add_parser("cards", help="Sweep sequential card ids", parents=[common])
    p_cards.add_argument("--start-id", type=int, default=1)
    p_cards.add_argument("--end-id", type=int, default=None)
    p_cards.add_argument("--max-consecutive-missing", type=int, default=20)

    sub.add_parser("all", help="Crawl duelists and cards", parents=[common])

    p_retry = sub.add_parser("retry-failed", help="Re-fetch entities whose last crawl attempt errored out (e.g. 429s)", parents=[common])
    p_retry.add_argument("--entity-type", choices=["card", "duelist"], default=None)

    p_inspect = sub.add_parser("inspect", help="Fetch and print raw HTML for one entity (no DB write)", parents=[common])
    p_inspect.add_argument("entity_type", choices=["card", "duelist"])
    p_inspect.add_argument("entity_id", type=int)

    args = parser.parse_args()
    session = make_session()

    if args.command == "inspect":
        inspect_one(session, args.base_url, args.entity_type, args.entity_id)
        return

    conn = get_conn(args.db)

    if args.command in ("duelists", "all"):
        crawl_list_backed(session, conn, args.base_url, "duelists", "duelist", "/api/data/duelist", "/api/view/duelist/{id}", args.delay, parser_fn=parse_duelist_html)

    if args.command == "cards":
        crawl_cards(session, conn, args.base_url, args.start_id, args.end_id, args.max_consecutive_missing, args.delay)
    elif args.command == "all":
        crawl_cards(session, conn, args.base_url, 1, None, 20, args.delay)
    elif args.command == "retry-failed":
        retry_failed(session, conn, args.base_url, args.delay, entity_type_filter=args.entity_type)

    print("\nSummary:")
    for table in ("cards", "duelists"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()


if __name__ == "__main__":
    main()


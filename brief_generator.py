import os
import random
import re
import smtplib
import sqlite3
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

import feedparser
import requests
from google import genai

# --- Configuration ---
DB_PATH = Path("data/seen_items.db")
OUTPUT_DIR = Path("briefs")
LOOKBACK_DAYS = 21  # Wider window ensures fresh papers across all 5 instruments

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "researcher@example.com")

# 5 Core Pharmaceutical Lab Instruments / Techniques
INSTRUMENT_QUERIES = {
    "Liquid Chromatography (HPLC / UPLC)": (
        '(HPLC OR UPLC OR "liquid chromatography") AND '
        '(troubleshooting OR "peak tailing" OR "split peak" OR "split peaks" '
        'OR "baseline drift" OR carryover OR "column fouling" OR "method optimization") '
        'NOT (clinical OR patient OR wastewater)'
    ),
    "Gas Chromatography (GC / Headspace)": (
        '("gas chromatography" OR "GC-MS" OR "headspace GC") AND '
        '(troubleshooting OR "residual solvent" OR "residual solvents" OR "septum bleed" '
        'OR "column bleed" OR liner OR splitless OR carryover OR "peak distortion") '
        'NOT (soil OR atmosphere OR clinical)'
    ),
    "Spectrophotometry (UV-Vis / FTIR)": (
        '("UV-Vis" OR spectrophotometry OR "infrared spectroscopy" OR FTIR) AND '
        '(pharmaceutical OR assay OR calibration OR "stray light" OR cuvette '
        'OR baseline OR validation OR troubleshooting) '
        'NOT (astrophysics OR solar OR clinical)'
    ),
    "Atomic Absorption Spectroscopy (AAS)": (
        '("atomic absorption" OR "graphite furnace" OR GFAAS OR FAAS) AND '
        '(pharmaceutical OR "elemental impurity" OR "elemental impurities" '
        'OR "heavy metal" OR "heavy metals" OR "matrix interference" '
        'OR "matrix modifier" OR digestion OR validation) '
        'NOT (soil OR river OR sediment)'
    ),
    "Dissolution Testing (USP Apparatus 1 & 2)": (
        '("dissolution testing" OR "dissolution apparatus" OR "USP apparatus" OR "dissolution method") AND '
        '(deaeration OR coning OR paddle OR basket OR surfactant OR variability '
        'OR hydrodynamics OR troubleshooting) '
        'NOT (clinical OR patient)'
    ),
}

# Optional chromatography/spectroscopy RSS feeds to supplement OpenAlex
RSS_FEEDS = [
    # Add any preferred lab-practice RSS feeds here
]


# --- SQLite State & Deduplication ---
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_items (
            item_id TEXT PRIMARY KEY,
            instrument TEXT NOT NULL,
            title TEXT NOT NULL,
            published_date TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO metadata (key, value) VALUES ('issue_number', '1')"
    )
    conn.commit()


def is_seen(conn: sqlite3.Connection, item_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_items WHERE item_id = ?", (item_id,)
    ).fetchone()
    return row is not None


def mark_seen(conn: sqlite3.Connection, items: list[dict], status: str = "used") -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT OR IGNORE INTO seen_items (item_id, instrument, title, published_date, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                item["item_id"],
                item["instrument"],
                item["title"],
                item["published_date"],
                status,
                now,
            )
            for item in items
        ],
    )
    conn.commit()


def get_and_increment_issue_number(conn: sqlite3.Connection, increment: bool = False) -> int:
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'issue_number'"
    ).fetchone()
    current = int(row[0]) if row else 1
    if increment:
        conn.execute(
            "UPDATE metadata SET value = ? WHERE key = 'issue_number'",
            (str(current + 1),),
        )
        conn.commit()
    return current


# --- Data Ingestion ---
def reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions = [
        (pos, word)
        for word, pos_list in inverted_index.items()
        for pos in pos_list
    ]
    positions.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in positions)


def strip_html(raw_text: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", raw_text or "")
    return re.sub(r"\s+", " ", clean).strip()


def fetch_for_instrument(instrument: str, query: str, conn: sqlite3.Connection) -> list[dict]:
    from_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    filter_clauses = [
        f"title_and_abstract.search:{query}",
        f"from_publication_date:{from_date}",
        "has_abstract:true",
        "type:article|review",
        "language:en",
    ]
    params = {
        "filter": ",".join(filter_clauses),
        "sort": "publication_date:desc",
        "per-page": 10,
        "mailto": CONTACT_EMAIL,
    }
    resp = requests.get("https://api.openalex.org/works", params=params, timeout=30)
    if not resp.ok:
        print(f"OpenAlex API error ({resp.status_code}) for {instrument}: {resp.text}")
    resp.raise_for_status()

    unseen = []
    for work in resp.json().get("results", []):
        work_id = work.get("doi") or work.get("id")
        if not work_id or is_seen(conn, work_id):
            continue
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))[:1800]
        if len(abstract) < 120:
            continue
        unseen.append(
            {
                "item_id": work_id,
                "instrument": instrument,
                "title": strip_html(work.get("title", "Untitled")),
                "url": work.get("doi") or work.get("id"),
                "published_date": work.get("publication_date", ""),
                "summary": abstract,
            }
        )
    return unseen


def build_seven_topic_pool(conn: sqlite3.Connection) -> list[dict]:
    """Build 7 candidates covering all 5 core instruments + 2 random extra slots."""
    instruments = list(INSTRUMENT_QUERIES.keys())
    seven_slots = instruments + random.sample(instruments, 2)
    random.shuffle(seven_slots)

    # Cache fetched lists per instrument so we don't duplicate API calls
    cache: dict[str, list[dict]] = {}
    pool: list[dict] = []
    used_ids: set[str] = set()

    for instrument in seven_slots:
        if instrument not in cache:
            cache[instrument] = fetch_for_instrument(
                instrument, INSTRUMENT_QUERIES[instrument], conn
            )
        # Pick the next available unseen item for this instrument slot
        for candidate in cache[instrument]:
            if candidate["item_id"] not in used_ids:
                pool.append(candidate)
                used_ids.add(candidate["item_id"])
                break

    return pool


# --- AI Deep-Dive Synthesis ---
def generate_deep_dive_brief(candidates: list[dict], issue_num: int) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    formatted_candidates = "\n\n".join(
        f"CANDIDATE {idx + 1}\n"
        f"Instrument Domain: {item['instrument']}\n"
        f"Title: {item['title']}\n"
        f"Published: {item['published_date']}\n"
        f"URL: {item['url']}\n"
        f"Abstract: {item['summary']}"
        for idx, item in enumerate(candidates)
    )

    prompt = f"""
You are a senior pharmaceutical analytical chemist writing a practical, easy-to-read technical newsletter for QC and R&D bench analysts.

Below are {len(candidates)} candidate topics across Liquid Chromatography (LC), Gas Chromatography (GC), Spectrophotometry, Atomic Absorption Spectroscopy (AAS), and Dissolution Testing.

TASK:
1. Evaluate all candidates and select **1 SINGLE TOPIC** that delivers the highest practical value (instrument troubleshooting, method reliability, or bench optimization).
2. Write a comprehensive, well-paced **6-minute read (~1,000–1,200 words)** deep-dive issue on that 1 selected topic.
3. At the very end, list 3 runner-up topics from the other candidates in a brief "Also Scanned on Our 7-Instrument Radar This Week" section.

OUTPUT STRICTLY IN THIS EXACT MARKDOWN STRUCTURE (no intro or outro commentary):

# The Analytical Bench Deep Dive — Issue #{issue_num:02d}
**[Compelling, Practical Headline Focused on the Single Selected Topic]**  
**Instrument Focus:** [Selected Instrument Domain] • **Reading Time:** ~6 min • **Category:** [Troubleshooting / Method Optimization]

---

### Analyst’s Note
[TODO: Replace with 3 practical sentences connecting this bench issue to real-world lab shift work.]

> **60-Second Bench Summary**
> * **The Problem:** [1 crisp sentence]
> * **Primary Culprits:** [1 crisp sentence]
> * **First-Line Fix:** [1 crisp sentence]
> * **Verification Time:** [Realistic bench estimate, e.g., ~30–45 minutes]

---

### 1. What Recent Research Highlights
[2–3 clear paragraphs explaining the study's core finding and practical significance in plain, accessible English.]

---

### 2. Why This Happens at the Bench (Root Mechanisms)
[Explain the underlying physical/chemical mechanism using 3–4 bolded bullet points so an analyst understands the 'why' behind the symptom.]

---

### 3. Step-by-Step Bench Troubleshooting & Optimization Protocol
[Provide a numbered 4-step practical bench workflow with specific, actionable checks.]

---

### 4. Routine Prevention Checklist for QC & R&D Labs
[3–4 concise bullet points on how to prevent this issue during routine prep or system suitability.]

---

### Primary Source & Weekly Instrument Radar
* **Featured Reference:** [Clean Paper Title & Source](exact_url_of_selected_candidate)

**Also Scanned on Our 7-Instrument Radar This Week:**
* **[Instrument Name]:** *[1-line takeaway of runner-up candidate A]*
* **[Instrument Name]:** *[1-line takeaway of runner-up candidate B]*
* **[Instrument Name]:** *[1-line takeaway of runner-up candidate C]*

RULES:
- Target 1,000–1,200 words so the issue genuinely delivers a thorough, satisfying 6-minute read without fluff.
- Use crisp, scannable formatting, plain-English explanations, and accurate analytical chemistry terminology.
- Base all technical facts and URLs strictly on the provided candidate abstracts; never invent links or data.

CANDIDATE POOL:
{formatted_candidates}
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


# --- Optional Email Delivery ---
def send_email_draft(subject: str, markdown_body: str) -> None:
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    recipient = os.getenv("RECIPIENT_EMAIL", smtp_user)

    if not (smtp_user and smtp_pass and recipient):
        print("SMTP credentials not configured; skipping email delivery.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.set_content(markdown_body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    print(f"Draft emailed to {recipient}.")


# --- Main Orchestrator ---
def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    candidates = build_seven_topic_pool(conn)
    if not candidates:
        print("No new instrument topics found. Exiting without API call.")
        conn.close()
        return

    issue_num = get_and_increment_issue_number(conn, increment=False)
    brief_markdown = generate_deep_dive_brief(candidates, issue_num)

    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_file = OUTPUT_DIR / f"issue_{issue_num:02d}_{date_slug}.md"
    output_file.write_text(brief_markdown + "\n", encoding="utf-8")
    print(f"Saved 6-minute deep-dive draft to {output_file}")

    # Mark only items whose URL appeared in the generated issue so unused candidates remain eligible next week
    referenced_items = [c for c in candidates if c["url"] in brief_markdown] or candidates[:1]
    mark_seen(conn, referenced_items, status="published")

    get_and_increment_issue_number(conn, increment=True)
    conn.close()

    send_email_draft(
        subject=f"[Draft] The Analytical Bench Deep Dive — Issue #{issue_num:02d}",
        markdown_body=brief_markdown,
    )


if __name__ == "__main__":
    main()

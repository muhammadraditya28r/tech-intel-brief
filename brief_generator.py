import os
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
LOOKBACK_DAYS = 7
MAX_CANDIDATES = 20

# Customize your niche keywords and RSS feeds here
OPENALEX_QUERY = (
    '("analytical method validation" OR "nitrosamine" OR "LC-MS/MS" '
    'OR "excipient stability") AND pharmaceutical'
)
RSS_FEEDS = [
    "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs/rss.xml",
]

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "researcher@example.com")


# --- SQLite State & Deduplication ---
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_items (
            item_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            published_date TEXT,
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


def mark_seen(conn: sqlite3.Connection, items: list[dict]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT OR IGNORE INTO seen_items (item_id, source, title, published_date, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (item["item_id"], item["source"], item["title"], item["published_date"], now)
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
    """Convert OpenAlex's word->positions map back into plain text."""
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


def fetch_openalex(days: int = LOOKBACK_DAYS) -> list[dict]:
    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "search": OPENALEX_QUERY,
        "filter": f"from_publication_date:{from_date},has_abstract:true",
        "sort": "publication_date:desc",
        "per-page": 15,
        "mailto": CONTACT_EMAIL,
    }
    resp = requests.get("https://api.openalex.org/works", params=params, timeout=30)
    resp.raise_for_status()

    items = []
    for work in resp.json().get("results", []):
        work_id = work.get("doi") or work.get("id")
        if not work_id:
            continue
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))[:1500]
        items.append(
            {
                "item_id": work_id,
                "source": "OpenAlex Peer-Reviewed Literature",
                "title": strip_html(work.get("title", "Untitled")),
                "url": work.get("doi") or work.get("id"),
                "published_date": work.get("publication_date", ""),
                "summary": abstract,
            }
        )
    return items


def fetch_rss(feed_urls: list[str]) -> list[dict]:
    items = []
    for url in feed_urls:
        parsed = feedparser.parse(url)
        feed_title = parsed.feed.get("title", "Regulatory RSS Feed")
        for entry in parsed.entries[:15]:
            entry_id = entry.get("id") or entry.get("link")
            if not entry_id:
                continue
            items.append(
                {
                    "item_id": entry_id,
                    "source": feed_title,
                    "title": strip_html(entry.get("title", "Untitled")),
                    "url": entry.get("link", ""),
                    "published_date": entry.get("published", ""),
                    "summary": strip_html(entry.get("summary", ""))[:1500],
                }
            )
    return items


# --- AI Synthesis ---
def generate_brief(candidates: list[dict], issue_num: int) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    formatted_candidates = "\n\n".join(
        f"ITEM {idx + 1}\n"
        f"Source: {item['source']}\n"
        f"Title: {item['title']}\n"
        f"Published: {item['published_date']}\n"
        f"URL: {item['url']}\n"
        f"Abstract/Summary: {item['summary']}"
        for idx, item in enumerate(candidates)
    )

    prompt = f"""
You are a senior R&D analytical chemist briefing an R&D director.
Review the candidate updates below from the past 7 days, select the **3 most practical, high-impact items** for analytical R&D and pharmaceutical QC teams, and generate a Markdown newsletter draft.

Output STRICTLY in this exact Markdown structure (no intro or outro commentary):

# The Analytical R&D Brief — Issue #{issue_num:02d}
**[4-8 Word Punchy Subtitle Summarizing the 3 Selected Topics]**

### Analyst’s Note
[TODO: Replace with 3 sentences connecting Item #1 to real-world bench or validation execution.]

---

### 1. [Category Name]: [Concise Headline]
* **The Shift:** [1 sentence: What specifically changed, was discovered, or was updated?]
* **Lab / Compliance Impact:** [1-2 sentences: How does this affect analytical testing, method validation, formulation stability, or regulatory submissions?]
* **Actionable Check:** [1 sentence: What should an R&D or QC team verify in their current workflow?]
* **Primary Source:** [Source Label](exact_url_from_input)

---

### 2. [Category Name]: [Concise Headline]
* **The Shift:** [1 sentence]
* **Lab / Compliance Impact:** [1-2 sentences]
* **Actionable Check:** [1 sentence]
* **Primary Source:** [Source Label](exact_url_from_input)

---

### 3. [Category Name]: [Concise Headline]
* **The Shift:** [1 sentence]
* **Lab / Compliance Impact:** [1-2 sentences]
* **Actionable Check:** [1 sentence]
* **Primary Source:** [Source Label](exact_url_from_input)

Rules:
- Use precise scientific and regulatory terminology in crisp, executive English.
- Rely strictly on the facts and exact URLs provided below; never fabricate data or links.

CANDIDATE UPDATES:
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

    raw_items = fetch_openalex() + fetch_rss(RSS_FEEDS)
    unseen_items = [item for item in raw_items if not is_seen(conn, item["item_id"])]

    if not unseen_items:
        print("No new items found since last run. Exiting without API call.")
        conn.close()
        return

    candidates = unseen_items[:MAX_CANDIDATES]
    issue_num = get_and_increment_issue_number(conn, increment=False)

    brief_markdown = generate_brief(candidates, issue_num)

    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_file = OUTPUT_DIR / f"issue_{issue_num:02d}_{date_slug}.md"
    output_file.write_text(brief_markdown + "\n", encoding="utf-8")
    print(f"Saved draft to {output_file}")

    # Persist all fetched unseen items so they are never re-processed
    mark_seen(conn, unseen_items)
    get_and_increment_issue_number(conn, increment=True)
    conn.close()

    send_email_draft(
        subject=f"[Draft] The Analytical R&D Brief — Issue #{issue_num:02d}",
        markdown_body=brief_markdown,
    )


if __name__ == "__main__":
    main()

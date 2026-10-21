"""Bangla news pipeline: extract -> transform -> load -> enrich.

Run it with:  python pipeline.py
"""
import json
import logging
import os
import sqlite3
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

SOURCE = "https://www.prothomalo.com/"  # any news homepage works; change it to try another site
DB = "data/news.db"
HEADERS = {"User-Agent": "Mozilla/5.0 (student learning project)"}
SKIP = ("/api/", "/collection/", "/topic/", "/author/")  # non-article pages found on Prothom Alo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


def fetch(url, retries=3):
    """GET a page, retrying with backoff (1s, 2s, 4s). Returns HTML text or None."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            log.warning("fetch failed (%s): %s", e, url)
            time.sleep(2**attempt)
    return None


# ---------- E: extract ----------
def extract(limit=30):
    html = fetch(SOURCE)
    if html is None:
        return pd.DataFrame()
    soup = BeautifulSoup(html, "html.parser")
    host = urlparse(SOURCE).netloc

    links = []
    for a in soup.select("a[href]"):
        url = urljoin(SOURCE, a["href"]).split("?")[0]
        # ponytail: "same site + deep path" guesses article links; use a site-specific CSS selector if it picks junk
        if (urlparse(url).netloc == host and url.count("/") >= 4 and url not in links
                and not any(s in url for s in SKIP)):
            links.append(url)

    rows = []
    for url in links[:limit]:
        page = fetch(url)
        if page is None:
            continue
        s = BeautifulSoup(page, "html.parser")
        h1 = s.find("h1")
        rows.append({
            "url": url,
            "title": h1.get_text(strip=True) if h1 else None,
            "body": " ".join(p.get_text(" ", strip=True) for p in s.find_all("p")),
        })
        time.sleep(1)  # be polite to the site
    log.info("extracted %d articles", len(rows))
    return pd.DataFrame(rows)


# ---------- T: transform (clean) ----------
def transform(df):
    if df.empty:
        return df
    df = df.dropna(subset=["title"]).copy()
    for col in ["title", "body"]:
        df[col] = (df[col]
                   .str.normalize("NFC")                    # one Unicode form for Bangla text
                   .str.replace(r"\s+", " ", regex=True)    # collapse spaces/newlines
                   .str.strip())
    df = df[df["body"].str.len() > 200]                     # drop empty / non-article pages
    df = df.drop_duplicates(subset="url")
    df["scraped_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    log.info("%d articles after cleaning", len(df))
    return df


# ---------- L: load ----------
def load(df):
    if df.empty:
        log.info("nothing to load")
        return
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    with sqlite3.connect(DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                url TEXT PRIMARY KEY,
                title TEXT, body TEXT, scraped_at TEXT,
                summary TEXT, category TEXT, embedding TEXT)""")
        before = con.total_changes
        # INSERT OR IGNORE + PRIMARY KEY on url = re-running never creates duplicates
        con.executemany(
            "INSERT OR IGNORE INTO articles (url, title, body, scraped_at) VALUES (?, ?, ?, ?)",
            df[["url", "title", "body", "scraped_at"]].itertuples(index=False))
        log.info("loaded %d new articles", con.total_changes - before)


# ---------- enrich with the LLM ----------
def enrich():
    from llm import embed, summarize

    with sqlite3.connect(DB) as con:
        rows = con.execute("SELECT url, title, body FROM articles WHERE summary IS NULL").fetchall()
        for url, title, body in rows:
            try:
                s = summarize(title, body)
                vec = embed(title + "\n" + s["summary"])
            except Exception as e:  # one bad article must not stop the whole run
                log.warning("enrich failed for %s: %s", url, e)
                continue
            con.execute("UPDATE articles SET summary = ?, category = ?, embedding = ? WHERE url = ?",
                        (s["summary"], s["category"], json.dumps(vec), url))
            con.commit()
            time.sleep(4)  # stay under the free-tier rate limit
    log.info("enriched %d articles", len(rows))


if __name__ == "__main__":
    load(transform(extract()))
    enrich()

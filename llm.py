"""Talks to the Gemini REST API with plain `requests`: summaries and embeddings."""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from the .env file
KEY = os.environ["GEMINI_API_KEY"]
BASE = "https://generativelanguage.googleapis.com/v1beta/models"
CHAT_MODEL = "gemini-2.5-flash"         # model names change: check ai.google.dev/gemini-api/docs/models
EMBED_MODEL = "gemini-embedding-001"
CATEGORIES = ["politics", "economy", "sports", "entertainment", "technology", "international", "other"]


def _post(model, method, payload):
    r = requests.post(f"{BASE}/{model}:{method}", json=payload,
                      headers={"x-goog-api-key": KEY}, timeout=60)
    r.raise_for_status()
    return r.json()


def summarize(title, body):
    prompt = (
        f"Summarize this news article in 2 short sentences in Bangla, "
        f"and pick one category from {CATEGORIES}.\n"
        'Reply only with JSON: {"summary": "...", "category": "..."}\n\n'
        f"Title: {title}\n\n{body[:4000]}"  # cut long articles: fewer tokens = cheaper and faster
    )
    data = _post(CHAT_MODEL, "generateContent", {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    })
    out = json.loads(data["candidates"][0]["content"]["parts"][0]["text"])
    if out.get("category") not in CATEGORIES:  # never trust LLM output blindly
        out["category"] = "other"
    return out


def embed(text):
    data = _post(EMBED_MODEL, "embedContent", {"content": {"parts": [{"text": text}]}})
    return data["embedding"]["values"]

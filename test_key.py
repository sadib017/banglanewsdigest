import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.environ["GEMINI_API_KEY"]
r = requests.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
    headers={"x-goog-api-key": key},
    json={"contents": [{"parts": [{"text": "Say hello in Bangla in one short sentence."}]}]},
    timeout=60,
)
print(r.status_code)
print(r.json()["candidates"][0]["content"]["parts"][0]["text"])
# models/scout_api.py

import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "meta-llama/llama-4-scout:free"
URL = "https://openrouter.ai/api/v1/chat/completions"
API="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"


HEADERS = {
    "Authorization": f"Bearer {API}",
    "Content-Type": "application/json"
}

def call_scout_model(prompt: str):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(URL, headers=HEADERS, json=payload)

    try:
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}).get("total_tokens", 0)
        return {"response": reply, "tokens_used": usage}
    except Exception as e:
        print("\n🔴 SCOUT API ERROR")
        print("Status code:", response.status_code)
        print("Raw response:", response.text)
        raise e

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

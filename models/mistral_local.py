import requests

def call_mistral(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        print(f"[Mistral Error] {response.status_code}: {response.text}")
        return "[Edge Model Error]"
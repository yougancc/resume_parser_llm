import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def run_mistral(prompt: str):
    for attempt in range(2):  # ✅ retry once
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300
            )

            response.raise_for_status()

            output = response.json().get("response", "").strip()

            if output:
                return output

        except Exception as e:
            print(f"Mistral Error (attempt {attempt+1}):", e)
            time.sleep(2)

    return "<json>{\"error\": \"llm_failed\"}</json>"
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"   # ✅ EXACT name from `ollama list`

def run_llm(prompt: str) -> str:
    for attempt in range(2):  # ✅ retry once
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI assistant. Return only valid JSON inside <json> tags."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False
                },
                timeout=300
            )

            response.raise_for_status()

            output = response.json()["message"]["content"]

            start = output.find("<json>")
            end = output.find("</json>")

            if start != -1 and end != -1:
                return output[start:end+7]

            return output.strip()

        except Exception as e:
            print(f"LLM Error (attempt {attempt+1}):", e)

    return "<json>{\"error\":\"llm_failed\"}</json>"
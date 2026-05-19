from llama_cpp import Llama

MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

_llm = None

MAX_CONTEXT = 2048
MAX_OUTPUT_TOKENS = 1100     # JSON needs room
MAX_INPUT_TOKENS = MAX_CONTEXT - MAX_OUTPUT_TOKENS - 200  # safety margin

def get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=MAX_CONTEXT,
            n_threads=4,
            temperature=0.1,
            verbose=False
        )
    return _llm


def run_llm(prompt: str) -> str:
    llm = get_llm()

    # ✅ TOKEN‑BASED INPUT TRUNCATION (CRITICAL)
    tokens = llm.tokenize(prompt.encode("utf-8"))
    tokens = tokens[:MAX_INPUT_TOKENS]
    safe_prompt = llm.detokenize(tokens).decode("utf-8", errors="ignore")

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a STRICT resume extraction engine. "
                    "Return ONLY valid JSON wrapped inside <json> tags. "
                    "Do NOT stop early. Complete the JSON fully."
                )
            },
            {
                "role": "user",
                "content": safe_prompt
            }
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        stop=["</json>"]
    )

    return response["choices"][0]["message"]["content"]
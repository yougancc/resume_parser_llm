from llama_cpp import Llama

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

_mistral = None

def get_mistral():
    global _mistral
    if _mistral is None:
        _mistral = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=6,
            temperature=0.2,
            verbose=False
        )
    return _mistral


def run_mistral(prompt: str):
    model = get_mistral()

    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a hiring assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800
    )

    return response["choices"][0]["message"]["content"]
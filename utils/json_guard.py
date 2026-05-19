import json
import re

def extract_json(text: str):
    """
    Extract valid JSON from LLM output.
    Handles:
    - ```json ... ``` fenced blocks
    - <json> ... </json> blocks
    """

    # 1. Remove markdown code fences if present
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    # 2. Try <json>...</json> first
    match = re.search(r"<json>\s*(\{.*\})\s*</json>", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # 3. Fallback: first balanced JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError("No valid JSON found in LLM output")
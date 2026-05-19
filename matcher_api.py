from fastapi import FastAPI, UploadFile, File
import requests
import json

from llm.llm_engine import run_llm
from utils.json_guard import extract_json
from llm.mistral_engine import run_mistral

app = FastAPI(title="Resume Matcher API")

JD_PROMPT = open("prompts/jd_prompt.txt").read()
ANALYSIS_PROMPT = open("prompts/analysis_prompt.txt").read()


# ✅ Call your existing parser API
def parse_resume(file):
    files = {"file": (file.filename, file.file, "application/pdf")}
    res = requests.post("http://127.0.0.1:8000/resume_parser", files=files)
    return res.json()


# ✅ Extract JD structure using Qwen
def parse_jd(jd_text):
    prompt = JD_PROMPT.replace("{{JD_TEXT}}", jd_text)
    raw = run_llm(prompt)
    return extract_json(raw)


# ✅ Matching logic (deterministic)
def match_skills(resume_skills, jd_skills):

    resume_set = {s.lower() for s in resume_skills}
    jd_set = {s.lower() for s in jd_skills}

    matched = list(resume_set & jd_set)
    missing = list(jd_set - resume_set)

    percentage = 0
    if jd_set:
        percentage = int((len(matched) / len(jd_set)) * 100)

    return percentage, matched, missing


@app.post("/resume_matcher")
async def resume_matcher(file: UploadFile = File(...), jd_text: str = ""):

    # --------------------------------
    # STEP 1: Parse Resume
    # --------------------------------
    resume_json = parse_resume(file)

    # --------------------------------
    # STEP 2: Parse JD
    # --------------------------------
    jd_json = parse_jd(jd_text)

    # --------------------------------
    # STEP 3: Match Skills (Python)
    # --------------------------------
    percentage, matched, missing = match_skills(
        resume_json.get("skills", []),
        jd_json.get("required_skills", [])
    )

    # --------------------------------
    # STEP 4: Gap Analysis (Mistral)
    # --------------------------------
    analysis_input = ANALYSIS_PROMPT \
        .replace("{{RESUME_JSON}}", json.dumps(resume_json)) \
        .replace("{{JD_JSON}}", json.dumps(jd_json)) \
        .replace("{{MATCHED}}", str(matched)) \
        .replace("{{MISSING}}", str(missing))

    raw_analysis = run_mistral(analysis_input)
    analysis_json = extract_json(raw_analysis)

    # --------------------------------
    # FINAL RESPONSE
    # --------------------------------
    response = {
        "match_percentage": percentage,
        "matched_skills": matched,
        "missing_skills": missing,
        **analysis_json
    }

    return response
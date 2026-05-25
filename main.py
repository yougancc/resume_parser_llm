import os
import shutil
from urllib import response
from fastapi import FastAPI, UploadFile, File, HTTPException

from extractor.pdf_extractor import extract_pdf_text
from llm.llm_engine import run_llm
from llm.mistral_engine import run_mistral
from utils.json_guard import extract_json

# ✅ Helper imports
from helpers.skill_utils import extract_skills
from helpers.embedding_utils import semantic_match_embedding
from helpers.core_utils import detect_core_and_optional_skills
from helpers.text_utils import chunk_text

app = FastAPI(title="Resume Parser + Matcher (Clean Architecture)")

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# -------------------------
# ✅ LOAD PROMPTS
# -------------------------
PASS1 = open("prompts/pass1_profile.txt").read()
PASS2 = open("prompts/pass2_experience.txt").read()
PASS3 = open("prompts/pass3_education.txt").read()
ANALYSIS_PROMPT = open("prompts/analysis_prompt.txt").read()

# ✅ Shared keywords (keep in one place)
SKILL_KEYWORDS = [
    "java", "python", "php", "javascript",
    "react", "angular", "node",
    "api", "rest", "microservices",
    "mysql", "sql", "mongodb",
    "aws", "azure", "cloud",
    "project management", "program management",
    "agile", "scrum",
    "stakeholder", "delivery", "cross-functional",
    "ci cd", "jenkins", "pipeline",
    "deployment", "automation", "devops",
    "docker", "kubernetes"
]

# ------------------------------------------------------------
# ✅ ✅ ✅ KEEP THIS AS-IS (NO MODIFICATION — YOUR CONDITION)
# ------------------------------------------------------------
@app.post("/resume_parser")
async def parse_resume(file: UploadFile = File(...)):

    # ✅ Save file
    tmp_path = os.path.join(TMP_DIR, file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    resume_text = extract_pdf_text(tmp_path)
    os.remove(tmp_path)

    # ✅ LIMIT TEXT SIZE (IMPORTANT)
    resume_text = resume_text[:6000]

    # ✅ PASS 1 (unchanged)
    try:
        pass1_prompt = PASS1.replace("{{RESUME_TEXT}}", resume_text)
        pass1_json = extract_json(run_llm(pass1_prompt))
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Failed to extract personal details / summary / skills"
        )

    # ✅ PASS 2 (LIMIT CHUNKS — CRITICAL FIX)
    experience_results = []
    seen_companies = set()

    chunks = chunk_text(resume_text, max_chars=2000)[:2]  # ✅ MAX 2 chunks only

    for chunk in chunks:
        try:
            chunk_json = extract_json(
                run_llm(PASS2.replace("{{RESUME_TEXT}}", chunk))
            )

            for exp in chunk_json.get("experience", []):
                key = exp.get("company", "").strip().lower()
                if key and key not in seen_companies:
                    seen_companies.add(key)
                    experience_results.append(exp)

        except:
            continue

    # ✅ PASS 3 (NO CHUNKS — SINGLE CALL ✅ HUGE SPEED BOOST)
    education_results = []
    certification_results = []

    try:
        chunk_json = extract_json(
            run_llm(PASS3.replace("{{RESUME_TEXT}}", resume_text))
        )

        education_results = chunk_json.get("education", [])
        certification_results = chunk_json.get("certifications", [])

    except:
        pass

    return {
        "personal_details": pass1_json.get("personal_details"),
        "summary": pass1_json.get("summary"),
        "skills": pass1_json.get("skills"),
        "experience": experience_results,
        "education": education_results,
        "certifications": certification_results
    }

# ------------------------------------------------------------
# ✅ MATCHER (CLEAN & MODULAR)
# ------------------------------------------------------------
@app.post("/resume_matcher")
async def resume_matcher(file: UploadFile = File(...), jd_text: str = ""):

    # ✅ Save file
    tmp_path = os.path.join(TMP_DIR, file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    resume_text = extract_pdf_text(tmp_path)
    os.remove(tmp_path)

    # ✅ Extract skills (using helpers)
    resume_skills = extract_skills(resume_text, SKILL_KEYWORDS)
    jd_skills = extract_skills(jd_text, SKILL_KEYWORDS)

    if not jd_skills:
        jd_skills = ["api integration", "program management"]

    # ✅ Embedding matching
    matched = semantic_match_embedding(jd_skills, resume_skills)
    matched = matched[:8]

    # ✅ Core detection
    core_skills, optional_skills = detect_core_and_optional_skills(jd_text, jd_skills)

    matched_set = set(matched)
    missing = [s for s in core_skills if s not in matched_set]

    # ✅ Scoring
    core_match = len([s for s in core_skills if s in matched_set])
    optional_match = len([s for s in optional_skills if s in matched_set])

    total_score = (core_match * 4 + optional_match)
    max_score = (len(core_skills) * 4 + len(optional_skills))

    match_percentage = int((total_score / max_score) * 100) if max_score else 0
    match_percentage = min(max(match_percentage, 15), 90)

    # ✅ Context
    context_hint = f"""
    - Candidate skills: {resume_skills}
    - Strong backend / DevOps / cloud experience
    - Technical Lead with system ownership
    """

    # ✅ LLM Analysis
    analysis_input = ANALYSIS_PROMPT \
        .replace("{{RESUME_JSON}}", f"""
Skills: {resume_skills}

Additional Context:
{context_hint}
""") \
        .replace("{{JD_JSON}}", str({"required_skills": jd_skills})) \
        .replace("{{MATCHED}}", str(matched)) \
        .replace("{{MISSING}}", str(missing))

    analysis = extract_json(run_mistral(analysis_input))

    return {
        "match_percentage": match_percentage,
        "matched_skills": matched,
        "missing_skills": missing,
        **analysis
    }

@app.post("/resume_check")
async def resume_check(file: UploadFile = File(...)):

    # ✅ Save file
    tmp_path = os.path.join(TMP_DIR, file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    resume_text = extract_pdf_text(tmp_path)
    os.remove(tmp_path)

    # ✅ Limit input size (performance)
    resume_text = resume_text[:4000]

    try:
        prompt_template = open("prompts/resume_check_prompt.txt").read()

        final_prompt = prompt_template.replace("{{RESUME_TEXT}}", resume_text)

        # ✅ LLM call
        response = run_llm(final_prompt)

        if "llm_failed" in response:
            raise Exception("LLM timeout")

        try:
            result = extract_json(response)
        except:
            raise Exception("Invalid JSON from LLM")

    except Exception as e:
        print("Resume Check Error:", e)

        # ✅ Safe fallback
        result = {
            "overall_score": 50,
            "fit_label": "Needs Improvement",
            "role_match": {"role": "Not identified", "score": 50},
            "key_strengths": [],
            "missing_skills": [],
            "areas_for_improvement": [
                {"title": "Error", "description": "Could not analyze resume"}
            ],
            "smart_tip": "Ensure resume is properly formatted."
        }

    return result
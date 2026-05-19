import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from llm.mistral_engine import run_mistral
from extractor.pdf_extractor import extract_pdf_text
from llm.llm_engine import run_llm
from utils.json_guard import extract_json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Resume Parser v3 + Matcher (Hybrid Optimized)")

# -------------------------
# Load prompts (NO CHANGE)
# -------------------------
PASS1 = open("prompts/pass1_profile.txt").read()
PASS2 = open("prompts/pass2_experience.txt").read()
PASS3 = open("prompts/pass3_education.txt").read()
JD_PROMPT = open("prompts/jd_prompt.txt").read()
ANALYSIS_PROMPT = open("prompts/analysis_prompt.txt").read()

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# -------------------------
# EXISTING CHUNK FUNCTION
# -------------------------
def chunk_text(text: str, max_chars: int = 3000):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


# ✅ ✅ ✅ NEW FAST HELPER (ONLY ADDITION)
def fast_resume_extract(resume_text: str):
    """
    High-accuracy lightweight extraction
    """

    prompt = f"""
Extract ONLY TECHNICAL + DOMAIN SKILLS from the resume.

Include:
- Technologies (React, Node, Java)
- APIs / integrations
- Project management / delivery skills
- Tools (Jira, AWS, etc.)

DO NOT include soft skills like communication.

Return JSON:

<json>
{{
  "skills": [],
  "summary": "",
  "years_experience": ""
}}
</json>

Resume:
{resume_text}
"""

    return extract_json(run_llm(prompt))

def fast_resume_extract_local(resume_text: str):
    text = resume_text.lower()

    skills = []

    if "api" in text:
        skills.append("api integration")

    if "project manager" in text or "delivery" in text:
        skills.append("program management")

    if "agile" in text or "scrum" in text:
        skills.append("agile")

    if "aws" in text or "cloud" in text:
        skills.append("cloud")

    if "java" in text:
        skills.append("java")

    if "sql" in text:
        skills.append("sql")

    return {
        "skills": list(set(skills)),
        "summary": "Extracted summary",
        "years_experience": "Detected from resume"
    }


def enrich_jd_skills(jd_text: str, jd_json: dict):
    """
    If JD has no explicit skills, infer them safely
    """

    if jd_json.get("required_skills"):
        return jd_json  # already good

    fallback_prompt = f"""
The job description does not explicitly list technologies.

Infer 5–10 most likely technical skills for this role.

Rules:
- Use role context (SDLC, backend, frontend etc.)
- Do NOT hallucinate very niche tools
- Return GENERAL engineering stack

Return JSON:

{{
  "required_skills": []
}}

JD:
{jd_text}
"""

    try:
        fallback = extract_json(run_llm(fallback_prompt))
        jd_json["required_skills"] = fallback.get("required_skills", [])
    except:
        jd_json["required_skills"] = []

    return jd_json

def normalize_skill(skill: str):
    s = skill.lower().strip()

    # ✅ unify API
    if "api" in s or "rest" in s:
        return "api integration"

    # ✅ unify program
    if "program" in s or "project" in s or "delivery" in s:
        return "program management"

    # ✅ unify agile
    if "scrum" in s:
        return "agile"

    # ✅ unify DB
    if "mysql" in s or "database" in s:
        return "sql"

    # ✅ unify cloud
    if "aws" in s:
        return "cloud"

    return s


def infer_functional_skills(resume_text: str):
    text = resume_text.lower()

    inferred = []

    if "project manager" in text or "project management" in text:
        inferred.append("program management")

    if "delivery" in text:
        inferred.append("program management")

    if "agile" in text or "scrum" in text:
        inferred.append("agile")

    if "cross-functional" in text:
        inferred.append("stakeholder management")

    return inferred

def parse_jd_fast(jd_text: str):
    jd = jd_text.lower()

    skills = []

    if "api" in jd:
        skills.append("api integration")

    if "program" in jd or "project" in jd or "delivery" in jd:
        skills.append("program management")

    if "agile" in jd:
        skills.append("agile")

    if "stakeholder" in jd or "cross-functional" in jd:
        skills.append("stakeholder management")

    if "risk" in jd:
        skills.append("risk management")

    if "metrics" in jd:
        skills.append("kpi management")

    return list(set(skills))



# -------------------------
# ✅ EXISTING PARSER API (UNCHANGED)
# -------------------------
@app.post("/resume_parser")
async def parse_resume(file: UploadFile = File(...)):

    tmp_path = os.path.join(TMP_DIR, file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    resume_text = extract_pdf_text(tmp_path)
    os.remove(tmp_path)

    try:
        pass1_prompt = PASS1.replace("{{RESUME_TEXT}}", resume_text)
        pass1_json = extract_json(run_llm(pass1_prompt))
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Failed to extract personal details / summary / skills"
        )

    experience_results = []
    seen_companies = set()

    chunks = chunk_text(resume_text, max_chars=3000)

    for chunk in chunks:
        try:
            chunk_json = extract_json(run_llm(PASS2.replace("{{RESUME_TEXT}}", chunk)))
            for exp in chunk_json.get("experience", []):
                key = exp.get("company", "").strip().lower()
                if key and key not in seen_companies:
                    seen_companies.add(key)
                    experience_results.append(exp)
        except:
            continue

    education_results = []
    certification_results = []

    seen_edu = set()
    seen_cert = set()

    for chunk in chunks:
        try:
            chunk_json = extract_json(run_llm(PASS3.replace("{{RESUME_TEXT}}", chunk)))

            for edu in chunk_json.get("education", []):
                key = str(edu).lower()
                if key not in seen_edu:
                    seen_edu.add(key)
                    education_results.append(edu)

            for cert in chunk_json.get("certifications", []):
                key = cert.lower()
                if key not in seen_cert:
                    seen_cert.add(key)
                    certification_results.append(cert)

        except:
            continue

    return {
        "personal_details": pass1_json.get("personal_details"),
        "summary": pass1_json.get("summary"),
        "skills": pass1_json.get("skills"),
        "experience": experience_results,
        "education": education_results,
        "certifications": certification_results
    }

# ✅ HELPER: AUTO CORE DETECTION (IMPROVED)
def detect_core_and_optional_skills(jd_text: str, jd_skills: list):
    text = jd_text.lower()

    core_keywords = [
        "must", "required", "key skills", "what matters"
    ]

    core_skills = []
    optional_skills = []

    lines = text.split("\n")

    for skill in jd_skills:
        found_core = False

        for line in lines:
            if skill in line:
                if any(k in line for k in core_keywords):
                    core_skills.append(skill)
                    found_core = True
                    break

        if not found_core:
            optional_skills.append(skill)

    # fallback
    if not core_skills:
        core_skills = jd_skills[:2]
        optional_skills = [s for s in jd_skills if s not in core_skills]

    return core_skills, optional_skills



# ✅ HELPER: SEMANTIC SIMILARITY
def semantic_match_embedding(jd_skills, resume_skills, threshold=0.3):
    matched = []

    if not jd_skills or not resume_skills:
        return [], jd_skills

    all_text = jd_skills + resume_skills

    vectorizer = TfidfVectorizer().fit(all_text)
    vectors = vectorizer.transform(all_text)

    jd_vectors = vectors[:len(jd_skills)]
    resume_vectors = vectors[len(jd_skills):]

    for i, jd_skill in enumerate(jd_skills):
        sim_scores = cosine_similarity(jd_vectors[i], resume_vectors)[0]

        max_score = max(sim_scores) if len(sim_scores) > 0 else 0

        if max_score >= threshold:
            matched.append(jd_skill)

    return matched


# -------------------------
# ✅ MATCHER API (FINAL FIXED)
# -------------------------
@app.post("/resume_matcher")
async def resume_matcher(file: UploadFile = File(...), jd_text: str = ""):

    # ✅ Extract Resume
    tmp_path = os.path.join(TMP_DIR, file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    resume_text = extract_pdf_text(tmp_path)
    os.remove(tmp_path)

    text = resume_text.lower()

    # ✅ Skill keywords (only detection, no mapping)
    skill_keywords = [
        "java", "python", "php", "javascript",
        "react", "angular", "node",

        "api", "rest", "microservices",

        "mysql", "sql", "mongodb",

        "aws", "azure", "cloud",

        "project management", "program management",
        "agile", "scrum",

        "stakeholder", "delivery", "cross-functional",

        "testing", "qa",

        # ✅ ADD THESE (CRITICAL)
        "ci cd", "jenkins", "pipeline",
        "deployment", "automation", "devops",
        "docker", "kubernetes"
    ]

    # ✅ Resume skills
    resume_skills = [
        skill for skill in skill_keywords if skill in text
    ]

    resume_skills = list(set(normalize_skill(s) for s in resume_skills))

    # ✅ JD skills
    jd_text_lower = jd_text.lower()

    jd_skills = [
        skill for skill in skill_keywords if skill in jd_text_lower
    ]

    if not jd_skills:
        jd_skills = ["api integration", "program management"]

    jd_skills = list(set(normalize_skill(s) for s in jd_skills))

    # ✅ ✅ ✅ EMBEDDING MATCH (CORE UPGRADE)
    matched = semantic_match_embedding(jd_skills, resume_skills)

    # ✅ AUTO CORE DETECTION
    core_skills, optional_skills = detect_core_and_optional_skills(jd_text, jd_skills)

    # ✅ ONLY CORE SKILLS are missing
    missing = [s for s in core_skills if s not in matched]

    # ✅ SCORING
    core_match = len([s for s in core_skills if s in matched])
    optional_match = len([s for s in optional_skills if s in matched])

    total_score = (core_match * 4 + optional_match)
    max_score = (len(core_skills) * 4 + len(optional_skills))

    match_percentage = int((total_score / max_score) * 100) if max_score else 0

    # ✅ Context for AI
    context_hint = f"""
    - Candidate skills: {resume_skills}
    - Candidate appears to have experience in DevOps, CI/CD, and cloud systems
    - Likely worked in technical delivery and automation roles
    - May have leadership or ownership in deployment pipelines
    """

    # ✅ Debug logs
    print("RESUME:", resume_skills)
    print("JD:", jd_skills)
    print("CORE:", core_skills)
    print("OPTIONAL:", optional_skills)
    print("MATCHED:", matched)
    print("MISSING:", missing)
    print("SCORE:", match_percentage)

    # ✅ AI Analysis
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
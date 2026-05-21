def detect_core_and_optional_skills(jd_text: str, jd_skills: list):
    text = jd_text.lower()

    core_keywords = ["must", "required", "key skills", "what matters"]

    core_skills = []
    optional_skills = []

    for skill in jd_skills:
        if any(skill in line and any(k in line for k in core_keywords)
               for line in text.split("\n")):
            core_skills.append(skill)
        else:
            optional_skills.append(skill)

    if not core_skills:
        core_skills = jd_skills[:2]
        optional_skills = [s for s in jd_skills if s not in core_skills]

    return core_skills, optional_skills
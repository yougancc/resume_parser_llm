def normalize_skill(skill: str):
    s = skill.lower().strip()

    if "api" in s or "rest" in s:
        return "api integration"

    if "project" in s or "delivery" in s or "program" in s:
        return "program management"

    if "devops" in s or "automation" in s:
        return "devops"

    if "kubernetes" in s or "docker" in s:
        return "kubernetes"

    if "scrum" in s:
        return "agile"

    if "mysql" in s or "database" in s:
        return "sql"

    if "aws" in s or "azure" in s:
        return "cloud"

    return s


def extract_skills(text: str, skill_keywords: list):
    return sorted(
        set(normalize_skill(k) for k in skill_keywords if k in text.lower())
    )
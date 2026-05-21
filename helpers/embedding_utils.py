from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_match_embedding(jd_skills, resume_skills, threshold=0.2):
    matched = []

    if not jd_skills or not resume_skills:
        return matched

    all_text = jd_skills + resume_skills

    vectorizer = TfidfVectorizer().fit(all_text)
    vectors = vectorizer.transform(all_text)

    jd_vectors = vectors[:len(jd_skills)]
    resume_vectors = vectors[len(jd_skills):]

    for i, jd_skill in enumerate(jd_skills):
        sim_scores = cosine_similarity(jd_vectors[i], resume_vectors)[0]

        if len(sim_scores) > 0 and round(max(sim_scores), 2) >= threshold:
            matched.append(jd_skill)

    return sorted(set(matched))
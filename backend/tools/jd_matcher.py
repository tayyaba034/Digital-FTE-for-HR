"""
tools/jd_matcher.py
Job Description ↔ CV matching engine.
Extracts keywords from JD and scores CV alignment.
Inspired by BowJob (https://github.com/rurahim/BowJob)
"""
import re
from collections import Counter
from typing import Any

# ── Common tech/skill keywords to look for ──────────────────────────────────
TECH_KEYWORDS = {
    # Languages
    "python", "javascript", "typescript", "java", "go", "golang", "rust", "c++",
    "c#", "ruby", "scala", "kotlin", "swift", "r", "matlab", "php",
    # Frameworks / Libraries
    "react", "next.js", "nextjs", "vue", "angular", "fastapi", "django", "flask",
    "spring", "express", "node.js", "nodejs", "langchain", "langgraph", "pytorch",
    "tensorflow", "scikit-learn", "pandas", "numpy",
    # Infrastructure
    "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "terraform", "ansible",
    "ci/cd", "github actions", "jenkins", "linux", "nginx",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "dynamodb", "snowflake", "bigquery",
    # AI/ML
    "llm", "rag", "machine learning", "deep learning", "nlp", "computer vision",
    "fine-tuning", "embeddings", "vector database", "pinecone", "weaviate",
    # Concepts
    "rest", "graphql", "grpc", "microservices", "event-driven", "agile", "scrum",
    "tdd", "ci/cd", "devops", "mlops",
}

SOFT_KEYWORDS = {
    "leadership", "communication", "collaboration", "problem-solving", "mentoring",
    "strategic", "stakeholder", "cross-functional", "ownership", "initiative",
}

LEVEL_KEYWORDS = {
    "junior": 1, "mid": 2, "senior": 3, "staff": 4, "principal": 5,
    "lead": 3, "manager": 4, "director": 5, "head of": 5, "vp": 6,
}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\s\.\+\#\/]", " ", text.lower())


def extract_keywords(text: str) -> dict[str, Any]:
    """
    Extract relevant keywords from a JD or CV.
    Returns: {tech_skills, soft_skills, seniority_level, raw_frequency}
    """
    normalised = _normalise(text)
    words = set(normalised.split())
    bigrams = {f"{a} {b}" for a, b in zip(normalised.split(), normalised.split()[1:])}
    tokens = words | bigrams

    tech_found = {kw for kw in TECH_KEYWORDS if kw in tokens}
    soft_found = {kw for kw in SOFT_KEYWORDS if kw in tokens}

    # Detect seniority
    level = 2  # default mid
    for kw, lvl in LEVEL_KEYWORDS.items():
        if kw in normalised:
            level = max(level, lvl)

    # Raw word frequency for less-structured keyword extraction
    word_freq = Counter(normalised.split())
    # Remove stop words
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "with",
            "is", "are", "be", "as", "on", "at", "by", "we", "you", "our",
            "will", "have", "has", "that", "this", "your", "their", "they"}
    word_freq = {w: c for w, c in word_freq.items() if w not in stop and len(w) > 2}

    return {
        "tech_skills": sorted(tech_found),
        "soft_skills": sorted(soft_found),
        "seniority_level": level,
        "top_keywords": sorted(word_freq, key=word_freq.get, reverse=True)[:30],  # type: ignore
    }


def score_cv_against_jd(cv_text: str, jd_text: str) -> dict[str, Any]:
    """
    Score how well a CV matches a Job Description.

    Returns:
        match_score: 0.0 - 1.0
        keywords_matched: list of matched tech skills
        keywords_missing: list of JD skills not in CV
        seniority_match: bool
        improvement_suggestions: list of strings
    """
    jd_data = extract_keywords(jd_text)
    cv_data = extract_keywords(cv_text)

    jd_tech = set(jd_data["tech_skills"])
    cv_tech = set(cv_data["tech_skills"])

    matched = jd_tech & cv_tech
    missing = jd_tech - cv_tech

    # Keyword match score (weighted 70%)
    keyword_score = len(matched) / max(len(jd_tech), 1)

    # Seniority match score (weighted 20%)
    jd_level = jd_data["seniority_level"]
    cv_level = cv_data["seniority_level"]
    level_diff = abs(jd_level - cv_level)
    seniority_score = max(0, 1 - (level_diff * 0.25))

    # Soft skills match (weighted 10%)
    jd_soft = set(jd_data["soft_skills"])
    cv_soft = set(cv_data["soft_skills"])
    soft_score = len(jd_soft & cv_soft) / max(len(jd_soft), 1) if jd_soft else 1.0

    # Composite score
    match_score = round(
        (keyword_score * 0.70) + (seniority_score * 0.20) + (soft_score * 0.10), 3
    )

    suggestions = []
    if missing:
        suggestions.append(f"Add these missing skills if you have them: {', '.join(sorted(missing)[:5])}")
    if level_diff > 1:
        suggestions.append(f"JD targets level {jd_level}, your CV signals level {cv_level} — adjust language accordingly")
    if keyword_score < 0.5:
        suggestions.append("Low keyword overlap — rewrite skills section to mirror JD terminology")

    return {
        "match_score": match_score,
        "keywords_matched": sorted(matched),
        "keywords_missing": sorted(missing),
        "jd_seniority": jd_level,
        "cv_seniority": cv_level,
        "seniority_match": level_diff <= 1,
        "improvement_suggestions": suggestions,
    }


def extract_requirements_from_jd(jd_text: str) -> list[str]:
    """
    Extract bullet-point style requirements from a JD.
    Looks for numbered/bulleted lists and 'Requirements' section.
    """
    lines = jd_text.split("\n")
    requirements = []
    in_requirements = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect section header
        if re.match(r"(requirements?|qualifications?|must.have|skills?)", stripped, re.IGNORECASE):
            in_requirements = True
            continue

        # Detect end of requirements section
        if in_requirements and re.match(r"(responsibilities|benefits|about us|nice to have)", stripped, re.IGNORECASE):
            in_requirements = False
            continue

        # Collect bulleted lines
        if in_requirements and re.match(r"^[-•*\d]+[\.\)]?\s+", stripped):
            req = re.sub(r"^[-•*\d]+[\.\)]?\s+", "", stripped)
            if len(req) > 10:
                requirements.append(req)

    return requirements[:20]  # Cap at 20

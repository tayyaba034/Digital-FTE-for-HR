"""
tools/cv_parser.py
Parse uploaded CVs (PDF/DOCX) and extract structured information.
"""
import io
import re
from typing import Any
import pdfplumber
import structlog
from docx import Document as DocxDocument

log = structlog.get_logger()

COMMON_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "react", "node", "node.js", "fastapi", "django", "flask", "spring",
    "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes", "aws",
    "gcp", "azure", "tensorflow", "pytorch", "scikit-learn", "machine learning",
    "deep learning", "nlp", "computer vision", "figma", "photoshop", "illustrator",
    "ui/ux", "graphic design", "adobe xd"
]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Auto-detect file type and extract text."""
    filename_lower = filename.lower()
    if filename_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename_lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif filename_lower.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def parse_cv_sections(raw_text: str) -> dict[str, str]:
    """
    Attempt to split a CV into logical sections.
    Returns dict like: {"summary": ..., "experience": ..., "education": ..., "skills": ...}
    """
    section_headers = {
        "summary": r"(summary|profile|objective|about me)",
        "experience": r"(experience|work history|employment|career history)",
        "education": r"(education|academic|qualifications|degrees)",
        "skills": r"(skills|technical skills|competencies|technologies)",
        "certifications": r"(certifications|certificates|licenses)",
        "projects": r"(projects|portfolio|open source)",
        "languages": r"(languages|spoken languages)",
    }

    lines = raw_text.split("\n")
    sections: dict[str, list[str]] = {"_header": []}
    current_section = "_header"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            sections.setdefault(current_section, []).append("")
            continue

        matched = False
        for section_key, pattern in section_headers.items():
            if re.match(pattern, stripped, re.IGNORECASE) and len(stripped) < 50:
                current_section = section_key
                sections.setdefault(current_section, [])
                matched = True
                break

        if not matched:
            sections.setdefault(current_section, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def extract_contact_info(raw_text: str) -> dict[str, str]:
    """Extract basic contact info via regex."""
    info = {}

    # Email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', raw_text)
    if email_match:
        info["email"] = email_match.group()

    # Phone
    phone_match = re.search(r'(\+?[\d\s\-\(\)]{10,})', raw_text)
    if phone_match:
        info["phone"] = phone_match.group().strip()

    # LinkedIn
    linkedin_match = re.search(r'linkedin\.com/in/[\w\-]+', raw_text, re.IGNORECASE)
    if linkedin_match:
        info["linkedin"] = "https://" + linkedin_match.group()

    # GitHub
    github_match = re.search(r'github\.com/[\w\-]+', raw_text, re.IGNORECASE)
    if github_match:
        info["github"] = "https://" + github_match.group()

    return info


def extract_skills(raw_text: str, sections: dict[str, str]) -> list[str]:
    """Extract skills from skills section + known skill patterns in full CV text."""
    skill_set: set[str] = set()
    skills_text = sections.get("skills", "")
    if skills_text:
        tokens = re.split(r"[,|\n;/•\-]+", skills_text)
        for token in tokens:
            t = token.strip()
            if 1 < len(t) <= 40:
                skill_set.add(t)

    text_l = raw_text.lower()
    for skill in COMMON_SKILLS:
        if skill in text_l:
            skill_set.add(skill)

    cleaned = []
    seen_lower: set[str] = set()
    for s in skill_set:
        normalized = re.sub(r"\s+", " ", s).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        cleaned.append(normalized)
    return sorted(cleaned)[:60]


def extract_projects(sections: dict[str, str], raw_text: str) -> list[dict[str, Any]]:
    """Extract project blocks with light keyword tagging."""
    projects_text = sections.get("projects", "")
    if not projects_text:
        # Fallback: try from raw text if a "projects" heading exists.
        m = re.search(r"(?is)\bprojects?\b(.*?)(\n\s*\n[A-Z][A-Za-z ]{2,30}\s*$|$)", raw_text)
        projects_text = (m.group(1) or "").strip() if m else ""
    if not projects_text:
        return []

    blocks = [b.strip() for b in re.split(r"\n\s*\n+", projects_text) if b.strip()]
    projects: list[dict[str, Any]] = []
    for block in blocks[:20]:
        lines = [ln.strip("•- \t") for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        title = lines[0][:120]
        summary = " ".join(lines[1:])[:700] if len(lines) > 1 else lines[0][:700]
        body_l = f"{title}\n{summary}".lower()
        matched = [skill for skill in COMMON_SKILLS if skill in body_l]
        projects.append({
            "title": title,
            "summary": summary,
            "keywords": matched[:12],
        })
    return projects


def parse_cv(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """
    Full CV parsing pipeline.
    Returns structured CV data ready for the Resume Agent.
    """
    raw_text = extract_text_from_file(file_bytes, filename)
    sections = parse_cv_sections(raw_text)
    contact_info = extract_contact_info(raw_text)
    skills = extract_skills(raw_text, sections)
    projects = extract_projects(sections, raw_text)

    return {
        "raw_text": raw_text,
        "sections": sections,
        "contact_info": contact_info,
        "skills": skills,
        "projects": projects,
        "word_count": len(raw_text.split()),
        "filename": filename,
    }

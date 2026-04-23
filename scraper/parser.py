SKILL_KEYWORDS = [
    "python", "pytorch", "tensorflow", "scikit-learn", "keras", "langchain",
    "openai", "llm", "gpt", "bert", "transformers", "huggingface",
    "docker", "kubernetes", "aws", "gcp", "azure", "sql", "postgresql",
    "mongodb", "redis", "kafka", "spark", "hadoop", "mlflow", "airflow",
    "fastapi", "flask", "django", "pandas", "numpy", "matplotlib",
    "nlp", "computer vision", "rag", "fine-tuning", "reinforcement learning",
    "langraph", "llamaindex", "pinecone", "chromadb", "weaviate",
    "playwright", "selenium", "scrapy",
]


def normalize_job(raw: dict) -> dict | None:
    """
    Normalize a raw job dict (from any source) into a standard schema.
    Accepts 'title' or 'position' as the job title key (RemoteOK uses 'position').
    Returns None if required fields are missing.
    """
    title = (raw.get("title") or raw.get("position") or "").strip()
    source_url = (raw.get("source_url") or raw.get("url") or "").strip()

    if not title or not source_url:
        return None

    return {
        "title": title[:500],
        "company": (raw.get("company") or "").strip()[:255],
        "description": (raw.get("description") or "").strip(),
        "location": (raw.get("location") or "").strip()[:255],
        "salary": (raw.get("salary") or None),
        "source_url": source_url[:1000],
        "posted_at": raw.get("posted_at"),
    }


def extract_skills(description: str) -> list[str]:
    """Return list of matched skill keywords found in description (lowercased match)."""
    desc_lower = description.lower()
    return [skill for skill in SKILL_KEYWORDS if skill in desc_lower]

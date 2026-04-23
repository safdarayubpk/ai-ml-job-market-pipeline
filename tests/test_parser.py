from scraper.parser import normalize_job, extract_skills, SKILL_KEYWORDS


def test_normalize_job_valid():
    raw = {
        "title": "ML Engineer",
        "company": "Acme",
        "description": "Python and PyTorch required",
        "location": "Remote",
        "salary": "$100k",
        "source_url": "https://example.com/job/1",
    }
    result = normalize_job(raw)
    assert result is not None
    assert result["title"] == "ML Engineer"
    assert result["source_url"] == "https://example.com/job/1"
    assert result["salary"] == "$100k"


def test_normalize_job_missing_url_returns_none():
    assert normalize_job({"title": "ML Engineer"}) is None


def test_normalize_job_missing_title_returns_none():
    assert normalize_job({"source_url": "https://example.com/job/1"}) is None


def test_normalize_job_truncates_long_title():
    raw = {"title": "A" * 600, "source_url": "https://example.com/job/1"}
    result = normalize_job(raw)
    assert result is not None
    assert len(result["title"]) == 500


def test_normalize_job_accepts_position_as_title():
    """RemoteOK JSON API uses 'position' not 'title'."""
    raw = {"position": "Senior AI Engineer", "source_url": "https://remoteok.com/job/1"}
    result = normalize_job(raw)
    assert result is not None
    assert result["title"] == "Senior AI Engineer"


def test_extract_skills_finds_known_keywords():
    desc = "We need expertise in Python, PyTorch, and LangChain for this LLM role."
    skills = extract_skills(desc)
    assert "python" in skills
    assert "pytorch" in skills
    assert "langchain" in skills


def test_extract_skills_case_insensitive():
    skills = extract_skills("Strong PYTHON and TENSORFLOW background required.")
    assert "python" in skills
    assert "tensorflow" in skills


def test_extract_skills_no_match():
    assert extract_skills("Experience with COBOL and mainframes.") == []


def test_skill_keywords_is_non_empty_list():
    assert isinstance(SKILL_KEYWORDS, list)
    assert len(SKILL_KEYWORDS) >= 20

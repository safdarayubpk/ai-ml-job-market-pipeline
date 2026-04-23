# AI/ML Job Market Intelligence Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end pipeline that scrapes public AI/ML job listings, applies clustering and classification, generates LLM insights via Claude API, and pushes automated reports to Google Sheets.

**Architecture:** Playwright scraper → PostgreSQL (SQLAlchemy) → scikit-learn ML pipeline → Claude API insight generator → gspread Google Sheets reporter. Everything triggered via a single `POST /run-pipeline` FastAPI endpoint using asyncio throughout.

**Tech Stack:** Python 3.11, Playwright, PostgreSQL, SQLAlchemy, Alembic, scikit-learn, Anthropic SDK, gspread, FastAPI, uvicorn, Docker Compose

---

## File Map

```
ai-ml-pipeline/
├── scraper/
│   ├── __init__.py
│   ├── playwright_scraper.py   # async Playwright scraping + asyncio.gather
│   └── parser.py               # normalize raw job dicts + extract_skills
├── db/
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy ORM: Job, JobFeatures, PipelineRun
│   └── session.py              # engine factory, get_session(), create_tables()
├── ml/
│   ├── __init__.py
│   ├── features.py             # TF-IDF vectorization, compute_skill_frequencies
│   ├── clustering.py           # KMeans, silhouette k-selection, PCA plot
│   └── classifier.py           # rule-based labeling, LR/SVM/RF comparison
├── llm/
│   ├── __init__.py
│   └── insight_generator.py   # prompt template + Claude API call
├── reporter/
│   ├── __init__.py
│   └── sheets.py              # gspread: authenticate, create tab, write tables
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI app, POST /run-pipeline, GET /health
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── sample_jobs.json   # 20 pre-defined jobs for integration tests
│   ├── test_parser.py
│   ├── test_features.py
│   ├── test_clustering.py
│   ├── test_classifier.py
│   └── test_pipeline.py       # integration test using fixtures + mocks
├── pipeline.py                # orchestrates all steps end-to-end
├── Dockerfile
├── docker-compose.yml
├── alembic.ini
├── .env.example
├── pytest.ini
└── requirements.txt
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `pytest.ini`
- Create: `scraper/__init__.py`, `db/__init__.py`, `ml/__init__.py`, `llm/__init__.py`, `reporter/__init__.py`, `api/__init__.py`, `tests/__init__.py`
- Create: `tests/fixtures/` directory

- [ ] **Step 1: Create `requirements.txt`**

```
playwright==1.44.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.30
alembic==1.13.1
scikit-learn==1.4.2
matplotlib==3.9.0
anthropic==0.28.0
gspread==6.1.2
google-auth==2.29.0
fastapi==0.111.0
uvicorn==0.30.1
python-dotenv==1.0.1
pytest==8.2.2
pytest-asyncio==0.23.7
beautifulsoup4==4.12.3
httpx==0.27.0
numpy==1.26.4
pandas==2.2.2
scipy==1.13.0
```

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install chromium
RUN playwright install-deps chromium
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create `docker-compose.yml`**

```yaml
version: "3.9"
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: pipeline
      POSTGRES_PASSWORD: pipeline
      POSTGRES_DB: jobmarket
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - db

volumes:
  pgdata:
```

- [ ] **Step 4: Create `.env.example`**

```
DATABASE_URL=postgresql://pipeline:pipeline@localhost:5432/jobmarket
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_SHEETS_ID=your_google_sheet_id_here
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
```

- [ ] **Step 5: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 6: Create empty `__init__.py` files**

```bash
touch scraper/__init__.py db/__init__.py ml/__init__.py llm/__init__.py reporter/__init__.py api/__init__.py tests/__init__.py
mkdir -p tests/fixtures
```

- [ ] **Step 7: Install dependencies**

```bash
pip install -r requirements.txt
playwright install chromium
```

Expected: all packages install without error.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt Dockerfile docker-compose.yml .env.example pytest.ini scraper/ db/ ml/ llm/ reporter/ api/ tests/
git commit -m "feat: project scaffold and dependencies"
```

---

## Task 2: Database Models + Session

**Files:**
- Create: `db/models.py`
- Create: `db/session.py`

- [ ] **Step 1: Write failing test for session (verifies models create tables correctly)**

Create `tests/test_db.py`:

```python
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from db.models import Base, Job, JobFeatures, PipelineRun

def test_tables_are_created():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "jobs" in tables
    assert "job_features" in tables
    assert "pipeline_runs" in tables

def test_job_insert_and_query():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    job = Job(title="ML Engineer", company="Acme", source_url="https://example.com/1")
    session.add(job)
    session.commit()
    result = session.query(Job).filter_by(source_url="https://example.com/1").first()
    assert result is not None
    assert result.title == "ML Engineer"
    session.close()

def test_job_deduplication_on_source_url():
    from sqlalchemy.exc import IntegrityError
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(Job(title="ML Engineer", source_url="https://example.com/1"))
    session.commit()
    session.add(Job(title="ML Engineer Duplicate", source_url="https://example.com/1"))
    try:
        session.commit()
        assert False, "Should have raised IntegrityError"
    except IntegrityError:
        session.rollback()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'db.models'`

- [ ] **Step 3: Create `db/models.py`**

```python
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    company = Column(String(255))
    description = Column(Text)
    location = Column(String(255))
    salary = Column(String(255))
    source_url = Column(String(1000), unique=True, nullable=False)
    posted_at = Column(DateTime, nullable=True)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    features = relationship(
        "JobFeatures", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )


class JobFeatures(Base):
    __tablename__ = "job_features"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), unique=True)
    cluster_id = Column(Integer)
    seniority_label = Column(String(50))
    skill_vector = Column(JSON)
    job = relationship("Job", back_populates="features")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_at = Column(DateTime, default=datetime.utcnow)
    jobs_scraped = Column(Integer)
    llm_report = Column(Text)
```

- [ ] **Step 4: Create `db/session.py`**

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from db.models import Base

load_dotenv()


def _make_engine(url: str | None = None):
    database_url = url or os.environ["DATABASE_URL"]
    return create_engine(database_url)


def create_tables(url: str | None = None):
    """Create all tables. Call once at startup."""
    engine = _make_engine(url)
    Base.metadata.create_all(engine)
    return engine


def get_session(url: str | None = None) -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    engine = _make_engine(url)
    return sessionmaker(bind=engine)()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Set up Alembic for PostgreSQL migrations**

```bash
alembic init db/migrations
```

Edit `alembic.ini` — find `sqlalchemy.url` line and replace:
```ini
sqlalchemy.url = postgresql://pipeline:pipeline@localhost:5432/jobmarket
```

Edit `db/migrations/env.py` — find the `target_metadata = None` line and replace the top section:
```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from db.models import Base
target_metadata = Base.metadata
```

Generate first migration:
```bash
docker-compose up -d db
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

Expected: migration file created in `db/migrations/versions/`, tables created in PostgreSQL.

- [ ] **Step 7: Commit**

```bash
git add db/ tests/test_db.py alembic.ini
git commit -m "feat: database models, session factory, alembic migration"
```

---

## Task 3: Job Parser + Tests

**Files:**
- Create: `scraper/parser.py`
- Create: `tests/fixtures/sample_jobs.json`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_parser.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_parser.py -v
```

Expected: `ModuleNotFoundError: No module named 'scraper.parser'`

- [ ] **Step 3: Create `scraper/parser.py`**

```python
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
```

- [ ] **Step 4: Create `tests/fixtures/sample_jobs.json`**

```json
[
  {"title": "Senior ML Engineer", "company": "AlphaAI", "description": "Looking for a Senior ML Engineer with strong Python, PyTorch, and LangChain skills. You will build LLM pipelines and RAG systems using PostgreSQL and Docker.", "location": "Remote", "salary": "$140k", "source_url": "https://remoteok.com/jobs/1"},
  {"title": "Junior Data Scientist", "company": "DataCo", "description": "Entry level position for a Data Scientist. Python and scikit-learn required. TensorFlow a plus. Work on classification and clustering problems.", "location": "Remote", "salary": "$70k", "source_url": "https://remoteok.com/jobs/2"},
  {"title": "ML Engineer", "company": "NeuralNet Inc", "description": "ML Engineer needed to fine-tune transformer models using PyTorch and Huggingface. LLM experience required. FastAPI for serving models.", "location": "Remote", "salary": "$120k", "source_url": "https://remoteok.com/jobs/3"},
  {"title": "Lead AI Engineer", "company": "GenAI Corp", "description": "Lead AI Engineer to head our LLM team. LangChain, OpenAI, RAG, and vector databases like Pinecone and ChromaDB. Strong Python and AWS required.", "location": "Remote", "salary": "$160k", "source_url": "https://remoteok.com/jobs/4"},
  {"title": "Senior Data Scientist", "company": "InsightLab", "description": "Senior Data Scientist for time-series forecasting and NLP projects. Python, scikit-learn, PyTorch, and MLflow. Experience with Airflow pipelines.", "location": "Remote", "salary": "$130k", "source_url": "https://remoteok.com/jobs/5"},
  {"title": "MLOps Engineer", "company": "CloudML", "description": "MLOps Engineer to build and maintain ML infrastructure. Docker, Kubernetes, AWS, MLflow, Airflow. Python scripting for automation.", "location": "Remote", "salary": "$125k", "source_url": "https://remoteok.com/jobs/6"},
  {"title": "Junior ML Engineer", "company": "StartupAI", "description": "Junior ML Engineer for our growing team. Familiarity with Python, scikit-learn, and basic neural networks with PyTorch. Entry level, mentorship provided.", "location": "Remote", "salary": "$65k", "source_url": "https://remoteok.com/jobs/7"},
  {"title": "NLP Engineer", "company": "TextAI", "description": "NLP Engineer with deep knowledge of transformers, BERT, GPT models. Python, Huggingface, LangChain, and RAG pipeline experience required. PostgreSQL for data storage.", "location": "Remote", "salary": "$135k", "source_url": "https://remoteok.com/jobs/8"},
  {"title": "Sr. AI Researcher", "company": "DeepLabs", "description": "Sr. AI Researcher for reinforcement learning and LLM alignment work. PyTorch, Python, and published research preferred. Fine-tuning experience essential.", "location": "Remote", "salary": "$150k", "source_url": "https://remoteok.com/jobs/9"},
  {"title": "Computer Vision Engineer", "company": "VisionCo", "description": "Computer vision engineer using PyTorch, OpenCV, and TensorFlow. Build image classification and object detection models. Docker and Kubernetes for deployment.", "location": "Remote", "salary": "$115k", "source_url": "https://remoteok.com/jobs/10"},
  {"title": "Data Engineer", "company": "PipelineOps", "description": "Data Engineer to build ETL pipelines with Python, Spark, Kafka, and Airflow. PostgreSQL, MongoDB, Redis. AWS infrastructure.", "location": "Remote", "salary": "$110k", "source_url": "https://remoteok.com/jobs/11"},
  {"title": "AI Product Engineer", "company": "ProductAI", "description": "Build AI-powered product features using LangChain, OpenAI, and RAG. Python, FastAPI, PostgreSQL. Experience with LLM APIs required.", "location": "Remote", "salary": "$120k", "source_url": "https://remoteok.com/jobs/12"},
  {"title": "Principal ML Engineer", "company": "ScaleAI", "description": "Principal ML Engineer to lead distributed training infrastructure. PyTorch, Kubernetes, AWS, MLflow. Strong Python and research background.", "location": "Remote", "salary": "$175k", "source_url": "https://remoteok.com/jobs/13"},
  {"title": "ML Engineer - LLMs", "company": "LLMStartup", "description": "ML Engineer focused on LLM fine-tuning using LoRA and QLoRA techniques. Python, PyTorch, Huggingface transformers. LangChain for orchestration.", "location": "Remote", "salary": "$130k", "source_url": "https://remoteok.com/jobs/14"},
  {"title": "Graduate AI Engineer", "company": "NewTech", "description": "Entry level position for recent graduates. Python basics, some scikit-learn and TensorFlow knowledge. We will train you on LLMs and NLP.", "location": "Remote", "salary": "$60k", "source_url": "https://remoteok.com/jobs/15"},
  {"title": "Senior MLOps Engineer", "company": "InfraML", "description": "Senior MLOps engineer to design ML platform. Docker, Kubernetes, GCP, MLflow. Python automation scripts. CI/CD for ML pipelines.", "location": "Remote", "salary": "$140k", "source_url": "https://remoteok.com/jobs/16"},
  {"title": "AI Automation Engineer", "company": "AutoBot", "description": "Build AI automation workflows with LangChain, Python, and FastAPI. RAG pipelines, vector databases including ChromaDB and Pinecone. Playwright for web automation.", "location": "Remote", "salary": "$115k", "source_url": "https://remoteok.com/jobs/17"},
  {"title": "Staff ML Engineer", "company": "BigTech", "description": "Staff ML engineer for recommendation systems. Python, PyTorch, TensorFlow, Spark, Kafka. Distributed training on GCP and AWS. 7+ years experience.", "location": "Remote", "salary": "$200k", "source_url": "https://remoteok.com/jobs/18"},
  {"title": "ML Engineer - Computer Vision", "company": "RoboVision", "description": "ML engineer for computer vision in robotics. Python, PyTorch, OpenCV. Docker, Kubernetes for deployment. Real-time inference optimization.", "location": "Remote", "salary": "$125k", "source_url": "https://remoteok.com/jobs/19"},
  {"title": "Junior NLP Engineer", "company": "LangCo", "description": "Junior NLP engineer for text classification and sentiment analysis. Python, scikit-learn, and BERT basics. Entry level, 0-1 years experience acceptable.", "location": "Remote", "salary": "$68k", "source_url": "https://remoteok.com/jobs/20"}
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_parser.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scraper/parser.py tests/test_parser.py tests/fixtures/sample_jobs.json
git commit -m "feat: job parser with normalize_job, extract_skills, and test fixtures"
```

---

## Task 4: Playwright Scraper

**Files:**
- Create: `scraper/playwright_scraper.py`

No automated tests (requires live browser). Verified by running manually.

- [ ] **Step 1: Create `scraper/playwright_scraper.py`**

```python
import asyncio
import logging
from playwright.async_api import async_playwright, Page

logger = logging.getLogger(__name__)

SOURCES = [
    "https://remoteok.com/remote-ai-jobs",
    "https://remoteok.com/remote-machine-learning-jobs",
]


async def _scrape_page_with_retry(page: Page, url: str, max_retries: int = 3) -> list[dict]:
    """Load a RemoteOK listing page and extract job rows. Retries on failure."""
    for attempt in range(max_retries):
        try:
            await page.goto(url, timeout=30000, wait_until="networkidle")
            await page.wait_for_timeout(2000)  # allow JS to render
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to load {url} after {max_retries} attempts: {e}")
                return []
            wait = 2 ** attempt
            logger.warning(f"Retrying {url} in {wait}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)

    results = []
    # RemoteOK renders jobs as <tr> elements with class "job"
    # Each row has data attributes: data-url, and child elements for title/company
    job_rows = await page.query_selector_all("tr.job")
    for row in job_rows:
        try:
            title_el = await row.query_selector("h2.title")
            company_el = await row.query_selector("h3.company")
            desc_el = await row.query_selector("div.description")
            location_el = await row.query_selector("div.location")
            salary_el = await row.query_selector("div.salary")
            url_attr = await row.get_attribute("data-url")

            title = await title_el.inner_text() if title_el else ""
            if not title.strip():
                continue  # skip header rows or ads

            results.append({
                "title": title.strip(),
                "company": (await company_el.inner_text()).strip() if company_el else "",
                "description": (await desc_el.inner_text()).strip() if desc_el else "",
                "location": (await location_el.inner_text()).strip() if location_el else "",
                "salary": (await salary_el.inner_text()).strip() if salary_el else None,
                "source_url": f"https://remoteok.com{url_attr}" if url_attr else url,
            })
        except Exception as e:
            logger.debug(f"Skipping row due to parse error: {e}")
            continue

    logger.info(f"Scraped {len(results)} jobs from {url}")
    return results


async def scrape_all_sources(max_per_source: int = 100) -> list[dict]:
    """
    Scrape all configured sources concurrently using asyncio.gather.
    Returns deduplicated list of raw job dicts.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        )

        pages = [await context.new_page() for _ in SOURCES]
        results = await asyncio.gather(
            *[_scrape_page_with_retry(page, url) for page, url in zip(pages, SOURCES)],
            return_exceptions=True,
        )
        await browser.close()

    all_jobs: list[dict] = []
    seen_urls: set[str] = set()
    for result in results:
        if isinstance(result, list):
            for job in result[:max_per_source]:
                url = job.get("source_url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_jobs.append(job)

    logger.info(f"Total unique jobs scraped: {len(all_jobs)}")
    return all_jobs
```

- [ ] **Step 2: Test the scraper manually**

```bash
python -c "
import asyncio
from scraper.playwright_scraper import scrape_all_sources
jobs = asyncio.run(scrape_all_sources())
print(f'Scraped {len(jobs)} jobs')
if jobs:
    print('First job:', jobs[0])
"
```

Expected: prints number of jobs scraped and a sample job dict.

> **Note:** If RemoteOK changes its HTML structure, update the selectors in `_scrape_page_with_retry`. Inspect the live page with `await page.content()` to find current selector names.

- [ ] **Step 3: Commit**

```bash
git add scraper/playwright_scraper.py
git commit -m "feat: async Playwright scraper for RemoteOK with retry and deduplication"
```

---

## Task 5: Feature Extraction + Tests

**Files:**
- Create: `ml/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_features.py`:

```python
from ml.features import build_tfidf_matrix, compute_skill_frequencies

SAMPLE_DESCRIPTIONS = [
    "We need a Python developer with strong PyTorch skills for LLM work.",
    "Looking for a machine learning engineer with scikit-learn and TensorFlow experience.",
    "Seeking a data scientist who knows Python and SQL for our analytics team.",
    "ML Engineer with LangChain, OpenAI, and RAG pipeline experience required.",
    "Backend developer with Python and PostgreSQL, some ML exposure preferred.",
]


def test_build_tfidf_matrix_row_count():
    matrix, _ = build_tfidf_matrix(SAMPLE_DESCRIPTIONS)
    assert matrix.shape[0] == len(SAMPLE_DESCRIPTIONS)


def test_build_tfidf_matrix_max_features():
    matrix, _ = build_tfidf_matrix(SAMPLE_DESCRIPTIONS)
    assert matrix.shape[1] <= 5000


def test_build_tfidf_matrix_is_sparse():
    from scipy.sparse import issparse
    matrix, _ = build_tfidf_matrix(SAMPLE_DESCRIPTIONS)
    assert issparse(matrix)


def test_compute_skill_frequencies_python_present():
    freqs = compute_skill_frequencies(SAMPLE_DESCRIPTIONS)
    assert "python" in freqs
    assert freqs["python"] == 100.0  # all 5 descriptions mention python


def test_compute_skill_frequencies_sorted_descending():
    freqs = compute_skill_frequencies(SAMPLE_DESCRIPTIONS)
    values = list(freqs.values())
    assert values == sorted(values, reverse=True)


def test_compute_skill_frequencies_empty_list():
    assert compute_skill_frequencies([]) == {}


def test_compute_skill_frequencies_excludes_zero_count_skills():
    freqs = compute_skill_frequencies(["This description mentions nothing relevant."])
    # All values should be > 0 (zero-count skills excluded)
    assert all(v > 0 for v in freqs.values())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'ml.features'`

- [ ] **Step 3: Create `ml/features.py`**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from scraper.parser import SKILL_KEYWORDS


def build_tfidf_matrix(descriptions: list[str]) -> tuple[spmatrix, TfidfVectorizer]:
    """
    Fit TF-IDF on job descriptions.
    Returns (sparse matrix, fitted vectorizer).
    min_df=1 used for small datasets; increase to 2 once you have 50+ jobs.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(descriptions)
    return matrix, vectorizer


def compute_skill_frequencies(descriptions: list[str]) -> dict[str, float]:
    """
    Return skill keyword frequencies as percentage of jobs mentioning each skill.
    Skills with 0 mentions are excluded. Result is sorted descending by frequency.
    """
    total = len(descriptions)
    if total == 0:
        return {}
    counts: dict[str, int] = {}
    for desc in descriptions:
        desc_lower = desc.lower()
        for skill in SKILL_KEYWORDS:
            if skill in desc_lower:
                counts[skill] = counts.get(skill, 0) + 1
    return {
        skill: round(count / total * 100, 1)
        for skill, count in sorted(counts.items(), key=lambda x: -x[1])
        if count > 0
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/features.py tests/test_features.py
git commit -m "feat: TF-IDF vectorization and skill frequency extraction"
```

---

## Task 6: Clustering + PCA Visualization + Tests

**Files:**
- Create: `ml/clustering.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_clustering.py`:

```python
import numpy as np
from scipy.sparse import csr_matrix
from ml.clustering import cluster_jobs, cluster_summary, find_optimal_k


def _make_matrix(n_samples: int = 40, n_features: int = 50) -> csr_matrix:
    rng = np.random.RandomState(42)
    return csr_matrix(rng.rand(n_samples, n_features))


def test_cluster_jobs_returns_label_array():
    matrix = _make_matrix(40, 50)
    _, labels = cluster_jobs(matrix, k=3)
    assert len(labels) == 40


def test_cluster_jobs_label_range():
    matrix = _make_matrix(40, 50)
    _, labels = cluster_jobs(matrix, k=4)
    assert set(labels).issubset({0, 1, 2, 3})


def test_cluster_summary_counts_sum_to_total():
    labels = np.array([0, 0, 1, 1, 2, 2, 2])
    summary = cluster_summary(labels)
    assert sum(v["count"] for v in summary.values()) == 7


def test_cluster_summary_percentages_sum_to_100():
    labels = np.array([0, 0, 1, 1, 2])
    summary = cluster_summary(labels)
    total_pct = sum(v["pct"] for v in summary.values())
    assert abs(total_pct - 100.0) < 0.5


def test_find_optimal_k_returns_int_in_range():
    matrix = _make_matrix(40, 50)
    k = find_optimal_k(matrix, k_range=range(2, 5))
    assert isinstance(k, int)
    assert 2 <= k <= 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_clustering.py -v
```

Expected: `ModuleNotFoundError: No module named 'ml.clustering'`

- [ ] **Step 3: Create `ml/clustering.py`**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.sparse import spmatrix


def find_optimal_k(matrix: spmatrix, k_range: range = range(3, 9)) -> int:
    """
    Choose k using silhouette score. Higher is better.
    Uses sample_size=min(500, n_samples) to keep it fast on large datasets.
    """
    n_samples = matrix.shape[0]
    if n_samples < max(k_range):
        k_range = range(2, max(3, n_samples // 2))

    scores: dict[int, float] = {}
    for k in k_range:
        if k >= n_samples:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        sample_size = min(500, n_samples)
        scores[k] = silhouette_score(matrix, labels, sample_size=sample_size, random_state=42)

    return max(scores, key=scores.get) if scores else 3


def cluster_jobs(matrix: spmatrix, k: int) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans with given k. Returns (fitted model, label array)."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)
    return km, labels


def save_pca_plot(
    matrix: spmatrix, labels: np.ndarray, output_path: str = "cluster_plot.png"
) -> None:
    """Reduce to 2D with PCA and save a scatter plot to output_path."""
    dense = matrix.toarray()
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(dense)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=60)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("AI/ML Job Listing Clusters (PCA 2D Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def cluster_summary(labels: np.ndarray) -> dict[int, dict]:
    """Return count and percentage for each cluster ID."""
    total = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    return {
        int(k): {"count": int(c), "pct": round(c / total * 100, 1)}
        for k, c in zip(unique, counts)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_clustering.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/clustering.py tests/test_clustering.py
git commit -m "feat: KMeans clustering with silhouette k-selection and PCA visualization"
```

---

## Task 7: Seniority Classifier + Tests

**Files:**
- Create: `ml/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_classifier.py`:

```python
import numpy as np
from scipy.sparse import csr_matrix
from ml.classifier import label_seniority, compare_classifiers, train_best_classifier, predict_seniority


def test_label_seniority_senior_title():
    assert label_seniority("Senior ML Engineer", "") == "senior"


def test_label_seniority_sr_abbreviation():
    assert label_seniority("Sr. AI Engineer", "") == "senior"


def test_label_seniority_lead():
    assert label_seniority("Lead ML Engineer", "") == "senior"


def test_label_seniority_junior_title():
    assert label_seniority("Junior Data Scientist", "") == "junior"


def test_label_seniority_jr_abbreviation():
    assert label_seniority("Jr. NLP Engineer", "") == "junior"


def test_label_seniority_entry_level_in_description():
    assert label_seniority("Data Scientist", "This is an entry level position.") == "junior"


def test_label_seniority_mid_is_default():
    assert label_seniority("ML Engineer", "Python and ML experience required.") == "mid"


def test_compare_classifiers_returns_three_models():
    rng = np.random.RandomState(42)
    X = csr_matrix(rng.rand(60, 30))
    y = ["junior"] * 20 + ["mid"] * 20 + ["senior"] * 20
    scores = compare_classifiers(X, y)
    assert set(scores.keys()) == {"LogisticRegression", "LinearSVC", "RandomForest"}


def test_compare_classifiers_scores_between_0_and_1():
    rng = np.random.RandomState(42)
    X = csr_matrix(rng.rand(60, 30))
    y = ["junior"] * 20 + ["mid"] * 20 + ["senior"] * 20
    scores = compare_classifiers(X, y)
    for name, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{name} score {score} out of range"


def test_predict_seniority_returns_labels():
    rng = np.random.RandomState(42)
    X = csr_matrix(rng.rand(60, 30))
    y = ["junior"] * 20 + ["mid"] * 20 + ["senior"] * 20
    model, le = train_best_classifier(X, y)
    predictions = predict_seniority(model, le, X)
    assert len(predictions) == 60
    assert all(p in {"junior", "mid", "senior"} for p in predictions)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_classifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'ml.classifier'`

- [ ] **Step 3: Create `ml/classifier.py`**

```python
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import spmatrix

SENIOR_RE = re.compile(
    r"\b(senior|sr\.?|lead|principal|staff|head of|vp|director|architect)\b", re.I
)
JUNIOR_RE = re.compile(
    r"\b(junior|jr\.?|entry[\s\-]?level|graduate|intern|associate|0[\s\-]?1 years?)\b", re.I
)


def label_seniority(title: str, description: str) -> str:
    """
    Rule-based seniority classifier using title + description patterns.
    Title takes precedence; description is checked only if title is neutral.
    Returns 'junior', 'mid', or 'senior'.
    """
    if SENIOR_RE.search(title):
        return "senior"
    if JUNIOR_RE.search(title):
        return "junior"
    if SENIOR_RE.search(description):
        return "senior"
    if JUNIOR_RE.search(description):
        return "junior"
    return "mid"


def compare_classifiers(X: spmatrix, y: list[str]) -> dict[str, float]:
    """
    Evaluate LR, LinearSVC, and RandomForest on (X, y) using 5-fold CV.
    Returns {model_name: mean_f1_macro_score}.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC": LinearSVC(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results: dict[str, float] = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y_enc, cv=5, scoring="f1_macro")
        results[name] = round(float(scores.mean()), 3)
    return results


def train_best_classifier(
    X: spmatrix, y: list[str]
) -> tuple[LogisticRegression, LabelEncoder]:
    """Train LogisticRegression (best for text tasks) and return (model, encoder)."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y_enc)
    return model, le


def predict_seniority(
    model: LogisticRegression, le: LabelEncoder, X: spmatrix
) -> list[str]:
    """Return predicted seniority label strings for each row in X."""
    y_pred = model.predict(X)
    return list(le.inverse_transform(y_pred))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_classifier.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/classifier.py tests/test_classifier.py
git commit -m "feat: rule-based seniority labeling and LR/SVM/RF classifier comparison"
```

---

## Task 8: LLM Insight Generator

**Files:**
- Create: `llm/insight_generator.py`

Tested manually (requires ANTHROPIC_API_KEY).

- [ ] **Step 1: Create `llm/insight_generator.py`**

```python
import os
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()

_PROMPT_TEMPLATE = """\
You are a market intelligence analyst specializing in AI/ML talent trends. \
Based on the structured job market data below, write a concise 2–3 paragraph briefing \
for a technical audience. Focus on the most notable trends, dominant skills, and what \
the cluster distribution reveals about where AI hiring is headed. \
Write in an analytical, confident tone. Flowing paragraphs only — no bullet points.

Market Data
-----------
Analysis date: {date_range}
Total jobs analyzed: {total_jobs}
Top 10 skills by mention rate: {top_skills}
Job type clusters: {cluster_breakdown}
Seniority distribution: {seniority_breakdown}
"""


def build_prompt(
    total_jobs: int,
    top_skills: dict[str, float],
    cluster_breakdown: dict[int, dict],
    cluster_names: dict[int, str],
    seniority_counts: dict[str, int],
    date_range: str,
) -> str:
    """Build the structured prompt string from pipeline output data."""
    top_skills_str = ", ".join(
        f"{skill} ({pct}%)" for skill, pct in list(top_skills.items())[:10]
    )
    cluster_str = "; ".join(
        f"{cluster_names.get(k, f'Cluster {k}')} ({v['pct']}%)"
        for k, v in cluster_breakdown.items()
    )
    total_seniority = sum(seniority_counts.values()) or 1
    seniority_str = ", ".join(
        f"{label} {round(count / total_seniority * 100, 1)}%"
        for label, count in seniority_counts.items()
    )
    return _PROMPT_TEMPLATE.format(
        date_range=date_range,
        total_jobs=total_jobs,
        top_skills=top_skills_str,
        cluster_breakdown=cluster_str,
        seniority_breakdown=seniority_str,
    )


def generate_insight(prompt: str) -> str:
    """
    Call Claude API with the given prompt and return the generated insight text.
    Retries once on rate limit errors with a 10-second delay.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for attempt in range(2):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except anthropic.RateLimitError:
            if attempt == 0:
                time.sleep(10)
            else:
                raise
```

- [ ] **Step 2: Test manually**

Copy `.env.example` to `.env` and fill in your `ANTHROPIC_API_KEY`, then:

```bash
python -c "
from llm.insight_generator import build_prompt, generate_insight
prompt = build_prompt(
    total_jobs=50,
    top_skills={'python': 95.0, 'pytorch': 72.0, 'langchain': 60.0, 'llm': 58.0},
    cluster_breakdown={0: {'count': 20, 'pct': 40.0}, 1: {'count': 18, 'pct': 36.0}, 2: {'count': 12, 'pct': 24.0}},
    cluster_names={0: 'LLM/GenAI', 1: 'Classical ML', 2: 'MLOps'},
    seniority_counts={'junior': 8, 'mid': 22, 'senior': 20},
    date_range='2026-04-23',
)
print(generate_insight(prompt))
"
```

Expected: 2-3 paragraphs of market analysis text.

- [ ] **Step 3: Commit**

```bash
git add llm/insight_generator.py
git commit -m "feat: Claude API insight generator with templated prompt"
```

---

## Task 9: Google Sheets Reporter

**Files:**
- Create: `reporter/sheets.py`

Tested manually (requires Google Service Account setup).

- [ ] **Step 1: Set up Google Cloud credentials (one-time setup)**

1. Go to https://console.cloud.google.com → create a project (e.g., "job-pipeline")
2. Enable "Google Sheets API" and "Google Drive API"
3. Create a Service Account → download JSON key → save as `service_account.json` in project root
4. Create a Google Sheet → share it with the service account email (Editor)
5. Copy the Sheet ID from the URL (the long string between `/d/` and `/edit`) → add to `.env` as `GOOGLE_SHEETS_ID`

- [ ] **Step 2: Create `reporter/sheets.py`**

```python
import os
import logging
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _get_client() -> gspread.Client:
    creds = Credentials.from_service_account_file(
        os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"], scopes=_SCOPES
    )
    return gspread.authorize(creds)


def push_report(
    llm_report: str,
    skill_frequencies: dict[str, float],
    cluster_breakdown: dict[int, dict],
    cluster_names: dict[int, str],
    seniority_counts: dict[str, int],
) -> str:
    """
    Push pipeline results to Google Sheets.
    Creates a new tab named by today's date (e.g., "2026-04-23").
    If the tab already exists, it is cleared and overwritten.
    Returns the spreadsheet URL, or empty string on failure.
    """
    try:
        client = _get_client()
        spreadsheet = client.open_by_key(os.environ["GOOGLE_SHEETS_ID"])
        tab_name = datetime.utcnow().strftime("%Y-%m-%d")

        try:
            worksheet = spreadsheet.add_worksheet(title=tab_name, rows=200, cols=10)
        except gspread.exceptions.APIError:
            worksheet = spreadsheet.worksheet(tab_name)
            worksheet.clear()

        rows: list[list] = []

        rows.append(["AI/ML Job Market Intelligence Report"])
        rows.append([llm_report])
        rows.append([])

        rows.append(["Top Skills", "% of Jobs"])
        for skill, pct in list(skill_frequencies.items())[:20]:
            rows.append([skill, pct])
        rows.append([])

        rows.append(["Cluster", "Job Count", "% of Total"])
        for cluster_id, stats in cluster_breakdown.items():
            name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            rows.append([name, stats["count"], stats["pct"]])
        rows.append([])

        rows.append(["Seniority", "Count"])
        for label, count in seniority_counts.items():
            rows.append([label, count])

        worksheet.update(rows, "A1")
        logger.info(f"Pushed report to Google Sheets tab: {tab_name}")
        return spreadsheet.url

    except Exception as e:
        logger.error(f"Google Sheets push failed: {e}")
        return ""
```

- [ ] **Step 3: Test manually**

```bash
python -c "
from reporter.sheets import push_report
url = push_report(
    llm_report='Test report: Python dominates AI hiring.',
    skill_frequencies={'python': 95.0, 'pytorch': 72.0},
    cluster_breakdown={0: {'count': 30, 'pct': 60.0}, 1: {'count': 20, 'pct': 40.0}},
    cluster_names={0: 'LLM/GenAI', 1: 'MLOps'},
    seniority_counts={'junior': 5, 'mid': 15, 'senior': 10},
)
print('Sheet URL:', url)
"
```

Expected: sheet URL printed, new dated tab visible in Google Sheet.

- [ ] **Step 4: Commit**

```bash
git add reporter/sheets.py
git commit -m "feat: Google Sheets reporter using gspread service account auth"
```

---

## Task 10: Pipeline Orchestrator + Integration Test

**Files:**
- Create: `pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_pipeline.py`:

```python
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base


@pytest.fixture
def sample_jobs():
    with open("tests/fixtures/sample_jobs.json") as f:
        return json.load(f)


@pytest.fixture
def sqlite_session(tmp_path):
    """In-memory SQLite session for tests — no PostgreSQL needed."""
    url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session, url
    session.close()


@pytest.mark.asyncio
async def test_pipeline_processes_fixture_jobs(sample_jobs, sqlite_session):
    session, db_url = sqlite_session

    with patch("pipeline.scrape_all_sources", new_callable=AsyncMock, return_value=sample_jobs), \
         patch("pipeline.get_session", return_value=session), \
         patch("pipeline.generate_insight", return_value="Strong demand for LLM skills."), \
         patch("pipeline.push_report", return_value="https://sheets.google.com/test"):

        from pipeline import run_pipeline
        result = await run_pipeline()

    assert result["jobs_scraped"] == 20
    assert result["total_jobs"] == 20
    assert result["clusters_found"] >= 2
    assert "LogisticRegression" in result["classifier_scores"]
    assert result["sheet_url"] == "https://sheets.google.com/test"


@pytest.mark.asyncio
async def test_pipeline_deduplicates_on_second_run(sample_jobs, sqlite_session):
    session, db_url = sqlite_session

    with patch("pipeline.scrape_all_sources", new_callable=AsyncMock, return_value=sample_jobs), \
         patch("pipeline.get_session", return_value=session), \
         patch("pipeline.generate_insight", return_value="Report."), \
         patch("pipeline.push_report", return_value=""):

        from pipeline import run_pipeline
        result1 = await run_pipeline()
        result2 = await run_pipeline()

    # Second run should save 0 new jobs (all duplicates)
    assert result1["jobs_scraped"] == 20
    assert result2["jobs_scraped"] == 0
    assert result2["total_jobs"] == 20
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: Create `pipeline.py`**

```python
import asyncio
import logging
from datetime import datetime, timezone

from db.models import Job, JobFeatures, PipelineRun
from db.session import get_session
from scraper.playwright_scraper import scrape_all_sources
from scraper.parser import normalize_job, extract_skills
from ml.features import build_tfidf_matrix, compute_skill_frequencies
from ml.clustering import find_optimal_k, cluster_jobs, cluster_summary, save_pca_plot
from ml.classifier import (
    label_seniority,
    compare_classifiers,
    train_best_classifier,
    predict_seniority,
)
from llm.insight_generator import build_prompt, generate_insight
from reporter.sheets import push_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_pipeline() -> dict:
    """
    Run the full pipeline: scrape → normalize → save → ML → LLM → Sheets.
    Returns a summary dict with run statistics.
    """
    session = get_session()
    try:
        # 1. Scrape
        logger.info("Starting scrape...")
        raw_jobs = await scrape_all_sources()

        # 2. Normalize, deduplicate, persist
        jobs_saved = 0
        all_jobs: list[Job] = []

        for raw in raw_jobs:
            normalized = normalize_job(raw)
            if not normalized:
                continue
            existing = (
                session.query(Job).filter_by(source_url=normalized["source_url"]).first()
            )
            if existing:
                all_jobs.append(existing)
                continue
            job = Job(**normalized)
            session.add(job)
            try:
                session.flush()
                jobs_saved += 1
                all_jobs.append(job)
            except Exception:
                session.rollback()

        session.commit()
        logger.info(f"Saved {jobs_saved} new jobs. Total in pipeline: {len(all_jobs)}")

        if len(all_jobs) < 10:
            raise ValueError(
                f"Not enough jobs to run ML pipeline ({len(all_jobs)} found, need 10+). "
                "Run the scraper first or check your data source."
            )

        # 3. ML pipeline
        descriptions = [j.description or "" for j in all_jobs]
        titles = [j.title or "" for j in all_jobs]

        matrix, _ = build_tfidf_matrix(descriptions)
        skill_freqs = compute_skill_frequencies(descriptions)

        k = find_optimal_k(matrix)
        _, cluster_labels = cluster_jobs(matrix, k)
        save_pca_plot(matrix, cluster_labels, "cluster_plot.png")
        logger.info(f"Clustered into {k} groups. Plot saved to cluster_plot.png")

        seniority_labels = [
            label_seniority(t, d) for t, d in zip(titles, descriptions)
        ]
        classifier_scores = compare_classifiers(matrix, seniority_labels)
        logger.info(f"Classifier comparison: {classifier_scores}")

        best_model, le = train_best_classifier(matrix, seniority_labels)
        predicted_seniority = predict_seniority(best_model, le, matrix)

        # Save features back to DB
        for job, cluster_id, seniority in zip(all_jobs, cluster_labels, predicted_seniority):
            skills = extract_skills(job.description or "")
            feat = session.query(JobFeatures).filter_by(job_id=job.id).first()
            if feat:
                feat.cluster_id = int(cluster_id)
                feat.seniority_label = seniority
                feat.skill_vector = skills
            else:
                session.add(
                    JobFeatures(
                        job_id=job.id,
                        cluster_id=int(cluster_id),
                        seniority_label=seniority,
                        skill_vector=skills,
                    )
                )
        session.commit()

        # 4. LLM insight
        # Cluster names: inspect cluster_plot.png and update these manually
        cluster_names = {i: f"Cluster {i}" for i in range(k)}
        breakdown = cluster_summary(cluster_labels)
        seniority_counts = {
            label: seniority_labels.count(label)
            for label in ["junior", "mid", "senior"]
        }
        date_range = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        prompt = build_prompt(
            total_jobs=len(all_jobs),
            top_skills=skill_freqs,
            cluster_breakdown=breakdown,
            cluster_names=cluster_names,
            seniority_counts=seniority_counts,
            date_range=date_range,
        )
        llm_report = generate_insight(prompt)
        logger.info("LLM insight generated.")

        # 5. Push to Google Sheets
        sheet_url = push_report(
            llm_report=llm_report,
            skill_frequencies=skill_freqs,
            cluster_breakdown=breakdown,
            cluster_names=cluster_names,
            seniority_counts=seniority_counts,
        )

        # 6. Record run
        session.add(PipelineRun(jobs_scraped=jobs_saved, llm_report=llm_report))
        session.commit()

        return {
            "jobs_scraped": jobs_saved,
            "total_jobs": len(all_jobs),
            "clusters_found": k,
            "classifier_scores": classifier_scores,
            "sheet_url": sheet_url,
        }

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

- [ ] **Step 4: Run integration tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrator and integration tests"
```

---

## Task 11: FastAPI Endpoint

**Files:**
- Create: `api/main.py`

- [ ] **Step 1: Create `api/main.py`**

```python
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException
from db.session import create_tables
from pipeline import run_pipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    logger.info("Database tables verified.")
    yield


app = FastAPI(
    title="AI/ML Job Market Intelligence Pipeline",
    description="Scrapes job listings, applies ML clustering/classification, generates LLM insights, and pushes to Google Sheets.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run-pipeline")
async def trigger_pipeline():
    """
    Trigger the full pipeline:
    1. Scrape AI/ML job listings
    2. Cluster and classify
    3. Generate LLM insight
    4. Push to Google Sheets
    Returns a run summary.
    """
    try:
        result = await run_pipeline()
        return {"status": "success", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
```

- [ ] **Step 2: Start the server and test manually**

```bash
# In one terminal: start PostgreSQL
docker-compose up -d db

# In another terminal: start the API
uvicorn api.main:app --reload --port 8000
```

Test health check:
```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

Test pipeline trigger:
```bash
curl -X POST http://localhost:8000/run-pipeline
```

Expected: JSON with `jobs_scraped`, `clusters_found`, `sheet_url`.

- [ ] **Step 3: Commit**

```bash
git add api/main.py
git commit -m "feat: FastAPI endpoint POST /run-pipeline with lifespan DB init"
```

---

## Task 12: README + Demo Prep

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

````markdown
# AI/ML Job Market Intelligence Pipeline

An end-to-end ML engineering project that scrapes public AI/ML job listings, applies clustering and classification, generates market insights using Claude AI, and delivers automated reports to Google Sheets.

## What it does

1. **Scrapes** AI/ML job listings from public sources using Playwright (async, with retry)
2. **Stores** all data in PostgreSQL via SQLAlchemy
3. **Clusters** jobs using KMeans (k selected via silhouette score) on TF-IDF embeddings
4. **Classifies** seniority (junior/mid/senior) with Logistic Regression, SVM, and Random Forest — then compares F1 scores
5. **Generates** a natural-language market briefing using Claude API
6. **Pushes** skill tables, cluster breakdown, and LLM report to Google Sheets automatically

## Skills Demonstrated

| Skill | Implementation |
|---|---|
| LLMs | Claude API insight generator with templated prompts |
| Clustering | KMeans with silhouette-based k selection + PCA visualization |
| Classification | LR vs SVM vs RandomForest seniority classifier with F1 comparison |
| Web scraping | Async Playwright with pagination and retry logic |
| Data pipelines | End-to-end `pipeline.py` orchestrator |
| Databases | PostgreSQL + SQLAlchemy ORM + Alembic migrations |
| Async programming | `asyncio.gather` for concurrent scraping + FastAPI |
| Google Sheets | gspread service account automation with dated tabs |

## Setup

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- Anthropic API key
- Google Cloud project with Sheets API enabled + service account JSON

### Install

```bash
git clone <repo-url>
cd ai-ml-pipeline
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, GOOGLE_SHEETS_ID, GOOGLE_SERVICE_ACCOUNT_FILE in .env
```

### Run with Docker

```bash
docker-compose up -d db
alembic upgrade head
uvicorn api.main:app --port 8000
```

### Trigger the pipeline

```bash
curl -X POST http://localhost:8000/run-pipeline
```

### Run tests

```bash
pytest -v
```

## Output

- `cluster_plot.png` — PCA scatter plot of job clusters
- Google Sheet — dated tab with skill frequencies, cluster breakdown, and LLM-generated briefing
- API response — run summary with jobs scraped, clusters found, and classifier scores
````

- [ ] **Step 2: Run full test suite one final time**

```bash
pytest -v
```

Expected: all tests PASS with no warnings.

- [ ] **Step 3: Do a full end-to-end live demo run**

```bash
docker-compose up -d db
alembic upgrade head
uvicorn api.main:app --port 8000 &
curl -X POST http://localhost:8000/run-pipeline | python -m json.tool
```

Verify:
- Cluster plot saved as `cluster_plot.png`
- Google Sheet has a new dated tab with data
- Response JSON shows correct counts

- [ ] **Step 4: Final commit**

```bash
git add README.md cluster_plot.png
git commit -m "feat: README, demo prep, project complete"
```

---

## Self-Review Notes

- All spec sections covered: scraping (Task 4), DB (Task 2), ML clustering (Task 6), ML classification (Task 7), LLM (Task 8), Google Sheets (Task 9), FastAPI async (Task 11), pipeline (Task 10)
- `SKILL_KEYWORDS` defined once in `scraper/parser.py` and imported by `ml/features.py` — no duplication
- `normalize_job` handles both `title` and `position` keys (RemoteOK API uses `position`)
- SQLite used in tests to avoid PostgreSQL dependency in CI
- Mocks in integration test isolate scraper + LLM + Sheets — only ML logic runs against real code
- `cluster_names` dict in `pipeline.py` uses generic names by default — engineer should inspect `cluster_plot.png` and rename clusters before the interview demo

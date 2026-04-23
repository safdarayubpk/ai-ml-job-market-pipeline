# AI/ML Job Market Intelligence Pipeline

A portfolio project demonstrating end-to-end ML engineering: async web scraping, unsupervised clustering, supervised classification, LLM-generated insights, and automated Google Sheets reporting — all behind a single FastAPI endpoint.

---

## What It Does

1. **Scrapes** public AI/ML job listings from RemoteOK using async Playwright
2. **Normalizes and deduplicates** jobs in PostgreSQL via SQLAlchemy ORM
3. **Clusters** job descriptions with KMeans + TF-IDF (silhouette score selects optimal k)
4. **Classifies** seniority (junior/mid/senior) by comparing Logistic Regression, LinearSVC, and Random Forest via 5-fold CV F1 score
5. **Generates** a 2-3 paragraph market intelligence briefing using Groq API (`llama-3.3-70b-versatile`)
6. **Pushes** the full report to Google Sheets (new dated tab per run)
7. **Exposes** everything via `POST /run-pipeline`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Scraping | Playwright (async), aiohttp retry |
| Storage | PostgreSQL, SQLAlchemy 2, Alembic |
| ML | scikit-learn (TF-IDF, KMeans, LR, SVC, RF), scipy, matplotlib |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| Reporting | gspread + Google Service Account |
| API | FastAPI, uvicorn |
| Testing | pytest, pytest-asyncio, SQLite in-memory |
| Infra | Docker Compose, Python 3.12 |

---

## Project Structure

```
.
├── api/
│   └── main.py              # FastAPI app — /health, /run-pipeline
├── db/
│   ├── models.py            # Job, JobFeatures, PipelineRun (SQLAlchemy)
│   └── session.py           # Engine + session factory
├── scraper/
│   ├── playwright_scraper.py # Async Playwright scraping with retry
│   └── parser.py            # normalize_job(), extract_skills()
├── ml/
│   ├── features.py          # TF-IDF matrix, skill frequency analysis
│   ├── clustering.py        # KMeans, silhouette k-selection, PCA plot
│   └── classifier.py        # Rule-based + ML seniority labeling/classification
├── llm/
│   └── insight_generator.py # Groq API — build_prompt(), generate_insight()
├── reporter/
│   └── sheets.py            # Google Sheets push via gspread
├── pipeline.py              # Full async pipeline orchestrator
├── tests/
│   ├── fixtures/
│   │   └── sample_jobs.json # 20 realistic AI/ML job fixtures
│   ├── test_db.py
│   ├── test_scraper.py
│   ├── test_features.py
│   ├── test_clustering.py
│   ├── test_classifier.py
│   └── test_pipeline.py     # Integration tests (SQLite + mocks)
├── alembic/                 # DB migrations
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.12+
- Docker + Docker Compose (for PostgreSQL)
- Google Cloud service account JSON (see below)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone and create virtual environment

```bash
git clone <repo>
cd ai-ml
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment

Create `.env` at the project root:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_SHEETS_ID=your_spreadsheet_id
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
DATABASE_URL=postgresql://pipeline:pipeline@localhost:5432/jobmarket
```

Place your Google service account JSON as `service_account.json` at the project root. Share your Google Sheet with the service account email.

### 3. Start PostgreSQL

```bash
docker-compose up -d
```

### 4. Run database migrations

```bash
alembic upgrade head
```

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

---

## Running the Pipeline

```bash
curl -X POST http://localhost:8000/run-pipeline
```

Example response:

```json
{
  "status": "success",
  "jobs_scraped": 47,
  "total_jobs": 47,
  "clusters_found": 4,
  "classifier_scores": {
    "LogisticRegression": 0.81,
    "LinearSVC": 0.79,
    "RandomForest": 0.76
  },
  "sheet_url": "https://docs.google.com/spreadsheets/d/..."
}
```

A PCA cluster plot is saved to `cluster_plot.png`. The Google Sheet gets a new tab named by today's date containing the LLM report, top skills, cluster breakdown, and seniority distribution.

---

## Running Tests

```bash
# Full suite (no external services needed — uses SQLite + mocks)
pytest -v

# Integration tests only
pytest tests/test_pipeline.py -v
```

36 tests, all passing.

---

## Key Design Decisions

**Silhouette score for k-selection:** Rather than hardcoding the number of clusters, `find_optimal_k()` tries k=2..7 and picks the k with the highest silhouette score. This adapts to whatever shape the scraped data takes on any given day.

**Classifier comparison:** All three classifiers (LR, LinearSVC, RF) are evaluated via 5-fold CV. The best F1 score is reported in the API response so you can see which model won on that run's data distribution.

**SQLite in tests:** The integration tests run against SQLite (not PostgreSQL) via pytest fixtures, so the full pipeline logic can be verified without a running database.

**Groq free tier:** Uses `llama-3.3-70b-versatile` via Groq's free API — no credit card required. Rate-limit retries are built in with a 10-second backoff.

**Deduplication by URL:** Jobs are keyed on `source_url`. A second pipeline run fetches no new jobs and skips straight to ML on the existing corpus — cheap and idempotent.

# AI/ML Job Market Intelligence Pipeline

A portfolio project demonstrating end-to-end ML engineering: async API scraping, unsupervised clustering, supervised classification, LLM-generated insights, and automated Google Sheets reporting — all behind a single FastAPI endpoint.

**Live demo result:** 97 real AI/ML jobs scraped, clustered into 8 groups, seniority classified, Groq LLM report generated, pushed to Google Sheets — all in one API call.

---

## What It Does

1. **Scrapes** live AI/ML job listings from the RemoteOK JSON API across 5 tags (`ai`, `machine-learning`, `llm`, `nlp`, `computer-vision`) using async `aiohttp`
2. **Normalizes and deduplicates** jobs in PostgreSQL via SQLAlchemy ORM — second run adds only new jobs
3. **Clusters** job descriptions with KMeans + TF-IDF (silhouette score automatically selects optimal k)
4. **Classifies** seniority (junior/mid/senior) by comparing Logistic Regression, LinearSVC, and Random Forest via 5-fold CV F1 score
5. **Generates** a 2-3 paragraph market intelligence briefing using Groq API (`llama-3.3-70b-versatile`, free tier)
6. **Pushes** the full report to Google Sheets — new dated tab per run (e.g. `2026-04-23`)
7. **Exposes** everything via `POST /run-pipeline`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Scraping | aiohttp (async), RemoteOK JSON API, BeautifulSoup4 |
| Storage | PostgreSQL, SQLAlchemy 2, Alembic |
| ML | scikit-learn (TF-IDF, KMeans, LR, LinearSVC, RF), scipy, matplotlib |
| LLM | Groq API — `llama-3.3-70b-versatile` (free tier) |
| Reporting | gspread + Google Service Account |
| API | FastAPI, uvicorn |
| Testing | pytest, pytest-asyncio, SQLite in-memory |
| Runtime | Python 3.12, virtual environment |

---

## Project Structure

```
.
├── api/
│   └── main.py              # FastAPI app — GET /health, POST /run-pipeline
├── db/
│   ├── models.py            # Job, JobFeatures, PipelineRun (SQLAlchemy ORM)
│   ├── session.py           # Engine + session factory, reads DATABASE_URL from .env
│   └── migrations/          # Alembic migration scripts
├── scraper/
│   ├── playwright_scraper.py # Async aiohttp scraper — RemoteOK JSON API, 5 tags
│   └── parser.py            # normalize_job(), extract_skills(), SKILL_KEYWORDS
├── ml/
│   ├── features.py          # TF-IDF matrix (5000 features), skill frequency analysis
│   ├── clustering.py        # KMeans, silhouette k-selection (k=2..7), PCA plot
│   └── classifier.py        # Rule-based labeling + LR/LinearSVC/RF comparison
├── llm/
│   └── insight_generator.py # build_prompt(), generate_insight() via Groq API
├── reporter/
│   └── sheets.py            # push_report() — creates dated tab, writes report + stats
├── pipeline.py              # Full async orchestrator (scrape→DB→ML→LLM→Sheets)
├── tests/
│   ├── fixtures/
│   │   └── sample_jobs.json # 20 realistic AI/ML job fixtures for tests
│   ├── test_db.py           # DB model + session tests
│   ├── test_parser.py       # normalize_job, extract_skills tests
│   ├── test_features.py     # TF-IDF + skill frequency tests
│   ├── test_clustering.py   # KMeans + silhouette tests
│   ├── test_classifier.py   # Seniority labeling + classifier tests
│   └── test_pipeline.py     # Integration tests (SQLite + mocked LLM/Sheets)
├── docker-compose.yml       # PostgreSQL + API containers
├── requirements.txt
└── .env                     # Credentials (gitignored)
```

---

## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL running locally (or Docker)
- Groq API key — free at [console.groq.com](https://console.groq.com)
- Google Cloud service account JSON with Sheets + Drive access

### 1. Clone and create virtual environment

```bash
git clone <repo>
cd ai-ml
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` at the project root:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_SHEETS_ID=your_spreadsheet_id
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
DATABASE_URL=postgresql://pipeline:pipeline@localhost:5433/jobmarket
```

Place your Google service account JSON as `service_account.json` at the project root. Share your Google Sheet with the service account's email address (Editor access).

### 3. Create the PostgreSQL database

```bash
sudo -u postgres psql -c "CREATE USER pipeline WITH PASSWORD 'pipeline';"
sudo -u postgres createdb -O pipeline jobmarket
```

Or with Docker:

```bash
docker compose up -d db
```

### 4. Start the API

```bash
uvicorn api.main:app --port 8000 &
```

Database tables are created automatically on first startup.

---

## Running the Pipeline

```bash
curl -X POST http://localhost:8000/run-pipeline
```

Real output from live run (2026-04-23):

```json
{
  "status": "success",
  "jobs_scraped": 97,
  "total_jobs": 97,
  "clusters_found": 8,
  "classifier_scores": {
    "LogisticRegression": 0.417,
    "LinearSVC": 0.404,
    "RandomForest": 0.351
  },
  "sheet_url": "https://docs.google.com/spreadsheets/d/1HcA5Y7P8BWZ3lGNvt7G6KPcB6WSTfr8620YZTbII09Y"
}
```

A PCA cluster plot is saved to `cluster_plot.png`. The Google Sheet gets a new tab named by today's date containing:
- LLM-generated market intelligence report (2-3 paragraphs)
- Top 20 skills by mention rate
- Cluster breakdown (job count + % per cluster)
- Seniority distribution (junior / mid / senior counts)

### Daily usage

Run these two commands whenever you want fresh data:

```bash
cd ~/Desktop/claude/ai-ml && source venv/bin/activate
curl -X POST http://localhost:8000/run-pipeline
```

The server stays running in the background after the first `uvicorn` start. After a laptop restart, start it again first:

```bash
uvicorn api.main:app --port 8000 &
```

Second and subsequent runs on the same day will show `jobs_scraped: 0` — this is correct. Deduplication prevents re-inserting the same jobs. The ML pipeline still runs on the full corpus and pushes a fresh report.

---

## Running Tests

No external services needed — tests use SQLite in-memory and mock the LLM and Sheets calls.

```bash
# Full suite
pytest -v

# Integration tests only
pytest tests/test_pipeline.py -v
```

**36 tests, all passing.**

---

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Key Design Decisions

**RemoteOK JSON API over Playwright:** The original design used Playwright (headless browser) to scrape HTML. Switched to the RemoteOK JSON API via `aiohttp` — more reliable, faster, no browser install needed, and returns structured data with HTML already parseable via BeautifulSoup.

**Silhouette score for k-selection:** Rather than hardcoding the number of clusters, `find_optimal_k()` tries k=2..7 and picks the k with the highest silhouette score. This adapts automatically to whatever the data looks like on any given day.

**Classifier comparison:** All three classifiers (LR, LinearSVC, RF) are evaluated via 5-fold cross-validation F1 macro score. The best score is reported in the API response so you can see which model won on that run's data distribution.

**SQLite in tests:** Integration tests run against SQLite via pytest fixtures — no PostgreSQL needed in CI. Scraper, LLM, and Sheets are mocked so the ML logic is tested in full isolation.

**Groq free tier:** Uses `llama-3.3-70b-versatile` — no credit card required. Rate-limit retries are built in with a 10-second backoff.

**Deduplication by URL:** Jobs are keyed on `source_url`. Re-running the pipeline never double-counts jobs — it only processes genuinely new listings and appends them to the existing corpus before re-running ML.

**Tables created on startup:** `Base.metadata.create_all()` runs in the FastAPI lifespan handler. No manual migration step needed for a fresh install — just start the server.

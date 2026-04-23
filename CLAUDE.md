# CLAUDE.md — AI/ML Job Market Intelligence Pipeline

## What This Project Is

An end-to-end ML data pipeline that scrapes RemoteOK for AI/ML jobs, clusters them with
K-Means + TF-IDF, classifies seniority with Logistic Regression / LinearSVC / Random Forest,
generates a market intelligence briefing via the Groq API (LLaMA 3.3 70B), and pushes the
full report to a dated Google Sheets tab — all triggered by a single `POST /run-pipeline`.

**Purpose:** Portfolio project demonstrating ML engineering skills aligned with the LimeoX job description.

---

## Essential Commands

```bash
# Activate environment (always do this first)
source venv/bin/activate

# Start the API server
uvicorn api.main:app --port 8000

# Trigger the full pipeline
curl -X POST http://localhost:8000/run-pipeline

# Health check
curl http://localhost:8000/health

# Run all tests (no external services needed)
pytest -v

# Run integration tests only
pytest tests/test_pipeline.py -v

# Start PostgreSQL via Docker
docker compose up -d db

# Start full stack (PostgreSQL + API)
docker compose up
```

---

## Environment Variables

Create a `.env` file at the project root (it is gitignored — never commit it):

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_SHEETS_ID=your_spreadsheet_id
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
DATABASE_URL=postgresql://pipeline:pipeline@localhost:5433/jobmarket
```

**Groq API key:** Free at console.groq.com — no credit card needed.
**Google Sheets ID:** Found in the Sheet URL between `/d/` and `/edit`.

---

## Sensitive Files — Never Commit

- `service_account.json` — Google Cloud service account key
- `.env` — all credentials
- Both are in `.gitignore` already

---

## Architecture

```
RemoteOK JSON API
      ↓ aiohttp (async)
scraper/playwright_scraper.py  — fetches 5 tags: ai, ml, llm, nlp, computer-vision
scraper/parser.py              — normalize_job(), extract_skills(), deduplication by URL
      ↓
db/ (PostgreSQL via SQLAlchemy + Alembic)
  models.py    — Job, JobFeatures, PipelineRun
  session.py   — reads DATABASE_URL from .env
      ↓
ml/features.py     — TF-IDF (5000 features) + skill frequency analysis
ml/clustering.py   — KMeans, silhouette k-selection (k=2..7), PCA plot → cluster_plot.png
ml/classifier.py   — rule-based labeling + LR / LinearSVC / RF 5-fold CV comparison
      ↓
llm/insight_generator.py  — build_prompt() + generate_insight() via Groq API
      ↓
reporter/sheets.py  — push_report() creates dated tab, writes report + stats tables
      ↓
api/main.py         — POST /run-pipeline, GET /health (FastAPI + uvicorn)
pipeline.py         — full async orchestrator wiring all steps together
```

---

## Key Design Decisions

**RemoteOK JSON API (not Playwright HTML scraping):** More reliable, faster, returns structured
data. Switched from original Playwright design after first implementation.

**Silhouette score for k-selection:** `find_optimal_k()` tries k=2..7 and picks the highest
silhouette score. Adapts to data distribution — do not hardcode k.

**Classifier comparison:** All three models (LR, LinearSVC, RF) are evaluated via 5-fold CV F1
macro. The best score is reported in the API response. Do not delete the comparison — it is
a deliberate demo feature.

**Groq LLaMA 3.3 70B (not Claude/OpenAI):** Free tier, no credit card. Rate-limit retries
built in with 10-second backoff.

**Deduplication by `source_url`:** Re-running never double-counts jobs. Second run on the same
day shows `jobs_scraped: 0` — this is correct behavior, not a bug.

**Tables created on startup:** `Base.metadata.create_all()` runs in FastAPI lifespan. No manual
migration step needed for fresh installs.

**SQLite in tests:** Integration tests use SQLite in-memory. Scraper, LLM, and Sheets are
mocked. No PostgreSQL required in CI.

---

## Testing Rules

- Tests live in `tests/` — one file per module
- Test fixtures (sample jobs) are in `tests/fixtures/sample_jobs.json`
- Never mock the ML logic — only mock external I/O (scraper, LLM, Sheets)
- Run `pytest -v` before every commit — 36 tests must all pass
- Tests use `asyncio_mode = auto` (set in `pytest.ini`)

---

## Module Responsibilities

| File | Responsibility |
|---|---|
| `scraper/playwright_scraper.py` | Fetch raw jobs from RemoteOK API async |
| `scraper/parser.py` | Clean, normalize, extract skills from raw jobs |
| `db/models.py` | SQLAlchemy ORM: Job, JobFeatures, PipelineRun |
| `db/session.py` | Engine + session factory |
| `ml/features.py` | TF-IDF matrix + skill frequency counts |
| `ml/clustering.py` | KMeans clustering + PCA visualization |
| `ml/classifier.py` | Seniority labeling + classifier comparison |
| `llm/insight_generator.py` | Groq prompt builder + API call |
| `reporter/sheets.py` | Google Sheets writer via gspread |
| `api/main.py` | FastAPI endpoints |
| `pipeline.py` | Async orchestrator — calls all modules in order |

---

## Superpowers Methodology

This project follows the Superpowers development workflow:

- Spec: `docs/superpowers/specs/2026-04-23-ai-ml-job-market-pipeline-design.md`
- Skills used: `brainstorming` → `writing-plans` → `test-driven-development` → `verification-before-completion`
- New features must go through `superpowers:brainstorming` before any code is written
- Each implementation task must use `superpowers:test-driven-development`
- Run `superpowers:verification-before-completion` before every commit or PR

---

## Common Issues

**`jobs_scraped: 0` on second run** — Expected. Deduplication by URL is working correctly.

**`DATABASE_URL` not found** — Ensure `.env` exists and `python-dotenv` is loading it. The
`db/session.py` calls `load_dotenv()` at import time.

**Google Sheets auth error** — Confirm the service account email has Editor access on the Sheet.
The email is in `service_account.json` under the `client_email` field.

**Groq rate limit** — The insight generator retries once after 10 seconds automatically.
If it fails twice, check your Groq API key in `.env`.

**Alembic migration needed** — If you add columns to models, run:
```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

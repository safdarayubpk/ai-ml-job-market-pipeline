# AI/ML Job Market Intelligence Pipeline — Design Spec

**Date:** 2026-04-23
**Goal:** Build a portfolio project that demonstrates AI/ML engineering skills aligned with a target job description at LimeoX. The system scrapes public AI/ML job listings, analyzes them with ML, generates insights via an LLM, and pushes automated reports to Google Sheets.

---

## 1. Project Overview

**Name:** AI/ML Job Market Intelligence Pipeline

**Purpose:** An end-to-end data intelligence system that monitors the AI/ML job market in real time. It scrapes public job listings, applies clustering and classification to identify market patterns, uses an LLM to generate a human-readable briefing, and delivers results to a Google Sheet automatically.

**Why this project for the interview:** It demonstrates scraping, databases, ML pipelines, LLM integration, Google Sheets automation, and async programming — covering nearly every technical requirement in the target job description. The output is immediately relatable to the interviewer and can be demoed live.

---

## 2. Data Source

Scrape public AI/ML job listings from one or more of:
- **RemoteOK** (remoteok.com) — public, no login, Playwright-friendly
- **We Work Remotely** (weworkremotely.com) — public job board
- **HackerNews "Who's Hiring"** monthly threads — plain text, rich skill data

Each listing captures: title, company, description, location, salary (if present), posted date, source URL.

---

## 3. Architecture

```
[Playwright Scraper]
        ↓  (raw job listings)
[PostgreSQL Database]
        ↓  (cleaned text)
[ML Pipeline - scikit-learn]
        ↓  (cluster labels, seniority, skill frequencies)
[LLM Insight Generator - Claude/OpenAI API]
        ↓  (natural language report)
[Google Sheets Reporter - gspread]

All triggered via: POST /run-pipeline  (FastAPI, async)
```

---

## 4. Folder Structure

```
ai-ml-pipeline/
├── scraper/
│   ├── playwright_scraper.py   # async Playwright scraping logic
│   └── parser.py               # extract fields from raw HTML
├── db/
│   ├── models.py               # SQLAlchemy ORM models
│   ├── session.py              # DB connection + session factory
│   └── migrations/             # Alembic migrations
├── ml/
│   ├── features.py             # TF-IDF vectorization, skill extraction
│   ├── clustering.py           # KMeans clustering + PCA visualization
│   └── classifier.py           # Seniority classifier (LR, SVM, comparison)
├── llm/
│   └── insight_generator.py    # Structured prompt + API call
├── reporter/
│   └── sheets.py               # gspread: write tables + LLM report to Sheet
├── api/
│   └── main.py                 # FastAPI app, /run-pipeline endpoint
├── pipeline.py                 # Orchestrates all steps end-to-end
├── docker-compose.yml          # PostgreSQL + API
└── requirements.txt
```

---

## 5. Component Details

### 5.1 Scraper (Playwright + asyncio)

- Uses `async_playwright` for non-blocking scraping
- Handles pagination (scrapes multiple pages per source)
- Deduplicates listings by URL before inserting into DB
- Stores raw HTML + extracted fields separately so data is never lost

### 5.2 Database (PostgreSQL + SQLAlchemy + Alembic)

**Tables:**
- `jobs` — id, title, company, description, location, salary, source_url, posted_at, scraped_at
- `job_features` — job_id, cluster_id, seniority_label, skill_vector (JSON)
- `pipeline_runs` — id, run_at, jobs_scraped, llm_report (text)

Alembic manages schema migrations so the DB is version-controlled.

### 5.3 ML Pipeline (scikit-learn)

**Step 1 — Feature Extraction:**
- TF-IDF vectorization on job description text (max 5000 features, English stop words removed)
- Separate keyword extraction for top skills (Python, PyTorch, LangChain, etc.) using frequency count

**Step 2 — Clustering (Unsupervised):**
- KMeans with k=5–8 (choose k via elbow method or silhouette score)
- PCA reduction to 2D for visualization
- Each cluster is inspected and given a human label (e.g., "LLM/GenAI", "Classical ML", "MLOps")
- Output: cluster label per job, cluster size percentages

**Step 3 — Seniority Classification (Supervised):**
- Label ~150–200 jobs manually (or with LLM assistance) as junior/mid/senior
- Train and compare: Logistic Regression, SVM, Random Forest
- Evaluate with F1 score; select best model
- Output: predicted seniority per job, accuracy comparison table

**Step 4 — Skill Frequency Analysis:**
- Count occurrence of 50+ predefined AI/ML skills across all listings
- Output: ranked skill list with percentages

### 5.4 LLM Insight Generator (Claude or OpenAI API)

Constructs a structured prompt from pipeline outputs and calls the LLM API to generate a market intelligence briefing.

**Input to LLM (structured summary):**
```
Total jobs analyzed: {n}
Top skills: Python ({x}%), PyTorch ({y}%), LangChain ({z}%)...
Cluster breakdown: {cluster_name} ({pct}%), ...
Seniority split: Senior {x}%, Mid {y}%, Junior {z}%
Date range: {start} to {end}
```

**Output:** 2–3 paragraph natural language report with market observations and trends.

The prompt is templated (not hardcoded) so it can be adjusted without touching business logic.

### 5.5 Google Sheets Reporter (gspread)

- Authenticates via Google Service Account (JSON key)
- Per pipeline run: creates a new sheet tab named by date (e.g., "2026-04-23")
- Tab contents:
  - Row 1: LLM-generated report text
  - Section 2: Skill frequency table
  - Section 3: Cluster breakdown table
  - Section 4: Seniority distribution table

### 5.6 API (FastAPI + asyncio)

**Endpoint:** `POST /run-pipeline`
- Triggers scrape → ML → LLM → Sheets in sequence
- Returns run summary: jobs scraped, clusters found, Sheet URL
- Async throughout; scraping uses `asyncio.gather` for concurrent page fetching

---

## 6. Error Handling

- Scraper: retry failed pages up to 3 times with exponential backoff
- DB: use transactions; rollback on failure
- LLM API: catch rate limit errors, retry once after 10s
- Sheets: fail gracefully — if push fails, log error but don't abort the run (ML results are already saved to DB)

---

## 7. Testing Strategy

- Unit tests for parser, feature extraction, and skill frequency logic
- Integration test: run full pipeline against a fixture of 20 pre-scraped jobs (no live scraping)
- Manual demo test: full live run before interview

---

## 8. Deployment

- **Local:** `docker-compose up` starts PostgreSQL + FastAPI
- **No cloud required:** everything runs on developer machine for demo purposes
- `docker-compose.yml` provided so any reviewer can spin it up in one command

---

## 9. Interview Demo Script

1. Show GitHub repo: README, clean folder structure, requirements.txt
2. Run `POST /run-pipeline` via curl or Postman (live or pre-recorded)
3. Open Google Sheet: show dated tab, skill tables, LLM report
4. Show PCA cluster plot and seniority classifier comparison table
5. Walk through key files: async scraper, ML pipeline, LLM prompt template

**Skill-to-requirement mapping:**

| Job Requirement | Project Evidence |
|---|---|
| LLMs | LLM insight generator with structured prompting |
| Clustering & classification | KMeans + seniority classifier with model comparison |
| Web scraping (Playwright) | Async Playwright scraper with pagination |
| Google Sheets automation | gspread reporter with dated tabs |
| Data pipelines | End-to-end `pipeline.py` orchestrator |
| Databases (PostgreSQL) | SQLAlchemy models + Alembic migrations |
| Async programming | asyncio throughout scraper + FastAPI |
| Comparative ML research | Classifier comparison table (LR vs SVM vs RF) |

---

## 10. Build Timeline (1.5 weeks)

| Days | Work |
|---|---|
| 1–2 | Scraper + PostgreSQL setup + Alembic migrations |
| 3–4 | ML pipeline: TF-IDF, KMeans, seniority classifier, visualizations |
| 5 | LLM integration: prompt template + API call |
| 6 | Google Sheets reporter |
| 7–8 | FastAPI endpoint, docker-compose, README, demo prep |

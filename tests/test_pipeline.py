import json
import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base


@pytest.fixture
def sample_jobs():
    with open("tests/fixtures/sample_jobs.json") as f:
        return json.load(f)


@pytest.fixture
def sqlite_session(tmp_path):
    """In-memory SQLite session for tests -- no PostgreSQL needed."""
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

    assert result1["jobs_scraped"] == 20
    assert result2["jobs_scraped"] == 0
    assert result2["total_jobs"] == 20

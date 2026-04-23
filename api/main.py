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

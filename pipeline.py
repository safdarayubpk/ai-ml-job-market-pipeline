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
    Run the full pipeline: scrape -> normalize -> save -> ML -> LLM -> Sheets.
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

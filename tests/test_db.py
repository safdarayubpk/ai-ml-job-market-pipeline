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

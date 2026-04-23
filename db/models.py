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

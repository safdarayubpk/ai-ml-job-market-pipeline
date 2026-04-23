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

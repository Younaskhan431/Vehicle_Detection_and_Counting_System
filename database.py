from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use absolute path to ensure FastAPI and Streamlit share the exact same SQLite database
DB_PATH = Path(__file__).resolve().parent / "vehicle_analytics.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a session and always closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

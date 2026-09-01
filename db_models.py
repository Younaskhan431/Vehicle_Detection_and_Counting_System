from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from database import Base


class Run(Base):
    """One processed video = one Run. Tracks status through pending -> processing -> complete/failed."""
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, processing, complete, failed
    model_path = Column(String)
    confidence_threshold = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


    total_frames = Column(Integer, nullable=True)
    total_time_sec = Column(Float, nullable=True)
    avg_inference_ms = Column(Float, nullable=True)
    output_video_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)

    class_counts = relationship("ClassCount", back_populates="run", cascade="all, delete-orphan")


class ClassCount(Base):
    """Per-class in/out totals for a single run."""
    __tablename__ = "class_counts"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    class_name = Column(String, nullable=False)
    in_count = Column(Integer, default=0)
    out_count = Column(Integer, default=0)

    run = relationship("Run", back_populates="class_counts")

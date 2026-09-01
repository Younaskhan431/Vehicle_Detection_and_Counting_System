from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class ClassCountOut(BaseModel):
    class_name: str
    in_count: int
    out_count: int

    class Config:
        from_attributes = True


class RunOut(BaseModel):
    id: int
    filename: str
    status: str
    model_path: Optional[str] = None
    confidence_threshold: float
    created_at: datetime
    total_frames: Optional[int] = None
    total_time_sec: Optional[float] = None
    avg_inference_ms: Optional[float] = None
    output_video_path: Optional[str] = None
    error_message: Optional[str] = None
    class_counts: List[ClassCountOut] = []

    class Config:
        from_attributes = True

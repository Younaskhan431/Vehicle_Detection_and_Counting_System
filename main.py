import csv
import io
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import Base, engine, get_db, SessionLocal
from db_models import Run, ClassCount
from schemas import RunOut
from processing import process_video_file
from model import MODEL_PATH, get_available_models

Base.metadata.create_all(bind=engine)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Vehicle Detection & Traffic Counting API",
    description="Backend API for YOLO vehicle detection, ByteTrack tracking, and directional traffic flow counting.",
    version="2.3.0",
)

# Enable CORS for external frontends or local Streamlit dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_processing_job(
    run_id: int,
    input_path: Path,
    output_path: Path,
    model_name: str,
    confidence: float,
    inbound_ratio: float = 0.40,
    outbound_ratio: float = 0.68,
    frame_stride: int = 2,
):
    """
    Runs in the background after upload returns.
    Uses its own isolated DB session to update status.
    Ensures cleanup of input file even if processing fails.
    """
    db = SessionLocal()
    try:
        run = db.get(Run, run_id)
        if not run:
            return

        run.status = "processing"
        db.commit()

        try:
            counts, stats = process_video_file(
                input_path=input_path,
                output_path=output_path,
                model_name_or_path=model_name,
                confidence=confidence,
                inbound_ratio=inbound_ratio,
                outbound_ratio=outbound_ratio,
                frame_stride=frame_stride,
            )

            run = db.get(Run, run_id)
            run.status = "complete"
            run.total_frames = stats["total_frames"]
            run.total_time_sec = stats["total_time_sec"]
            run.avg_inference_ms = stats["avg_inference_ms"]
            run.output_video_path = str(output_path)
            db.commit()

            # Remove any stale class counts and record new tallies
            db.query(ClassCount).filter(ClassCount.run_id == run.id).delete()
            for class_name, c in counts.items():
                db.add(
                    ClassCount(
                        run_id=run.id,
                        class_name=class_name,
                        in_count=c["in"],
                        out_count=c["out"],
                    )
                )
            db.commit()
        except Exception as processing_err:
            run = db.get(Run, run_id)
            if run:
                run.status = "failed"
                run.error_message = str(processing_err)
                db.commit()
            raise
    except Exception as e:
        print(f"Background processing job {run_id} failed: {e}")
    finally:
        # Always clean up input file
        if isinstance(input_path, Path) and input_path.exists():
            try:
                os.remove(input_path)
            except OSError as cleanup_err:
                print(f"Warning: Failed to clean up input file {input_path}: {cleanup_err}")
        db.close()


@app.get("/models")
def list_available_models():
    """Lists all available YOLO model weight files in the project."""
    return {"models": get_available_models()}


@app.post("/runs/upload", response_model=RunOut)
def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: str = Form(MODEL_PATH),
    confidence: float = Form(0.4),
    inbound_ratio: float = Form(0.40),
    outbound_ratio: float = Form(0.68),
    frame_stride: int = Form(2),
    db: Session = Depends(get_db),
):
    """
    Accepts a video, saves it to disk, creates a Run record, and starts
    background detection/tracking with extended corridor counting. Returns immediately with status='pending'.
    """
    unique_prefix = uuid.uuid4().hex[:8]
    sanitized_filename = f"{unique_prefix}_{Path(file.filename).name}"
    input_path = UPLOAD_DIR / sanitized_filename

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    run = Run(
        filename=file.filename,
        status="pending",
        model_path=model_name,
        confidence_threshold=confidence,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    output_path = OUTPUT_DIR / f"run_{run.id}_output.mp4"
    background_tasks.add_task(
        run_processing_job,
        run.id,
        input_path,
        output_path,
        model_name,
        confidence,
        inbound_ratio,
        outbound_ratio,
        frame_stride,
    )

    return run


@app.get("/runs", response_model=list[RunOut])
def list_runs(db: Session = Depends(get_db)):
    """History of all runs, most recent first."""
    return db.query(Run).order_by(Run.created_at.desc()).all()


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    """Poll this to check a run's status and fetch results when completed."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.delete("/runs/{run_id}", status_code=status.HTTP_200_OK)
def delete_run(run_id: int, db: Session = Depends(get_db)):
    """Deletes a run record and its associated video output from disk."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Clean up output file from disk if present
    if run.output_video_path and os.path.exists(run.output_video_path):
        try:
            os.remove(run.output_video_path)
        except OSError:
            pass

    db.delete(run)
    db.commit()
    return {"message": f"Run #{run_id} successfully deleted"}


@app.delete("/runs", status_code=status.HTTP_200_OK)
def delete_all_runs(db: Session = Depends(get_db)):
    """Clears all runs and class counts from the database and deletes output video files."""
    runs = db.query(Run).all()
    for r in runs:
        if r.output_video_path and os.path.exists(r.output_video_path):
            try:
                os.remove(r.output_video_path)
            except OSError:
                pass
    db.query(ClassCount).delete()
    db.query(Run).delete()
    db.commit()
    return {"message": "All past runs and output records successfully deleted."}


@app.get("/runs/{run_id}/video")
def get_run_video(run_id: int, db: Session = Depends(get_db)):
    """Streams the processed video file for in-browser playback."""
    run = db.get(Run, run_id)
    if not run or not run.output_video_path or not os.path.exists(run.output_video_path):
        raise HTTPException(status_code=404, detail="Video not found or processing not complete")
    return FileResponse(run.output_video_path, media_type="video/mp4")


@app.get("/runs/{run_id}/report.csv")
def get_run_report(run_id: int, db: Session = Depends(get_db)):
    """Generates and downloads a CSV report for a completed run."""
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["run_id", run.id])
    writer.writerow(["filename", run.filename])
    writer.writerow(["model", run.model_path])
    writer.writerow(["total_frames", run.total_frames])
    writer.writerow(["total_time_sec", run.total_time_sec])
    writer.writerow(["avg_inference_ms", run.avg_inference_ms])
    writer.writerow([])
    writer.writerow(["class", "in", "out", "total"])
    for cc in run.class_counts:
        writer.writerow([cc.class_name, cc.in_count, cc.out_count, cc.in_count + cc.out_count])

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=run_{run_id}_report.csv"},
    )


@app.get("/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    """Aggregate vehicle in/out totals per class across every completed run."""
    rows = (
        db.query(ClassCount.class_name, func.sum(ClassCount.in_count), func.sum(ClassCount.out_count))
        .join(Run)
        .filter(Run.status == "complete")
        .filter(func.lower(ClassCount.class_name) != "bicycle")
        .group_by(ClassCount.class_name)
        .all()
    )
    return [
        {"class_name": name, "total_in": int(in_sum or 0), "total_out": int(out_sum or 0)}
        for name, in_sum, out_sum in rows
    ]

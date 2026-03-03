"""
Retraining Router

Endpoint to trigger model retraining pipeline as a background task.
"""

import logging
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from app.config import settings
from app.schemas import RetrainRequest, RetrainResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Simple in-memory job tracking
_retrain_jobs: dict[str, dict] = {}


def _run_retrain(job_id: str, from_scratch: bool) -> None:
    """Background task: Run the retraining pipeline."""
    _retrain_jobs[job_id]["status"] = "running"
    logger.info(f"Retraining job {job_id} started")

    try:
        ml_dir = Path(settings.ML_DIR).resolve()
        retrain_script = ml_dir / "retrain.py"

        if not retrain_script.exists():
            _retrain_jobs[job_id]["status"] = "failed"
            _retrain_jobs[job_id]["result"] = {"error": "retrain.py not found"}
            return

        cmd = [sys.executable, str(retrain_script)]
        if from_scratch:
            cmd.append("--from-scratch")

        result = subprocess.run(
            cmd,
            cwd=str(ml_dir),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode == 0:
            _retrain_jobs[job_id]["status"] = "completed"
            _retrain_jobs[job_id]["result"] = {
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "message": "Retraining completed successfully",
            }
            logger.info(f"Retraining job {job_id} completed")
        else:
            _retrain_jobs[job_id]["status"] = "failed"
            _retrain_jobs[job_id]["result"] = {
                "error": result.stderr[-1000:] if result.stderr else "Unknown error",
            }
            logger.error(f"Retraining job {job_id} failed: {result.stderr[:500]}")

    except subprocess.TimeoutExpired:
        _retrain_jobs[job_id]["status"] = "failed"
        _retrain_jobs[job_id]["result"] = {"error": "Retraining timed out (1 hour limit)"}
    except Exception as e:
        _retrain_jobs[job_id]["status"] = "failed"
        _retrain_jobs[job_id]["result"] = {"error": str(e)}
        logger.exception(f"Retraining job {job_id} error")


@router.post("", response_model=RetrainResponse)
async def trigger_retrain(
    request: RetrainRequest = RetrainRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Trigger model retraining pipeline.

    Runs as a background task. Returns a job ID to check status.
    The pipeline will:
    1. Merge new labeled data from data/new_labels/
    2. Retrain the EfficientNet classifier
    3. Evaluate on test set
    4. Re-export to ONNX
    5. (Optional) TensorRT quantization
    6. Version the new model
    """
    # Check if a job is already running
    running = [j for j in _retrain_jobs.values() if j["status"] == "running"]
    if running:
        return RetrainResponse(
            job_id=running[0]["job_id"],
            status="running",
            message="A retraining job is already in progress",
        )

    job_id = str(uuid.uuid4())[:8]
    _retrain_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "result": None,
    }

    background_tasks.add_task(_run_retrain, job_id, request.from_scratch)

    return RetrainResponse(
        job_id=job_id,
        status="queued",
        message=f"Retraining job queued. Check status at GET /retrain/{job_id}",
    )


@router.get("/{job_id}")
async def get_retrain_status(job_id: str):
    """Get the status of a retraining job."""
    job = _retrain_jobs.get(job_id)
    if not job:
        return {"error": "Job not found", "job_id": job_id}

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "result": job.get("result"),
    }

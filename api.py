"""
FastAPI application exposing the mystery coloring generator as an HTTP API.
"""

import io
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from config import Config
from main import MysteryColoringGenerator


JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
JOB_WORKERS = int(os.getenv("JOB_WORKERS", "2"))
JOB_STORE: Dict[str, dict] = {}
JOB_STORE_LOCK = threading.Lock()
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=JOB_WORKERS)

app = FastAPI(
    title="Mystery Coloring API",
    description="Upload an image, receive a job id immediately, then poll until the ZIP is ready.",
    version="2.0.0",
)


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def format_datetime(value: datetime) -> str:
    """Format datetimes consistently for API responses."""
    return value.isoformat().replace("+00:00", "Z")


def sanitize_output_name(filename: Optional[str]) -> str:
    """Create a safe base filename for generated assets."""
    raw_name = os.path.splitext(os.path.basename(filename or "image"))[0]
    cleaned_name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name).strip("_")
    return cleaned_name or "image"


def parse_forced_colors(value: Optional[str]) -> List[Tuple[int, int, int]]:
    """Parse forced RGB colors from the CLI-style semicolon format."""
    if not value:
        return []

    forced_colors: List[Tuple[int, int, int]] = []
    for color_str in value.split(";"):
        rgb = tuple(map(int, color_str.strip().split(",")))
        if len(rgb) != 3 or any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(f"Invalid RGB values: {color_str}")
        forced_colors.append((rgb[0], rgb[1], rgb[2]))

    return forced_colors


def build_config(
    colors: int,
    difficulty: int,
    symbols: Literal["numbers", "letters", "custom"],
    min_area: int,
    resolution: int,
    symbol_size: float,
    prefill_dark: int,
    mode_filter: int,
    no_bilateral: bool,
    force_colors: Optional[str],
) -> Config:
    """Build a generator config from submitted form values."""
    if symbols not in {"numbers", "letters", "custom"}:
        raise ValueError("symbols must be one of: numbers, letters, custom")

    config = Config()
    config.output_dir = "output"
    config.legend_position = "bottom"
    config.num_colors = colors
    config.difficulty_level = difficulty
    config.symbol_type = symbols
    config.min_region_area = min_area
    config.max_image_dimension = resolution
    config.font_size_ratio = symbol_size
    config.prefill_dark_threshold = prefill_dark
    config.prefill_dark_regions = prefill_dark > 0
    config.mode_filter_size = mode_filter
    config.bilateral_filter_enabled = not no_bilateral
    config.forced_colors = parse_forced_colors(force_colors)
    config.validate()
    return config


def encode_png(image, label: str) -> bytes:
    """Encode a BGR image as PNG bytes."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Failed to encode {label} image")
    return buffer.tobytes()


def cleanup_expired_jobs() -> None:
    """Remove terminal jobs that are older than the configured TTL."""
    now_ts = time.time()

    with JOB_STORE_LOCK:
        expired_job_ids = [
            job_id
            for job_id, job in JOB_STORE.items()
            if job["status"] in {"completed", "failed"} and job["expires_at_ts"] <= now_ts
        ]

        for job_id in expired_job_ids:
            del JOB_STORE[job_id]


def create_job_record(base_name: str, source_filename: str) -> dict:
    """Create the initial in-memory job record."""
    now = utc_now()
    expires_at = datetime.fromtimestamp(time.time() + JOB_TTL_SECONDS, tz=timezone.utc)

    return {
        "job_id": uuid4().hex,
        "status": "queued",
        "output_name": base_name,
        "source_filename": source_filename,
        "created_at": format_datetime(now),
        "updated_at": format_datetime(now),
        "completed_at": None,
        "error": None,
        "files": [f"{base_name}_combined.png", f"{base_name}_preview.png"],
        "content_type": "application/zip",
        "archive_name": f"{base_name}_outputs.zip",
        "expires_at": format_datetime(expires_at),
        "expires_at_ts": expires_at.timestamp(),
        "result_bytes": None,
    }


def update_job(job_id: str, **changes) -> None:
    """Update a job record safely."""
    with JOB_STORE_LOCK:
        job = JOB_STORE.get(job_id)
        if job is None:
            return

        job.update(changes)
        job["updated_at"] = format_datetime(utc_now())


def serialize_job(job: dict, request: Request) -> dict:
    """Convert an internal job record into an API response payload."""
    job_id = job["job_id"]
    payload = {
        "job_id": job_id,
        "status": job["status"],
        "output_name": job["output_name"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "completed_at": job["completed_at"],
        "expires_at": job["expires_at"],
        "files": job["files"],
        "status_url": str(request.url_for("get_job_status", job_id=job_id)),
        "download_url": str(request.url_for("download_job_result", job_id=job_id)),
    }

    if job["status"] == "failed":
        payload["error"] = job["error"]

    return payload


def generate_job_archive(job_id: str, image_bytes: bytes, source_filename: str, config: Config) -> None:
    """Run the image generation pipeline in the background for one job."""
    update_job(job_id, status="processing", error=None)

    with JOB_STORE_LOCK:
        job = JOB_STORE.get(job_id)
        if job is None:
            return
        base_name = job["output_name"]

    try:
        generator = MysteryColoringGenerator(config)
        loaded_image = generator.load_image_bytes(image_bytes, source_filename)
        outputs = generator.generate_images(loaded_image)

        combined_bytes = encode_png(outputs["combined"], "combined")
        preview_bytes = encode_png(outputs["preview"], "preview")

        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{base_name}_combined.png", combined_bytes)
            archive.writestr(f"{base_name}_preview.png", preview_bytes)

        completed_at = format_datetime(utc_now())
        update_job(
            job_id,
            status="completed",
            completed_at=completed_at,
            result_bytes=archive_buffer.getvalue(),
        )
    except Exception as exc:
        update_job(job_id, status="failed", completed_at=format_datetime(utc_now()), error=str(exc))


def get_job_or_404(job_id: str) -> dict:
    """Fetch a job or raise 404."""
    cleanup_expired_jobs()

    with JOB_STORE_LOCK:
        job = JOB_STORE.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.get("/")
def root() -> dict:
    """Basic info endpoint."""
    return {
        "name": "Mystery Coloring API",
        "docs": "/docs",
        "health": "/health",
        "generate": "/generate",
        "job_status": "/jobs/{job_id}",
        "job_download": "/jobs/{job_id}/download",
    }


@app.head("/")
def root_head() -> Response:
    """HEAD response for platform probes."""
    return Response(status_code=200)


@app.get("/health")
def health() -> dict:
    """Health endpoint for Render."""
    return {"status": "ok"}


@app.head("/health")
def health_head() -> Response:
    """HEAD health response for platform probes."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(
    request: Request,
    image: UploadFile = File(...),
    output_name: Optional[str] = Form(None),
    colors: int = Form(16),
    difficulty: int = Form(5),
    symbols: Literal["numbers", "letters", "custom"] = Form("numbers"),
    min_area: int = Form(50),
    resolution: int = Form(1400),
    symbol_size: float = Form(0.5),
    prefill_dark: int = Form(500),
    mode_filter: int = Form(5),
    no_bilateral: bool = Form(False),
    force_colors: Optional[str] = Form(None),
):
    """
    Create a generation job and return immediately with a job id.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="A filename is required for the uploaded image")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")

    try:
        config = build_config(
            colors=colors,
            difficulty=difficulty,
            symbols=symbols,
            min_area=min_area,
            resolution=resolution,
            symbol_size=symbol_size,
            prefill_dark=prefill_dark,
            mode_filter=mode_filter,
            no_bilateral=no_bilateral,
            force_colors=force_colors,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cleanup_expired_jobs()

    base_name = sanitize_output_name(output_name or image.filename)
    job = create_job_record(base_name, image.filename)
    job_id = job["job_id"]

    with JOB_STORE_LOCK:
        JOB_STORE[job_id] = job

    JOB_EXECUTOR.submit(generate_job_archive, job_id, image_bytes, image.filename, config)

    payload = serialize_job(job, request)
    payload["message"] = "Job created. Poll status_url until status is completed."

    headers = {
        "Location": str(request.url_for("get_job_status", job_id=job_id)),
        "Retry-After": "5",
    }
    return JSONResponse(status_code=202, content=payload, headers=headers)


@app.get("/jobs/{job_id}", name="get_job_status")
def get_job_status(job_id: str, request: Request) -> dict:
    """Return the status of a generation job."""
    job = get_job_or_404(job_id)
    return serialize_job(job, request)


@app.get("/jobs/{job_id}/download", name="download_job_result")
def download_job_result(job_id: str) -> StreamingResponse:
    """Download the generated ZIP once the job is complete."""
    job = get_job_or_404(job_id)

    if job["status"] in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail=f"Job is not ready yet (status: {job['status']})")

    if job["status"] == "failed":
        raise HTTPException(status_code=409, detail=job["error"] or "Job failed")

    result_bytes = job.get("result_bytes")
    if result_bytes is None:
        raise HTTPException(status_code=500, detail="Job completed but result archive is missing")

    headers = {
        "Content-Disposition": f'attachment; filename="{job["archive_name"]}"'
    }
    return StreamingResponse(io.BytesIO(result_bytes), media_type=job["content_type"], headers=headers)

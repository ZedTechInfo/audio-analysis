from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    upload_dir: Path = _path_env("AUTOACE_UPLOAD_DIR", PROJECT_ROOT / "temp_uploads")
    job_db: Path = _path_env(
        "AUTOACE_JOB_DB",
        (_path_env("AUTOACE_UPLOAD_DIR", PROJECT_ROOT / "temp_uploads") / "jobs.sqlite3"),
    )
    environment: str = os.getenv("AUTOACE_ENVIRONMENT", "development")
    inference_engine: str = os.getenv("AUTOACE_INFERENCE_ENGINE", "hybrid")
    ser_model_id: str = os.getenv("AUTOACE_SER_MODEL_ID", "superb/wav2vec2-base-superb-er")
    ser_window_seconds: int = _int_env("AUTOACE_SER_WINDOW_SECONDS", 8)
    ser_hop_seconds: int = _int_env("AUTOACE_SER_HOP_SECONDS", 4)
    ser_weight: float = float(os.getenv("AUTOACE_SER_WEIGHT", "0.5"))
    max_upload_bytes: int = _int_env("AUTOACE_MAX_UPLOAD_BYTES", 500 * 1024 * 1024)
    max_uncompressed_bytes: int = _int_env("AUTOACE_MAX_UNCOMPRESSED_BYTES", 500 * 1024 * 1024)
    max_files: int = _int_env("AUTOACE_MAX_FILES", 5000)
    max_audio_seconds: int = _int_env("AUTOACE_MAX_AUDIO_SECONDS", 3600)
    worker_count: int = _int_env("AUTOACE_WORKER_COUNT", 1)
    vad_top_db: int = _int_env("AUTOACE_VAD_TOP_DB", 30)
    noise_floor_threshold: float = float(os.getenv("AUTOACE_NOISE_FLOOR_THRESHOLD", "0.001"))
    noise_ratio_threshold: float = float(os.getenv("AUTOACE_NOISE_RATIO_THRESHOLD", "0.03"))
    overlap_frame_ratio: float = float(os.getenv("AUTOACE_OVERLAP_FRAME_RATIO", "0.12"))
    api_key: str = os.getenv("AUTOACE_API_KEY", "")
    require_api_key: bool = os.getenv(
        "AUTOACE_REQUIRE_API_KEY", "true" if os.getenv("AUTOACE_ENVIRONMENT") == "production" else "false"
    ).lower() == "true"
    allow_signup: bool = os.getenv("AUTOACE_ALLOW_SIGNUP", "true").lower() == "true"
    static_version: str = os.getenv("AUTOACE_STATIC_VERSION", "2026-08-09-1")


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)

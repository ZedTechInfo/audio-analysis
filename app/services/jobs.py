from __future__ import annotations

import csv
import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.schemas import AnalysisResult
from app.services.audio_engine import analyze_audio_file
from app.services.batch import validate_batch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, database: Path):
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    result_path TEXT,
                    error_path TEXT
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS application_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    job_id TEXT,
                    artifact TEXT
                )"""
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
            for name, definition in {
                "job_type": "TEXT NOT NULL DEFAULT 'unknown'",
                "source_name": "TEXT",
            }.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")

    def _connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def create(self, job_id: str, *, job_type: str = "unknown", source_name: str | None = None) -> None:
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, status, created_at, updated_at, error, result_path, error_path, job_type, source_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, "queued", now, now, None, None, None, job_type, source_name),
            )

    def update(self, job_id: str, status: str, **values: str | None) -> None:
        fields = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, _now()]
        for key in ("error", "result_path", "error_path", "job_type", "source_name"):
            if key in values:
                fields.append(f"{key} = ?")
                params.append(values[key])
        params.append(job_id)
        with self._lock, self._connect() as connection:
            connection.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", params)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def add_log(self, level: str, source: str, message: str, *, job_id: str | None = None, artifact: str | None = None) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO application_logs (created_at, level, source, message, job_id, artifact) VALUES (?, ?, ?, ?, ?, ?)",
                (_now(), level, source, message, job_id, artifact),
            )

    def list_logs(self, page: int, page_size: int, level: str | None = None) -> tuple[list[dict[str, Any]], int]:
        where = " WHERE level = ?" if level else ""
        values = (level,) if level else ()
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            total = connection.execute(f"SELECT COUNT(*) FROM application_logs{where}", values).fetchone()[0]
            offset = (page - 1) * page_size
            rows = connection.execute(
                f"SELECT id, created_at, level, source, message, job_id, artifact FROM application_logs{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                (*values, page_size, offset),
            ).fetchall()
            return [dict(row) for row in rows], total


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = JobStore(settings.job_db)
        self.executor = ThreadPoolExecutor(max_workers=settings.worker_count)

    def submit(
        self,
        batch_dir: Path,
        job_id: str | None = None,
        *,
        job_type: str = "unknown",
        source_name: str | None = None,
    ) -> str:
        job_id = job_id or uuid.uuid4().hex
        self.store.create(job_id, job_type=job_type, source_name=source_name)
        self.executor.submit(self._run, job_id, batch_dir)
        return job_id

    def _run(self, job_id: str, batch_dir: Path) -> None:
        result_path = batch_dir / "results.json"
        error_path = batch_dir / "errors.json"
        try:
            self.store.update(job_id, "validating")
            rows = validate_batch(batch_dir, self.settings)
            self.store.update(job_id, "processing")
            results: list[dict[str, Any]] = []
            errors: list[dict[str, str]] = []
            for row in rows:
                try:
                    raw = analyze_audio_file(
                        str(batch_dir / row["name"]),
                        use_ser=self.settings.inference_engine == "hybrid",
                    )
                    if raw is None:
                        raise RuntimeError("Audio engine returned no result")
                    result = AnalysisResult(**raw)
                    serialized = result.model_dump(mode="json")
                    serialized["filename"] = row["name"]
                    results.append(serialized)
                except Exception as exc:
                    errors.append({"filename": row["name"], "error": str(exc)})

            result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
            with (batch_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                fieldnames = list(results[0]) if results else [
                    "filename",
                    "emotional_tone",
                    "emotional_intensity",
                    "background_noise_present",
                    "background_noise_type",
                    "background_noise_severity",
                    "audio_quality",
                    "speaker_overlap_present",
                    "long_silence_present",
                    "confidence",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            status = "completed_with_errors" if errors else "completed"
            self.store.update(
                job_id,
                status,
                result_path=str(result_path),
                error_path=str(error_path),
            )
        except Exception as exc:
            self.store.update(job_id, "failed", error=str(exc))

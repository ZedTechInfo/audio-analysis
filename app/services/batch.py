from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf

from app.config import Settings
from app.services.errors import BatchValidationError


SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def save_upload(upload_file, destination: Path, settings: Settings) -> int:
    """Stream an upload to disk while enforcing the compressed-size limit."""
    written = 0
    with destination.open("wb") as output:
        while chunk := upload_file.file.read(1024 * 1024):
            written += len(chunk)
            if written > settings.max_upload_bytes:
                raise BatchValidationError("Upload exceeds the maximum allowed size")
            output.write(chunk)
    if written == 0:
        raise BatchValidationError("Upload is empty")
    return written


def extract_batch(archive_path: Path, destination: Path, settings: Settings) -> None:
    """Safely extract a ZIP archive and reject traversal, symlinks, and zip bombs."""
    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise BatchValidationError("Invalid ZIP archive") from exc

    with archive:
        members = archive.infolist()
        if len(members) > settings.max_files:
            raise BatchValidationError("Batch contains too many archive entries")

        total_size = sum(member.file_size for member in members)
        if total_size > settings.max_uncompressed_bytes:
            raise BatchValidationError("Uncompressed batch exceeds the maximum allowed size")

        root = destination.resolve()
        for member in members:
            name = member.filename.replace("\\", "/")
            target = (destination / name).resolve()
            if name.startswith("/") or target != root and root not in target.parents:
                raise BatchValidationError("Archive contains an unsafe path")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise BatchValidationError("Archive contains a symbolic link")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _parse_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "name" not in (reader.fieldnames or []):
            raise BatchValidationError("Manifest must contain a name column")
        rows = []
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name or Path(name).name != name:
                raise BatchValidationError("Manifest filenames must be root-level filenames")
            expected = None
            raw_result = (row.get("result_json") or "").strip()
            if raw_result:
                try:
                    expected = json.loads(raw_result)
                except json.JSONDecodeError as exc:
                    raise BatchValidationError(f"Invalid result_json for {name}") from exc
            rows.append({"name": name, "expected": expected})
        return rows


def validate_batch(batch_dir: Path, settings: Settings) -> list[dict[str, Any]]:
    """Validate manifest/audio matching and return processing records."""
    manifest_path = batch_dir / "labels.csv"
    rows = _parse_manifest(manifest_path) if manifest_path.is_file() else []
    if not rows:
        rows = [
            {"name": path.name, "expected": None}
            for path in batch_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO
        ]
    if not rows:
        raise BatchValidationError("Batch contains no supported audio files")
    if len(rows) > settings.max_files:
        raise BatchValidationError("Batch contains too many audio files")

    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise BatchValidationError("Manifest contains duplicate filenames")
    missing = [name for name in names if not (batch_dir / name).is_file()]
    if missing:
        raise BatchValidationError(f"Manifest files are missing: {missing[:5]}")
    unsupported = [name for name in names if Path(name).suffix.lower() not in SUPPORTED_AUDIO]
    if unsupported:
        raise BatchValidationError(f"Unsupported audio files: {unsupported[:5]}")
    oversized = []
    for name in names:
        if not (batch_dir / name).stat().st_size:
            raise BatchValidationError(f"Audio file is empty: {name}")
        try:
            duration = float(sf.info(batch_dir / name).duration)
        except Exception:
            try:
                duration = float(librosa.get_duration(path=batch_dir / name))
            except Exception as exc:
                raise BatchValidationError(f"Invalid audio file: {name}") from exc
        if duration > settings.max_audio_seconds:
            oversized.append(name)
    if oversized:
        raise BatchValidationError(f"Audio files exceed the maximum duration: {oversized[:5]}")
    return rows

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from app.config import Settings
from app.services.batch import extract_batch
from app.services.errors import BatchValidationError


class BatchSecurityTests(unittest.TestCase):
    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "unsafe.zip"
            destination = root / "extract"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", "must not extract")

            with self.assertRaises(BatchValidationError):
                extract_batch(archive_path, destination, Settings(upload_dir=root, job_db=root / "jobs.sqlite3"))

            self.assertFalse((root / "outside.txt").exists())

    def test_rejects_uncompressed_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "large.zip"
            destination = root / "extract"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("large.txt", "x" * 20)

            limited = Settings(
                upload_dir=root,
                job_db=root / "jobs.sqlite3",
                max_uncompressed_bytes=10,
            )
            with self.assertRaises(BatchValidationError):
                extract_batch(archive_path, destination, limited)

    def test_counts_files_not_directory_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "folder.zip"
            destination = root / "extract"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("audio/", "")
                for index in range(3):
                    archive.writestr(f"audio/call_{index}.wav", b"\x00" * 10)

            limited = Settings(upload_dir=root, job_db=root / "jobs.sqlite3", max_files=3)
            extract_batch(archive_path, destination, limited)
            self.assertEqual(len(list(destination.rglob("*.wav"))), 3)

    def test_rejects_too_many_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "many.zip"
            destination = root / "extract"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in range(4):
                    archive.writestr(f"call_{index}.wav", b"\x00" * 10)

            limited = Settings(upload_dir=root, job_db=root / "jobs.sqlite3", max_files=3)
            with self.assertRaises(BatchValidationError):
                extract_batch(archive_path, destination, limited)

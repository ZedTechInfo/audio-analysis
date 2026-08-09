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

import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app import main


class MissingArtifactTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.job_id = uuid.uuid4().hex
        main.job_manager.store.create(self.job_id, job_type="single_audio", source_name="missing.wav")
        missing = Path(main.settings.upload_dir) / self.job_id / "results.json"
        main.job_manager.store.update(
            self.job_id,
            "completed",
            result_path=str(missing),
            error_path=str(missing.with_name("errors.json")),
        )

    def test_missing_results_artifact_is_a_controlled_404(self):
        response = self.client.get(f"/api/jobs/{self.job_id}/results")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Results are no longer available for this evaluation")

    def test_missing_download_artifact_is_a_controlled_404(self):
        response = self.client.get(f"/api/jobs/{self.job_id}/download?format=json")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Result file not found")


if __name__ == "__main__":
    unittest.main()

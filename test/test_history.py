import io
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return output.getvalue()


RAW_RESULT = {
    "emotional_tone": "neutral", "emotional_intensity": "low",
    "background_noise_present": False, "background_noise_type": "",
    "background_noise_severity": "none", "audio_quality": "clear",
    "speaker_overlap_present": False, "long_silence_present": False,
    "confidence": 0.8,
}


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.created = []

    def tearDown(self):
        import shutil
        for job_id in self.created:
            row = main.job_manager.store.get(job_id)
            if row and row.get("result_path"):
                shutil.rmtree(Path(row["result_path"]).parent, ignore_errors=True)

    def wait_for_completion(self, job_id):
        for _ in range(40):
            response = self.client.get(f"/api/jobs/{job_id}")
            if response.json()["status"] in {"completed", "completed_with_errors", "failed"}:
                return
            time.sleep(0.025)
        self.fail("job did not finish")

    @patch("app.services.jobs.analyze_audio_file", return_value=RAW_RESULT)
    def test_history_contains_completed_audio_summary(self, _analyze):
        response = self.client.post(
            "/api/jobs/audio",
            files={"audio_file": ("reference-call.wav", wav_bytes(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["job_id"]
        self.created.append(job_id)
        self.wait_for_completion(job_id)

        history = self.client.get("/api/jobs/history")
        self.assertEqual(history.status_code, 200)
        item = next(row for row in history.json()["items"] if row["id"] == job_id)
        self.assertEqual(item["job_type"], "single_audio")
        self.assertEqual(item["source_name"], "reference-call.wav")
        self.assertEqual(item["total_files"], 1)
        self.assertEqual(item["successful_files"], 1)
        self.assertEqual(item["failed_files"], 0)
        self.assertEqual(item["average_confidence"], 0.8)
        self.assertTrue(item["result_available"])


if __name__ == "__main__":
    unittest.main()

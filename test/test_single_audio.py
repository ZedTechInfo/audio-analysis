import io
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.config import settings


def wav_bytes(seconds=0.1):
    frames = b"\x00\x00" * int(16000 * seconds)
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(frames)
    return output.getvalue()


RAW_RESULT = {
    "emotional_tone": "neutral", "emotional_intensity": "low",
    "background_noise_present": False, "background_noise_type": "",
    "background_noise_severity": "none", "audio_quality": "clear",
    "speaker_overlap_present": False, "long_silence_present": False,
    "confidence": 0.7,
}


class SingleAudioUploadTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.created = []

    def tearDown(self):
        for job_id in self.created:
            row = main.job_manager.store.get(job_id)
            if row:
                import shutil
                shutil.rmtree(Path(row["result_path"]).parent, ignore_errors=True)

    def wait_for_completion(self, job_id):
        for _ in range(40):
            status = self.client.get(f"/api/jobs/{job_id}").json()
            if status["status"] in {"completed", "completed_with_errors", "failed"}:
                return status
            time.sleep(0.025)
        self.fail("single-file job did not finish")

    @patch("app.services.jobs.analyze_audio_file", return_value=RAW_RESULT)
    def test_single_file_success_schema_and_downloads(self, _analyze):
        response = self.client.post(
            "/api/jobs/audio",
            files={"audio_file": ("original.wav", wav_bytes(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["job_id"]
        self.created.append(job_id)
        self.assertEqual(self.wait_for_completion(job_id)["status"], "completed")

        result = self.client.get(f"/api/jobs/{job_id}/results")
        self.assertEqual(result.status_code, 200)
        rows = result.json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["filename"], "original.wav")
        self.assertEqual(set(rows[0]), {
            "filename", "emotional_tone", "emotional_intensity",
            "background_noise_present", "background_noise_type",
            "background_noise_severity", "audio_quality",
            "speaker_overlap_present", "long_silence_present", "confidence",
        })
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/download?format=csv").status_code, 200)
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/download?format=json").status_code, 200)

    def test_rejects_unsupported_empty_and_invalid_audio(self):
        for filename, content in (("voice.txt", b"not audio"), ("empty.wav", b""), ("bad.wav", b"not wav")):
            response = self.client.post("/api/jobs/audio", files={"audio_file": (filename, content)})
            self.assertEqual(response.status_code, 400, filename)

    def test_single_upload_requires_auth_when_enabled(self):
        original = main.settings
        try:
            from dataclasses import replace
            main.settings = replace(settings, require_api_key=True, api_key="test-key")
            response = self.client.post(
                "/api/jobs/audio",
                files={"audio_file": ("voice.wav", wav_bytes(), "audio/wav")},
            )
            self.assertEqual(response.status_code, 401)
        finally:
            main.settings = original

    def test_single_upload_applies_duration_and_size_limits(self):
        from dataclasses import replace
        original = main.settings
        try:
            main.settings = replace(settings, max_audio_seconds=0, max_upload_bytes=8)
            oversized = self.client.post(
                "/api/jobs/audio",
                files={"audio_file": ("voice.wav", wav_bytes(), "audio/wav")},
            )
            self.assertEqual(oversized.status_code, 400)
            self.assertIn("maximum allowed size", oversized.json()["detail"])
            main.settings = replace(settings, max_audio_seconds=0, max_upload_bytes=100000)
            too_long = self.client.post(
                "/api/jobs/audio",
                files={"audio_file": ("voice.wav", wav_bytes(), "audio/wav")},
            )
            self.assertEqual(too_long.status_code, 400)
            self.assertIn("maximum duration", too_long.json()["detail"])
        finally:
            main.settings = original


if __name__ == "__main__":
    unittest.main()

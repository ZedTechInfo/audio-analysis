import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from scripts.run_eval import calculate_metrics, load_manifest, validate_prediction


class RunEvalTests(unittest.TestCase):
    def test_load_manifest_parses_embedded_json(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_dir = Path(directory)
            (batch_dir / "labels.csv").write_text(
                'name,result_json\ncall.ogg,"{ ""emotional_tone"": ""neutral"" }"\n',
                encoding="utf-8",
            )

            rows = load_manifest(batch_dir)

        self.assertEqual(rows[0]["name"], "call.ogg")
        self.assertEqual(rows[0]["expected"]["emotional_tone"], "neutral")

    def test_calculate_metrics_reports_accuracy_and_macro_f1(self):
        expected = {
            "emotional_tone": "neutral",
            "emotional_intensity": "low",
            "background_noise_present": False,
            "background_noise_type": "",
            "background_noise_severity": "none",
            "audio_quality": "clear",
            "speaker_overlap_present": False,
            "long_silence_present": False,
        }
        prediction = {**expected, "confidence": 0.8, "filename": "call.ogg"}
        metrics = calculate_metrics([{"expected": expected, "prediction": prediction}])

        self.assertEqual(metrics["labeled_files"], 1)
        self.assertEqual(metrics["exact_schema_match_rate"], 1.0)
        self.assertEqual(metrics["fields"]["emotional_tone"]["macro_f1"], 1.0)

    def test_prediction_serializer_keeps_filename_as_metadata(self):
        raw = {
            "emotional_tone": "neutral",
            "emotional_intensity": "low",
            "background_noise_present": False,
            "background_noise_type": "",
            "background_noise_severity": "none",
            "audio_quality": "clear",
            "speaker_overlap_present": False,
            "long_silence_present": False,
            "confidence": 0.8,
        }
        prediction = validate_prediction("call.ogg", raw)
        self.assertEqual(set(prediction), set(raw) | {"filename"})
        with self.assertRaises(ValidationError):
            validate_prediction("call.ogg", {**raw, "unexpected": True})


if __name__ == "__main__":
    unittest.main()

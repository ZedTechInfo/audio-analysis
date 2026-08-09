import unittest

from app.services.audio_engine import _calibrate_confidence
from app.services.classifier import _confidence_from_evidence


class ConfidenceCalibrationTests(unittest.TestCase):
    def test_evidence_produces_distinct_bounded_scores(self):
        low = _confidence_from_evidence(0.1)
        high = _confidence_from_evidence(0.9)
        self.assertLess(low, high)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_recording_quality_reduces_confidence(self):
        clean = _calibrate_confidence(
            0.8,
            noise_severity="none",
            audio_quality="clear",
            overlap_present=False,
            long_silence=False,
            speech_coverage=0.8,
        )
        compromised = _calibrate_confidence(
            0.8,
            noise_severity="medium",
            audio_quality="slightly_impaired",
            overlap_present=True,
            long_silence=True,
            speech_coverage=0.15,
        )
        self.assertLess(compromised, clean)
        self.assertNotEqual(clean, compromised)


if __name__ == "__main__":
    unittest.main()

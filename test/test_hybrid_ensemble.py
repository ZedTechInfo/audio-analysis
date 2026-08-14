import unittest

from app.services.classifier import _ensemble_tone, _TONES

# SER checkpoint label order: neu, hap, ang, sad (no distressed).
ID2LABEL = {0: "neu", 1: "hap", 2: "ang", 3: "sad"}


class EnsembleToneTest(unittest.TestCase):
    def test_heuristic_reaches_distressed_ser_cannot(self):
        # SER is confident it's "ang" (upset); heuristic votes distressed.
        # With the heuristic weighted higher, the fifth tone still wins.
        probs = [0.0, 0.0, 0.9, 0.1]
        tone, conf = _ensemble_tone(probs, ID2LABEL, "distressed", 0.9, ser_weight=0.4)
        self.assertEqual(tone, "distressed")
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_confident_ser_still_wins(self):
        # A strong SER vote is not ignored when the heuristic is weak.
        probs = [0.9, 0.05, 0.03, 0.02]
        tone, _ = _ensemble_tone(probs, ID2LABEL, "upset", 0.5, ser_weight=0.7)
        self.assertEqual(tone, "neutral")

    def test_all_five_tones_reachable(self):
        # ser_weight=0 => the heuristic vote decides; every tone must be winnable.
        flat = [0.25, 0.25, 0.25, 0.25]
        for expected in _TONES:
            tone, _ = _ensemble_tone(flat, ID2LABEL, expected, 0.9, ser_weight=0.0)
            self.assertEqual(tone, expected)


if __name__ == "__main__":
    unittest.main()

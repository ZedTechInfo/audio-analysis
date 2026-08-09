from __future__ import annotations

import numpy as np
import librosa
from pathlib import Path
import threading

from app.config import settings

_SER_EXTRACTOR = None
_SER_MODEL = None
_SER_LOCK = threading.Lock()


def _confidence_from_evidence(evidence: float, floor: float = 0.42, ceiling: float = 0.94) -> float:
    """Map rule evidence to a bounded, deliberately non-probabilistic score."""
    return round(float(np.clip(floor + (ceiling - floor) * evidence, floor, ceiling)), 3)


def _load_ser_model():
    """Load the local SER model once and reuse it for the batch."""
    global _SER_EXTRACTOR, _SER_MODEL
    if _SER_MODEL is None:
        with _SER_LOCK:
            if _SER_MODEL is None:
                from dotenv import load_dotenv
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
                project_root = Path(__file__).resolve().parents[2]
                load_dotenv(project_root / ".env")
                _SER_EXTRACTOR = AutoFeatureExtractor.from_pretrained(settings.ser_model_id, token=True)
                _SER_MODEL = AutoModelForAudioClassification.from_pretrained(settings.ser_model_id, token=True).eval()
    return _SER_EXTRACTOR, _SER_MODEL


def _heuristic_emotion(
    feature_vector: np.ndarray,
    waveform: np.ndarray | None,
    sample_rate: int,
):
    """Classify emotion from waveform-level prosodic and spectral features."""
    if waveform is None or len(waveform) == 0:
        return "neutral", "medium", 0.60

    rms = librosa.feature.rms(y=waveform)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    f0, _, _ = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=min(librosa.note_to_hz("C7"), (sample_rate / 2) - 1),
        sr=sample_rate,
    )
    voiced_pitch = f0[np.isfinite(f0)]
    pitch_mean = float(np.mean(voiced_pitch)) if voiced_pitch.size else 0.0
    pitch_std = float(np.std(voiced_pitch)) if voiced_pitch.size else 0.0
    pitch_range = float(np.max(voiced_pitch) - np.min(voiced_pitch)) if voiced_pitch.size else 0.0

    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)[0]))
    centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]))
    _ = (feature_vector, pitch_mean)

    if rms_mean < 0.025:
        evidence = np.clip((0.025 - rms_mean) / 0.025, 0.0, 1.0)
        return "neutral", "medium", _confidence_from_evidence(float(evidence), 0.58, 0.82)
    if rms_mean > 0.03 and rms_std > 0.08 and pitch_std > 60:
        evidence = np.mean([
            np.clip((rms_mean - 0.03) / 0.05, 0.0, 1.0),
            np.clip((rms_std - 0.08) / 0.16, 0.0, 1.0),
            np.clip((pitch_std - 60) / 180, 0.0, 1.0),
        ])
        return "upset", "high", _confidence_from_evidence(float(evidence), 0.58, 0.91)
    if rms_mean > 0.03 and rms_std <= 0.08 and pitch_std < 80:
        evidence = np.mean([
            np.clip((rms_mean - 0.03) / 0.05, 0.0, 1.0),
            np.clip((0.08 - rms_std) / 0.08, 0.0, 1.0),
            np.clip((80 - pitch_std) / 80, 0.0, 1.0),
        ])
        return "satisfied", "medium", _confidence_from_evidence(float(evidence), 0.55, 0.88)
    if rms_mean > 0.02 and pitch_std > 100 and pitch_range > 1000:
        evidence = np.mean([
            np.clip((rms_mean - 0.02) / 0.05, 0.0, 1.0),
            np.clip((pitch_std - 100) / 220, 0.0, 1.0),
            np.clip((pitch_range - 1000) / 1800, 0.0, 1.0),
        ])
        return "distressed", "high", _confidence_from_evidence(float(evidence), 0.62, 0.94)
    evidence = np.mean([
        np.clip(rms_mean / 0.04, 0.0, 1.0),
        np.clip(pitch_std / 140, 0.0, 1.0),
        np.clip(pitch_range / 1200, 0.0, 1.0),
    ])
    return "neutral", "medium", _confidence_from_evidence(float(evidence), 0.48, 0.7)


def predict_emotion(
    feature_vector: np.ndarray,
    waveform: np.ndarray | None = None,
    sample_rate: int = 16000,
    use_model: bool = False,
):
    """Predict AutoAce tone using heuristics or local pretrained SER."""
    if not use_model or waveform is None:
        return _heuristic_emotion(feature_vector, waveform, sample_rate)

    import torch

    extractor, model = _load_ser_model()
    window_size = sample_rate * settings.ser_window_seconds
    hop_size = sample_rate * settings.ser_hop_seconds
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) <= window_size:
        chunks = [waveform]
    else:
        starts = range(0, len(waveform) - window_size + 1, hop_size)
        chunks = [waveform[start : start + window_size] for start in starts]

    probabilities = []
    for chunk in chunks:
        if len(chunk) < sample_rate:
            continue
        inputs = extractor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        with _SER_LOCK, torch.no_grad():
            logits = model(**inputs).logits
        probabilities.append(torch.softmax(logits, dim=-1)[0].cpu().numpy())

    if not probabilities:
        return _heuristic_emotion(feature_vector, waveform, sample_rate)

    probability_matrix = np.asarray(probabilities)
    mean_probabilities = np.mean(probability_matrix, axis=0)
    label_index = int(np.argmax(mean_probabilities))
    source_label = str(model.config.id2label[label_index]).lower()
    # The base model only has four classes (neu, hap, ang, sad) and does not
    # natively support AutoAce's separate distressed category.
    tone = {
        "neu": "neutral",
        "hap": "satisfied",
        "ang": "upset",
        "sad": "frustrated",
    }.get(source_label, "neutral")
    ranked = np.sort(mean_probabilities)
    top_probability = float(ranked[-1])
    margin = float(ranked[-1] - ranked[-2]) if len(ranked) > 1 else top_probability
    entropy = float(-np.sum(mean_probabilities * np.log(np.clip(mean_probabilities, 1e-9, 1.0))))
    normalized_entropy = entropy / max(np.log(len(mean_probabilities)), 1.0)
    chunk_agreement = float(np.mean(np.argmax(probability_matrix, axis=1) == label_index))
    confidence = _confidence_from_evidence(
        float(np.mean([top_probability, np.clip(margin * 2.0, 0.0, 1.0), 1.0 - normalized_entropy, chunk_agreement])),
        0.35,
        0.94,
    )
    intensity = "high" if confidence >= 0.75 else "medium" if confidence >= 0.5 else "low"
    return tone, intensity, confidence

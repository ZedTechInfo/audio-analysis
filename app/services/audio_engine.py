from __future__ import annotations

import warnings

import librosa
import numpy as np

from app.config import settings
from app.services.errors import AudioAnalysisError

warnings.filterwarnings("ignore", module="librosa")


def _frame_mask(intervals: np.ndarray, frame_count: int, hop_length: int, sample_count: int) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=bool)
    centers = np.minimum(np.arange(frame_count) * hop_length + hop_length // 2, sample_count - 1)
    for start, end in intervals:
        mask |= (centers >= start) & (centers < end)
    return mask


def _noise_type(flatness: float, centroid: float, bandwidth: float) -> str:
    if flatness >= 0.18:
        return "sharp static"
    if centroid < 900 and bandwidth < 1200:
        return "mechanical noise"
    if centroid > 2800:
        return "office chatter"
    if flatness < 0.04 and bandwidth < 1800:
        return "TV"
    return "ambient noise"


def _calibrate_confidence(
    confidence: float,
    *,
    noise_severity: str,
    audio_quality: str,
    overlap_present: bool,
    long_silence: bool,
    speech_coverage: float,
) -> float:
    """Reduce classifier confidence when the recording weakens the evidence."""
    penalty = {
        "none": 1.0,
        "low": 0.98,
        "medium": 0.9,
        "high": 0.78,
    }[noise_severity]
    penalty *= {"clear": 1.0, "slightly_impaired": 0.9, "severely_impaired": 0.72}[audio_quality]
    penalty *= 0.93 if overlap_present else 1.0
    penalty *= 0.9 if long_silence else 1.0
    if speech_coverage < 0.1:
        penalty *= 0.82
    elif speech_coverage < 0.25:
        penalty *= 0.92
    return round(float(np.clip(confidence * penalty, 0.0, 1.0)), 3)


def analyze_audio_file(file_path: str, use_ser: bool = True) -> dict:
    """Analyze one file and return a schema-compatible result or raise an error."""
    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0 or not np.isfinite(y).all():
            raise ValueError("Audio is empty or contains non-finite samples")

        hop_length = 256
        frame_length = 1024
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        max_rms = max(float(np.max(rms)), np.finfo(float).eps)
        silent_frames = rms < 0.01 * max_rms
        padded_silence = np.concatenate(([False], silent_frames, [False]))
        transitions = np.diff(padded_silence.astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        longest_silence = float(np.max(ends - starts) * hop_length / sr) if len(starts) else 0.0

        speech_intervals = librosa.effects.split(y, top_db=settings.vad_top_db)
        speech_mask = _frame_mask(speech_intervals, len(rms), hop_length, len(y))
        noise_mask = ~speech_mask

        flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        noise_rms = float(np.median(rms[noise_mask])) if np.any(noise_mask) else 0.0
        speech_rms = float(np.median(rms[speech_mask])) if np.any(speech_mask) else max_rms
        noise_ratio = noise_rms / max(speech_rms, np.finfo(float).eps)
        background_noise_present = bool(
            noise_rms >= settings.noise_floor_threshold
            and noise_ratio >= settings.noise_ratio_threshold
        )

        # Fallback: sharp static can have low relative energy but high spectral flatness.
        if not background_noise_present and np.any(noise_mask):
            noise_flatness_for_fallback = float(np.median(flatness[noise_mask]))
            if noise_flatness_for_fallback >= 0.04 and noise_rms >= settings.noise_floor_threshold:
                background_noise_present = True

        # Broadband static can overlap speech and therefore evade the
        # non-speech mask. Require both a high-flatness tail and high spectral
        # centroid so ordinary voiced energy is not called static.
        speech_flatness = float(np.percentile(flatness[speech_mask], 95)) if np.any(speech_mask) else 0.0
        speech_centroid = float(np.percentile(centroid[speech_mask], 90)) if np.any(speech_mask) else 0.0
        in_speech_static = speech_flatness >= 0.05 and speech_centroid >= 2000
        if not background_noise_present and in_speech_static:
            background_noise_present = True

        if background_noise_present:
            noise_flatness = float(np.median(flatness[noise_mask])) if np.any(noise_mask) else float(np.median(flatness))
            noise_centroid = float(np.median(centroid[noise_mask])) if np.any(noise_mask) else float(np.median(centroid))
            noise_bandwidth = float(np.median(bandwidth[noise_mask])) if np.any(noise_mask) else float(np.median(bandwidth))

            # Check broadband static before narrower TV/mechanical signatures.
            if in_speech_static or noise_flatness >= 0.18 or noise_bandwidth > 2500:
                noise_type = "sharp static"
            elif noise_centroid < 900 and noise_bandwidth < 1200:
                noise_type = "mechanical noise"
            elif noise_centroid > 2800:
                noise_type = "office chatter"
            elif noise_flatness < 0.04 and noise_bandwidth < 1800:
                noise_type = "TV"
            else:
                noise_type = "ambient noise"

            if noise_type in {"sharp static", "mechanical noise", "office chatter"}:
                noise_severity = "medium"
            elif noise_type == "TV" and noise_ratio >= settings.noise_ratio_threshold:
                noise_severity = "medium"
            elif noise_ratio >= 0.35:
                noise_severity = "high"
            elif noise_ratio >= 0.1:
                noise_severity = "medium"
            else:
                noise_severity = "low"
        else:
            noise_type = ""
            noise_severity = "none"

        clipping_ratio = float(np.mean(np.abs(y) >= 0.99))
        peak_level = float(np.percentile(np.abs(y), 99.9))
        if clipping_ratio > 0.005:
            audio_quality = "severely_impaired"
        elif clipping_ratio > 0.0005 or peak_level < 0.02:
            audio_quality = "slightly_impaired"
        else:
            audio_quality = "clear"

        # Mono audio cannot prove overlap. Require both unusually concentrated
        # speech energy and independent spectral/energy variation as a
        # conservative proxy for two active speakers.
        if np.any(speech_mask):
            speech_values = rms[speech_mask]
            high_energy_ratio = float(np.mean(speech_values > 1.5 * np.median(speech_values)))
            energy_variation = float(np.std(speech_values) / max(np.mean(speech_values), np.finfo(float).eps))
            centroid_variation = float(np.std(centroid[speech_mask]))
            overlap_ratio = high_energy_ratio if energy_variation >= 0.9 or centroid_variation >= 650 else 0.0
        else:
            overlap_ratio = 0.0
        speaker_overlap_present = overlap_ratio >= settings.overlap_frame_ratio

        from .classifier import predict_emotion

        tone, intensity, confidence = predict_emotion(
            np.zeros(13), waveform=y, sample_rate=sr, use_model=use_ser
        )
        long_silence_present = bool(longest_silence > 10.0)
        confidence = _calibrate_confidence(
            confidence,
            noise_severity=noise_severity,
            audio_quality=audio_quality,
            overlap_present=bool(speaker_overlap_present),
            long_silence=long_silence_present,
            speech_coverage=float(np.mean(speech_mask)),
        )
        return {
            "emotional_tone": tone,
            "emotional_intensity": intensity,
            "background_noise_present": background_noise_present,
            "background_noise_type": noise_type,
            "background_noise_severity": noise_severity,
            "audio_quality": audio_quality,
            "speaker_overlap_present": bool(speaker_overlap_present),
            "long_silence_present": long_silence_present,
            "confidence": confidence,
        }
    except AudioAnalysisError:
        raise
    except Exception as exc:
        raise AudioAnalysisError(f"Unable to analyze audio file: {exc}") from exc

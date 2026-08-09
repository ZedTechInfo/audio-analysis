from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

class EmotionalTone(str, Enum):
    neutral = "neutral"
    satisfied = "satisfied"
    frustrated = "frustrated"
    upset = "upset"
    distressed = "distressed"

class EmotionalIntensity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class NoiseSeverity(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"

class AudioQuality(str, Enum):
    clear = "clear"
    slightly_impaired = "slightly_impaired"
    severely_impaired = "severely_impaired"

class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str
    error: str | None = None


class JobHistoryItem(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str
    job_type: str
    source_name: str | None = None
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    average_confidence: float | None = None
    result_available: bool = False
    error: str | None = None


class JobHistoryResponse(BaseModel):
    items: list[JobHistoryItem]
    total: int


class LogItem(BaseModel):
    id: int
    created_at: str
    level: str
    source: str
    message: str
    job_id: str | None = None
    artifact: str | None = None


class LogResponse(BaseModel):
    items: list[LogItem]
    page: int
    page_size: int
    total: int
    page_count: int

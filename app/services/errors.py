class BatchValidationError(ValueError):
    """Raised when an uploaded batch does not satisfy the input contract."""


class AudioAnalysisError(RuntimeError):
    """Raised when an audio file cannot be analyzed."""

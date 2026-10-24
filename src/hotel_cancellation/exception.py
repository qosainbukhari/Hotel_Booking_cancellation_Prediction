class HotelCancellationError(Exception):
    """Base error for every error in this project."""


class ConfigError(HotelCancellationError):
    """Raised when a config file is missing or has wrong values."""


class DataValidationError(HotelCancellationError):
    """Raised when data does not match the expected rules (schema)."""


class ModelError(HotelCancellationError):
    """Raised when a model cannot be trained, saved or loaded."""

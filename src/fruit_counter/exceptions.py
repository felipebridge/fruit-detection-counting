"""Exception hierarchy.

Every error raised deliberately by the package derives from :class:`FruitCounterError`,
so callers (CLI, API servers) can distinguish expected failures from bugs.
"""


class FruitCounterError(Exception):
    """Base class for all package errors."""


class ConfigError(FruitCounterError):
    """Invalid or inconsistent configuration."""


class InputError(FruitCounterError):
    """Missing, unreadable or unsupported input source."""


class ModelError(FruitCounterError):
    """The detection model could not be loaded or used."""


class OutputError(FruitCounterError):
    """Results could not be written."""

class FruitCounterError(Exception):
    """Base class for expected failures, as opposed to bugs."""


class ConfigError(FruitCounterError):
    pass


class InputError(FruitCounterError):
    pass


class ModelError(FruitCounterError):
    pass


class OutputError(FruitCounterError):
    pass

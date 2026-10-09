class FlyAIError(RuntimeError):
    """Base error for the local FlyAI CLI integration."""


class FlyAICommandNotFoundError(FlyAIError):
    pass


class FlyAITimeoutError(FlyAIError):
    pass


class FlyAIUnavailableError(FlyAIError):
    pass


class FlyAIInvalidJsonError(FlyAIError):
    pass


class FlyAIResponseError(FlyAIError):
    pass

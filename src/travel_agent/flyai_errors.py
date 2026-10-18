class FlyAIError(RuntimeError):
    """Base error for the local FlyAI CLI integration."""


class FlyAICommandNotFoundError(FlyAIError):
    pass


class FlyAITimeoutError(FlyAIError):
    pass


class FlyAIUnavailableError(FlyAIError):
    pass


class FlyAIRateLimitError(FlyAIUnavailableError):
    """FlyAI 上游限流（HTTP 429）。属于可恢复错误，应当退避重试。"""


class FlyAIInvalidJsonError(FlyAIError):
    pass


class FlyAIResponseError(FlyAIError):
    pass

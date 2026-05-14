class AceAIError(Exception):
    """Base exception class for AceAI errors."""


class AceAIConfigurationError(AceAIError):
    """Raised when AceAI is misconfigured or missing required settings."""


class UnannotatedToolParamError(AceAIConfigurationError):
    """Raised when a tool parameter does not use typing.Annotated."""


class AceAIValidationError(AceAIError):
    """Raised when inputs fail validation."""


class AceAIRuntimeError(AceAIError):
    """Raised when runtime execution fails unexpectedly."""


class AceAIImplementationError(AceAIError):
    """Raised when required functionality is not implemented."""


class LLMProviderError(AceAIError):
    """Wrapper for LLM provider errors bubbled up through AceAI."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        context_window: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.context_window = context_window
        self.status_code = status_code


class LLMContextWindowExceededError(LLMProviderError):
    """Raised when a provider rejects a request for exceeding context limits."""


__all__ = [
    "AceAIError",
    "AceAIConfigurationError",
    "UnannotatedToolParamError",
    "AceAIValidationError",
    "AceAIRuntimeError",
    "AceAIImplementationError",
    "LLMProviderError",
    "LLMContextWindowExceededError",
]

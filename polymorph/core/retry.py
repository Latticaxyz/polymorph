"""Retry utilities for robust API calls."""

import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import httpx

from polymorph.utils.logging import get_logger


logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    retry_exceptions: tuple[type[Exception], ...] = (
        httpx.HTTPError,
        asyncio.TimeoutError,
    ),
):
    """Decorator to add retry logic to async functions.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        retry_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(min=min_wait, max=max_wait),
                retry=retry_if_exception_type(retry_exceptions),
                reraise=True,
            ):
                with attempt:
                    try:
                        return await func(*args, **kwargs)
                    except retry_exceptions as e:
                        logger.warning(
                            f"Attempt {attempt.retry_state.attempt_number}/{max_attempts} "
                            f"failed for {func.__name__}: {e}"
                        )
                        raise

        return wrapper

    return decorator


class RateLimitError(Exception):
    """Exception raised when API rate limit is exceeded."""

    pass

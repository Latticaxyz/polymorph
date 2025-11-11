import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar, cast

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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

        return cast(F, wrapper)

    return decorator


class RateLimitError(Exception):
    pass

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from polymorph.utils.logging import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    retry_exceptions: tuple[type[Exception], ...] = (
        httpx.HTTPError,
        asyncio.TimeoutError,
    ),
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
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
            # This is logically unreachable, but makes the type checker happy
            raise RuntimeError("with_retry: AsyncRetrying loop exited without returning or raising")

        return wrapper

    return decorator


class RateLimitError(Exception):
    pass

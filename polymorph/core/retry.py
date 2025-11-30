import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from polymorph.core.rate_limit import RateLimitError
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _should_retry(exception: BaseException) -> bool:
    if isinstance(exception, RateLimitError):
        logger.warning("Rate limit hit (429), will retry with backoff")
        return True

    if isinstance(exception, httpx.HTTPStatusError):
        status = exception.response.status_code

        if 500 <= status < 600:
            logger.warning(f"Server error {status}, will retry")
            return True

        if status == 429:
            logger.warning("Rate limit (429) not wrapped in RateLimitError, will retry")
            return True

        logger.debug(f"Client error {status}, will not retry")
        return False

    if isinstance(exception, (httpx.RequestError, asyncio.TimeoutError)):
        logger.warning(f"Network error {type(exception).__name__}, will retry")
        return True

    return False


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    retry_exceptions: tuple[type[Exception], ...] = (
        httpx.HTTPError,
        asyncio.TimeoutError,
        RateLimitError,
    ),
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(min=min_wait, max=max_wait, multiplier=1.5),
                retry=retry_if_exception(_should_retry),
                reraise=True,
            ):
                with attempt:
                    try:
                        return await func(*args, **kwargs)
                    except retry_exceptions as e:
                        attempt_num = attempt.retry_state.attempt_number
                        logger.warning(
                            f"Attempt {attempt_num}/{max_attempts} "
                            f"failed for {func.__name__}: {type(e).__name__}: {e}"
                        )

                        if isinstance(e, RateLimitError) and attempt_num < max_attempts:
                            extra_wait = min_wait * (attempt_num + 1)
                            logger.info(f"Rate limit error, adding {extra_wait:.1f}s extra wait")
                            await asyncio.sleep(extra_wait)

                        raise

            # This is logically unreachable, but makes the type checker happy
            raise RuntimeError("with_retry: AsyncRetrying loop exited without returning or raising")

        return wrapper

    return decorator

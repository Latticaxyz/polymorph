import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import ClassVar

from polymorph.utils.logging import get_logger

logger = get_logger(__name__)


# TODO: ARCHITECTURE - Current design raises RateLimitError which triggers retry with exponential backoff
# TODO: This is inefficient compared to just sleeping when rate limit is reached
# TODO: Consider redesigning to use asyncio.sleep() instead of raise/retry pattern
# TODO: Would eliminate unnecessary exception handling overhead and make timing more predictable


class RateLimiter:
    _instances: ClassVar[dict[str, "RateLimiter"]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self, name: str, max_requests: int, time_window_seconds: float):
        self.name = name
        self.max_requests = max_requests
        self.time_window = timedelta(seconds=time_window_seconds)
        self.requests: deque[datetime] = deque()
        self._instance_lock = asyncio.Lock()

        logger.debug(f"RateLimiter '{name}' initialized: " f"{max_requests} requests per {time_window_seconds}s")

    @classmethod
    async def get_instance(cls, name: str, max_requests: int, time_window_seconds: float) -> "RateLimiter":
        async with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, max_requests, time_window_seconds)
            return cls._instances[name]

    def _cleanup_old_requests(self, now: datetime) -> None:
        cutoff = now - self.time_window
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    async def acquire(self) -> None:
        # TODO: CRITICAL BUG - Race condition! self.requests.append(now) is outside the lock
        # TODO: Multiple concurrent requests can all pass the limit check, then all append to queue
        # TODO: Fix by moving append inside the lock OR redesign to use asyncio.sleep instead of raising
        # TODO: Consider replacing exception-based design with sleep-based waiting for better efficiency
        async with self._instance_lock:
            now = datetime.now()
            self._cleanup_old_requests(now)

            if len(self.requests) >= self.max_requests:
                oldest = self.requests[0]
                wait_until = oldest + self.time_window
                sleep_time = (wait_until - now).total_seconds()

                if sleep_time > 0:
                    logger.warning(
                        f"RateLimiter '{self.name}': at limit "
                        f"({len(self.requests)}/{self.max_requests}), "
                        f"raising RateLimitError"
                    )
                    raise RateLimitError(
                        f"Rate limit exceeded for '{self.name}': "
                        f"{self.max_requests} requests per {self.time_window.total_seconds()}s"
                    )

        self.requests.append(now)  # TODO: MOVE THIS INSIDE THE LOCK ABOVE!

    def get_stats(self) -> dict[str, object]:
        now = datetime.now()
        self._cleanup_old_requests(now)

        return {
            "name": self.name,
            "current_count": len(self.requests),
            "max_requests": self.max_requests,
            "time_window_seconds": self.time_window.total_seconds(),
            "utilization_pct": (len(self.requests) / self.max_requests * 100),
        }


class RateLimitError(Exception):
    pass


GAMMA_RATE_LIMIT = {"max_requests": 120, "time_window_seconds": 10}
CLOB_RATE_LIMIT = {"max_requests": 95, "time_window_seconds": 10}
DATA_API_RATE_LIMIT = {"max_requests": 190, "time_window_seconds": 10}

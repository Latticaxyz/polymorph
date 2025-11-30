import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import ClassVar

from polymorph.utils.logging import get_logger

logger = get_logger(__name__)


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
        async with self._instance_lock:
            now = datetime.now()
            self._cleanup_old_requests(now)

            if len(self.requests) >= self.max_requests:
                oldest = self.requests[0]
                wait_until = oldest + self.time_window
                sleep_time = (wait_until - now).total_seconds()

                if sleep_time > 0:
                    logger.debug(
                        f"RateLimiter '{self.name}': at limit "
                        f"({len(self.requests)}/{self.max_requests}), "
                        f"sleeping {sleep_time:.2f}s"
                    )
                    await asyncio.sleep(sleep_time + 0.1)

                    now = datetime.now()
                    self._cleanup_old_requests(now)

        self.requests.append(now)

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

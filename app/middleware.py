import time
from collections import defaultdict

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter keyed by client IP.

    Each IP is allowed `limit` requests within any rolling 60-second window.
    Timestamps outside the window are evicted on every request to bound memory.
    Set limit=0 to disable.
    """

    def __init__(self, app, limit: int = 60):
        super().__init__(app)
        self._limit = limit
        self._window = 60.0
        # IP → list of request timestamps inside the current window
        self._store: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if self._limit == 0:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self._window

        timestamps = self._store[ip]
        # Evict expired timestamps
        self._store[ip] = [t for t in timestamps if t > cutoff]

        if len(self._store[ip]) >= self._limit:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded: {self._limit} requests/minute per IP."},
                headers={"Retry-After": "60"},
            )

        self._store[ip].append(now)
        return await call_next(request)

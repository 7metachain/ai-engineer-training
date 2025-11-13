import asyncio
import time
from typing import Callable, Dict
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class TokenBucket:
    """令牌桶算法：按速率填充令牌，支持突发容量。"""

    def __init__(self, rate: float, capacity: int):
        self.rate = float(rate)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, cost: float = 1.0) -> tuple[bool, int, float]:
        """消费令牌。

        Returns:
            (allowed, remaining, retry_after)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            if elapsed > 0:
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.updated_at = now

            if self.tokens >= cost:
                self.tokens -= cost
                return True, max(0, int(self.tokens)), 0.0

            need = cost - self.tokens
            retry_after = need / self.rate if self.rate > 0 else float("inf")
            return False, 0, retry_after

    def time_to_full(self) -> float:
        return (self.capacity - self.tokens) / self.rate if self.rate > 0 else float("inf")

class TokenBucketRateLimiter(BaseHTTPMiddleware):
    """基于令牌桶的限流中间件。默认按客户端IP限流。"""

    def __init__(
        self,
        app: FastAPI,
        rate_per_sec: float,
        burst_capacity: int,
        key_func: Callable[[Request], str] | None = None,
        tokens_per_request: float = 1.0,
        exempt_paths: set[str] | None = None,
        ttl_seconds: int = 600,
    ) -> None:
        super().__init__(app)
        self.rate = float(rate_per_sec)
        self.capacity = int(burst_capacity)
        self.tokens_per_request = float(tokens_per_request)
        self.key_func = key_func or self._default_key
        self.exempt_paths = exempt_paths or set()
        self.ttl_seconds = int(ttl_seconds)
        self.buckets: Dict[str, TokenBucket] = {}
        self.last_seen: Dict[str, float] = {}
        self._global_lock = asyncio.Lock()

    def _default_key(self, request: Request) -> str:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            ip = xff.split(",")[0].strip()
        else:
            ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
        return ip

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        key = self.key_func(request)
        now = time.monotonic()

        async with self._global_lock:
            bucket = self.buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(rate=self.rate, capacity=self.capacity)
                self.buckets[key] = bucket
            self.last_seen[key] = now

            if len(self.last_seen) > 1000:
                cutoff = now - self.ttl_seconds
                stale_keys = [k for k, t in self.last_seen.items() if t < cutoff]
                for k in stale_keys:
                    self.buckets.pop(k, None)
                    self.last_seen.pop(k, None)

        allowed, remaining, retry_after = await bucket.consume(self.tokens_per_request)

        policy = f"token_bucket; rate={self.rate}/s; burst={self.capacity}"
        if not allowed:
            reset = max(0, int(retry_after))
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后重试", "retry_after": reset},
                headers={
                    "Retry-After": str(reset),
                    "X-RateLimit-Policy": policy,
                    "X-RateLimit-Limit": str(self.capacity),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset),
                },
            )

        response = await call_next(request)
        reset = max(0, int(bucket.time_to_full()))
        response.headers["X-RateLimit-Policy"] = policy
        response.headers["X-RateLimit-Limit"] = str(self.capacity)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset)
        return response

app = FastAPI(title="限流中间件演示（令牌桶）")

app.add_middleware(
    TokenBucketRateLimiter,
    rate_per_sec=5.0,
    burst_capacity=10,
    tokens_per_request=1.0,
    exempt_paths={"/docs", "/openapi.json"},
)

@app.get("/")
async def root():
    return {"message": "OK"}

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.get("/work")
async def work():
    import asyncio
    await asyncio.sleep(0.2)
    return {"message": "done"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("限流中间件:app", host="0.0.0.0", port=8000, reload=True)
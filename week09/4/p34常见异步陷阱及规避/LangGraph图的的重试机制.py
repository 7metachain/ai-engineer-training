try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    from langgraph.graph import StateGraph
    START = "__start__"
    END = "__end__"
import asyncio
import time
import logging
import random
from functools import wraps
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

NODE_TIMEOUTS = {
    "get_weather": 5.0,
    "default": 5.0,
}

ATTEMPT_COUNTER = 0

def with_timeout(timeout_seconds: float):
    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(f"函数 {func.__name__} 执行超时 ({timeout_seconds}s)")
                raise TimeoutError(f"操作超时: {timeout_seconds}秒")
        return wrapper
    return decorator

class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
        retryable_exceptions: tuple = (TimeoutError, ConnectionError),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

def with_retry(retry_config: RetryConfig | None = None):
    if retry_config is None:
        retry_config = RetryConfig()

    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception: Exception | None = None
            for attempt in range(retry_config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if not isinstance(e, retry_config.retryable_exceptions):
                        raise
                    last_exception = e
                    logger.warning(
                        f"尝试 {attempt + 1}/{retry_config.max_attempts} 失败: {type(e).__name__}: {e}"
                    )
                    if attempt < retry_config.max_attempts - 1:
                        delay = (
                            retry_config.base_delay * (2 ** attempt)
                            if retry_config.exponential_backoff
                            else retry_config.base_delay
                        )
                        if retry_config.jitter:
                            delay *= random.uniform(0.8, 1.2)
                        delay = min(delay, retry_config.max_delay)
                        logger.info(f"等待 {delay:.2f} 秒后重试...")
                        await asyncio.sleep(delay)
            logger.error(f"所有 {retry_config.max_attempts} 次尝试都失败")
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator

def resilient_node(**retry_kwargs):
    def decorator(func: Callable[..., Awaitable[Any]]):
        func = with_timeout(NODE_TIMEOUTS.get(func.__name__, NODE_TIMEOUTS["default"]))(func)
        func = with_retry(RetryConfig(**retry_kwargs))(func)
        return func
    return decorator

async def weather(city: str) -> str:
    await asyncio.sleep(4)
    return f"{city} 晴，25°C"

@resilient_node(max_attempts=3, base_delay=2.0)
async def get_weather(state):
    global ATTEMPT_COUNTER
    ATTEMPT_COUNTER += 1
    city = state.get("city", "北京")
    delay = state.get("delay", 0)
    if delay:
        await asyncio.sleep(delay)
    if ATTEMPT_COUNTER == 1:
        await asyncio.sleep(2.5)
    result = await weather(city)
    return {"result": result, "attempt": ATTEMPT_COUNTER}

# 构建极简 LangGraph 工作流
workflow = StateGraph(dict)
workflow.add_node("get_weather", get_weather)
workflow.add_edge(START, "get_weather")
workflow.add_edge("get_weather", END)

# 运行
async def main():
    app = workflow.compile()
    inputs = {"city": "上海"}
    async for event in app.astream(inputs):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    from langgraph.graph import StateGraph
    START = "__start__"
    END = "__end__"
import asyncio
import time
import logging
from functools import wraps
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

NODE_TIMEOUTS = {
    "get_weather": 2.0,
    "default": 5.0,
}

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

async def weather(city: str) -> str:
    await asyncio.sleep(3)
    return f"{city} 晴，25°C"

# 定义异步节点
@with_timeout(NODE_TIMEOUTS.get("get_weather", NODE_TIMEOUTS["default"]))
async def get_weather(state):
    city = state.get("city", "北京")
    delay = state.get("delay", 0)
    if delay:
        await asyncio.sleep(delay)
    result = await weather(city)
    return {"result": result}

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

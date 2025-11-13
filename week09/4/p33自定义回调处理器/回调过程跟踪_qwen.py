import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class AsyncProgressCallback(AsyncCallbackHandler):
    """异步进度回调处理器（适配 DashScope / qwen-turbo）"""

    def __init__(self):
        self.progress_updates: List[Dict[str, Any]] = []
        self.start_time = None
        self.current_step = 0
        self.total_steps = 0

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """LLM 开始时调用"""
        self.start_time = datetime.now()
        update = {
            "timestamp": datetime.now().isoformat(),
            "event": "llm_start",
            "prompts": prompts,
            "step": self.current_step
        }
        self.progress_updates.append(update)
        print("[Callback]", update)

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """新 token 生成时调用"""
        update = {
            "timestamp": datetime.now().isoformat(),
            "event": "new_token",
            "token": token,
            "step": self.current_step
        }
        self.progress_updates.append(update)
        print("[Callback]", update)

    async def on_llm_end(self, response, **kwargs: Any) -> None:
        """LLM 结束时调用"""
        duration = (datetime.now() - self.start_time).total_seconds()
        # DashScope 的 response 结构可能不同，兼容处理
        token_usage = getattr(response, "token_usage", {}) or {}
        total_tokens = token_usage.get("total_tokens", 0)
        update = {
            "timestamp": datetime.now().isoformat(),
            "event": "llm_end",
            "duration": duration,
            "token_count": total_tokens,
            "step": self.current_step
        }
        self.progress_updates.append(update)
        print("[Callback]", update)

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """链开始时调用"""
        self.current_step += 1
        update = {
            "timestamp": datetime.now().isoformat(),
            "event": "chain_start",
            "chain": serialized.get("name", "unknown"),
            "inputs": inputs,
            "step": self.current_step
        }
        self.progress_updates.append(update)
        print("[Callback]", update)

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """链结束时调用"""
        update = {
            "timestamp": datetime.now().isoformat(),
            "event": "chain_end",
            "outputs": outputs,
            "step": self.current_step
        }
        self.progress_updates.append(update)
        print("[Callback]", update)

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """工具开始时调用"""
        self.current_step += 1
        update = {
            "timestamp": datetime.now().isoformat(),
            "event": "tool_start",
            "tool": serialized.get("name", "unknown"),
            "input": input_str,
            "step": self.current_step
        }
        self.progress_updates.append(update)
        print("[Callback]", update)

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """工具结束时调用"""
        update = {
            "timestamp": datetime.now().isoformat(),
            "event": "tool_end",
            "output": output,
            "step": self.current_step
        }
        self.progress_updates.append(update)
        print("[Callback]", update)

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Agent 动作时调用"""
        update = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_action",
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
            "step": self.current_step
        }
        self.progress_updates.append(update)
        print("[Callback]", update)

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Agent 完成时调用"""
        duration = (datetime.now() - self.start_time).total_seconds()
        update = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_finish",
            "return_values": finish.return_values,
            "log": finish.log,
            "total_duration": duration,
            "total_steps": self.current_step
        }
        self.progress_updates.append(update)
        print("[Callback]", update)


class WebSocketProgressTracker:
    """WebSocket 进度跟踪器（示例，可按需接入真实 WebSocket）"""

    def __init__(self, websocket=None):
        self.websocket = websocket
        self.callback = AsyncProgressCallback()

    async def send_progress_update(self, update: Dict[str, Any]):
        """发送进度更新到 WebSocket（示例打印）"""
        # 这里仅打印，可替换为真实 WebSocket 发送
        print("[WebSocket 模拟] 发送进度更新:", update)

    async def track_execution(self, runnable, *args, **kwargs):
        """跟踪可运行对象的执行"""
        config = {"callbacks": [self.callback]}
        try:
            await self.send_progress_update({
                "status": "started",
                "timestamp": datetime.now().isoformat()
            })
            result = await runnable.ainvoke(*args, config=config, **kwargs)
            await self.send_progress_update({
                "status": "completed",
                "result": str(result),
                "timestamp": datetime.now().isoformat()
            })
            return result
        except Exception as e:
            await self.send_progress_update({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise


def _make_llm():
    """
    初始化 DashScope（通义千问）聊天模型。
    依赖环境变量：
        DASHSCOPE_API_KEY  - 必需，你的 DashScope API Key
        DASHSCOPE_MODEL    - 可选，默认 qwen-turbo
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY 环境变量")
    return ChatTongyi(model=model, temperature=0.9)


def _make_chain():
    """
    构建 Prompt → LLM → 字符串输出 的可组合链。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "Say hello"),
        ]
    )
    llm = _make_llm()
    return prompt | llm | StrOutputParser()


async def async_generate_with_callback(chain, product: str):
    """带回调的异步调用"""
    tracker = WebSocketProgressTracker()
    return await tracker.track_execution(chain, {"product": product})


async def generate_concurrently_with_callback():
    """并发调用并跟踪进度（debug：1个任务）"""
    chain = _make_chain()
    tasks = [async_generate_with_callback(chain, "toothpaste") for _ in range(1)]
    results = await asyncio.gather(*tasks)
    return results


async def main():
    """主函数：演示带回调的并发调用"""
    print("=== 开始并发调用（带进度回调） ===")
    results = await generate_concurrently_with_callback()
    print("=== 全部完成 ===")
    for idx, r in enumerate(results, 1):
        print(f"结果 {idx}: {r}")


if __name__ == "__main__":
    asyncio.run(main())
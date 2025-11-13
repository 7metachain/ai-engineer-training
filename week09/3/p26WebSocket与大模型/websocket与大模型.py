import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import httpx

app = FastAPI()

async def _stream_qwen(prompt: str, system: str | None, websocket: WebSocket):
    """使用 DashScope 兼容模式 Chat Completions 的 SSE 异步流式接口。

    为避免阻塞事件循环，改为通过 httpx.AsyncClient 直接消费 SSE 数据，
    并将增量内容逐段通过 WebSocket 发送给客户端。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        await websocket.send_text("错误：未设置 DASHSCOPE_API_KEY")
        return

    # 兼容模式（OpenAI 风格）SSE 端点，地域可通过环境变量切换
    url = os.getenv(
        "DASHSCOPE_COMPAT_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen-turbo",
        "messages": messages,
        "stream": True,
    }

    try:
        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    # 读取错误信息并反馈
                    try:
                        body = await resp.aread()
                        msg = body.decode("utf-8", errors="ignore")
                    except Exception:
                        msg = ""
                    await websocket.send_text(f"错误：HTTP {resp.status_code} {msg}")
                    return

                # 逐行消费 SSE（形如："data: {json}"）
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    # 过滤无关字段（event:, id:, retry: 等），只处理 data: 行
                    if line.startswith("data:"):
                        data = line[5:].strip()
                    else:
                        continue

                    # 结束标记
                    if data == "[DONE]":
                        break

                    # 尝试解析 JSON，并提取增量内容
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        # 若不是 JSON，直接原样发送，便于故障定位
                        if data:
                            await websocket.send_text(data)
                        continue

                    # OpenAI 兼容模式通常在 choices[0].delta.content 中携带增量内容
                    choices = obj.get("choices") or []
                    content = ""
                    if choices:
                        delta = choices[0].get("delta") or {}
                        message = choices[0].get("message") or {}
                        content = (
                            delta.get("content")
                            or message.get("content")
                            or ""
                        )

                    if content:
                        await websocket.send_text(content)

    except httpx.RequestError as e:
        await websocket.send_text(f"错误：网络请求失败 {e}")
    except WebSocketDisconnect:
        # 客户端断开连接，结束本次流式推送
        return
    except Exception as e:
        await websocket.send_text(f"错误：{e}")

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"prompt": raw}

            prompt = payload.get("prompt") or ""
            system = payload.get("system")
            if not prompt:
                await websocket.send_text("错误：prompt 不能为空")
                continue

            await _stream_qwen(prompt, system, websocket)

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    port = int(os.getenv("WS_PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

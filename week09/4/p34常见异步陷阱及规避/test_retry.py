import asyncio
import importlib.util
import time
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).with_name("LangGraph图的的重试机制.py")
    mod_name = f"retry_mod_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.mark.asyncio
async def test_with_retry_success_fast():
    m = _load_module()

    attempts = {"n": 0}

    @m.with_retry(
        m.RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            exponential_backoff=False,
            jitter=False,
            retryable_exceptions=(TimeoutError,),
        )
    )
    async def sometimes_fail_once():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("first fail")
        return "ok"

    result = await sometimes_fail_once()
    assert result == "ok"
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_with_retry_exhausts_and_raises():
    m = _load_module()

    attempts = {"n": 0}

    @m.with_retry(
        m.RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            exponential_backoff=False,
            jitter=False,
            retryable_exceptions=(TimeoutError,),
        )
    )
    async def always_timeout():
        attempts["n"] += 1
        raise TimeoutError("always fail")

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await always_timeout()
    elapsed = time.monotonic() - start

    # 3 尝试意味着 2 次等待，每次 0.01s（无指数退避，无抖动）
    assert attempts["n"] == 3
    assert elapsed >= 0.015


@pytest.mark.asyncio
async def test_get_weather_retries_then_succeeds():
    m = _load_module()
    m.ATTEMPT_COUNTER = 0

    start = time.monotonic()
    result = await m.get_weather({"city": "上海"})
    elapsed = time.monotonic() - start

    assert "上海" in result["result"]
    assert result["attempt"] == 2
    # 第一次调用超时约 5s + 重试等待约 1.6~2.4s + 第二次成功约 4s
    assert elapsed >= 9.0


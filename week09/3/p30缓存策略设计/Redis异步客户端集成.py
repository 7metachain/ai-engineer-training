import asyncio
import redis.asyncio as aioredis  # 兼容 Python 3.12/3.13，避免老版 aioredis 导入问题
import pickle
import json
from typing import Any, Optional, Union, Callable, List, Dict
from functools import wraps
import hashlib
from datetime import timedelta
import logging
import os
import threading
import time
import random
from fastapi import FastAPI, Body
from contextlib import asynccontextmanager
import httpx

# 初始化 FastAPI 应用与日志输出
# - 提供示例端点与生命周期钩子
# - 配置基础日志级别便于观察缓存命中/设置情况
app = FastAPI()
logging.basicConfig(level=logging.INFO)
app_logger = logging.getLogger("cache-demo")

class RedisCache:
    """Redis异步缓存"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.default_ttl = 300  # 5分钟默认过期时间
    
    async def connect(self):
        """连接Redis（幂等）"""
        # 已连接时直接返回，避免重复建立连接
        if self.redis:
            return
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # 保持bytes类型以便存储任意数据
                max_connections=20,
                socket_connect_timeout=5,
                socket_timeout=30,
            )
            # 测试连接
            await self.redis.ping()
            logging.info("Redis连接成功")
        except Exception as e:
            logging.error(f"Redis连接失败: {e}")
            raise
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存（加入 TTL 抖动，缓解缓存雪崩）"""
        try:
            # 序列化值
            serialized_value = self._serialize(value)
            base_ttl = ttl or self.default_ttl
            # 当 TTL 足够大时引入 ±10% 抖动，避免同时过期导致雪崩
            jitter = random.randint(-int(base_ttl * 0.1), int(base_ttl * 0.1)) if base_ttl >= 10 else 0
            actual_ttl = max(1, base_ttl + jitter)

            await self.redis.setex(key, actual_ttl, serialized_value)
        except Exception as e:
            logging.error(f"缓存设置失败 {key}: {e}")
            # 失败时不抛出异常，保证业务继续
            pass
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            value = await self.redis.get(key)
            if value is not None:
                return self._deserialize(value)
            return None
        except Exception as e:
            logging.error(f"缓存获取失败 {key}: {e}")
            return None

    async def get_many(self, keys: List[str]) -> List[Optional[Any]]:
        """批量获取缓存，减少网络往返次数"""
        try:
            raw_values = await self.redis.mget(keys)
            return [self._deserialize(v) if v is not None else None for v in raw_values]
        except Exception as e:
            logging.error(f"批量缓存获取失败 {keys}: {e}")
            return [None] * len(keys)

    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None):
        """批量设置缓存（带 TTL 抖动）"""
        try:
            base_ttl = ttl or self.default_ttl
            jitter = random.randint(-int(base_ttl * 0.1), int(base_ttl * 0.1)) if base_ttl >= 10 else 0
            actual_ttl = max(1, base_ttl + jitter)
            pipe = self.redis.pipeline(transaction=True)
            for k, v in mapping.items():
                pipe.setex(k, actual_ttl, self._serialize(v))
            await pipe.execute()
        except Exception as e:
            logging.error(f"批量缓存设置失败: {e}")
            pass
    
    async def delete(self, key: str):
        """删除缓存"""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logging.error(f"缓存删除失败 {key}: {e}")
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logging.error(f"缓存检查失败 {key}: {e}")
            return False
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """原子递增"""
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logging.error(f"缓存递增失败 {key}: {e}")
            return 0

    async def close(self):
        """关闭 Redis 连接（幂等）"""
        try:
            if self.redis:
                await self.redis.close()
                self.redis = None
        except Exception as e:
            logging.error(f"关闭 Redis 连接失败: {e}")
    
    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        if isinstance(value, (str, int, float)):
            return str(value).encode('utf-8')
        elif isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False).encode('utf-8')
        else:
            # 使用pickle处理复杂对象
            return pickle.dumps(value)
    
    def _deserialize(self, value: bytes) -> Any:
        """反序列化值"""
        try:
            # 尝试JSON反序列化
            return json.loads(value.decode('utf-8'))
        except json.JSONDecodeError:
            try:
                # 尝试字符串解码
                return value.decode('utf-8')
            except UnicodeDecodeError:
                # 使用pickle反序列化
                return pickle.loads(value)

# 缓存装饰器
def cache(ttl: int = 300, key_prefix: str = "", exclude_params: list = None):
    """缓存装饰器（支持旁路与分布式锁防击穿）
    
    - 通过 `_cache_bypass=True` 可跳过缓存（灰度/调试场景）
    - 使用分布式锁在高并发场景进行二次检查，缓解缓存击穿
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 确保 Redis 连接可用
            if cache_instance.redis is None:
                await cache_instance.connect()

            # 旁路控制：kwargs 传入 `_cache_bypass=True` 时跳过缓存
            bypass = bool(kwargs.pop("_cache_bypass", False))

            # 生成缓存键（支持排除某些不参与键生成的参数）
            cache_key = generate_cache_key(func, args, kwargs, key_prefix, exclude_params)
            
            # 非旁路时先尝试命中缓存
            if not bypass:
                cached_result = await cache_instance.get(cache_key)
                if cached_result is not None:
                    logging.info(f"缓存命中: {cache_key}")
                    return cached_result

            # 分布式锁：避免同一键在高并发下重复计算
            lock = DistributedLock(cache_instance, lock_timeout=max(5, int(ttl * 0.3)))
            acquired = await lock.acquire(cache_key, timeout=3)
            try:
                if acquired:
                    # 二次检查：获锁后再查一次，避免N+1计算
                    if not bypass:
                        cached_again = await cache_instance.get(cache_key)
                        if cached_again is not None:
                            logging.info(f"缓存命中(二次检查): {cache_key}")
                            return cached_again
                    
                    # 执行被装饰的函数
                    result = await func(*args, **kwargs)
                    # 写入缓存（带TTL抖动）
                    await cache_instance.set(cache_key, result, ttl)
                    logging.info(f"缓存设置: {cache_key}")
                    return result
                else:
                    # 无法拿到锁时短暂等待，期望其他工作者写入缓存
                    for _ in range(20):  # 最多等待约2秒
                        await asyncio.sleep(0.1)
                        cached_result = await cache_instance.get(cache_key)
                        if cached_result is not None:
                            logging.info(f"缓存命中(等待后): {cache_key}")
                            return cached_result
                    # 仍未命中则自行计算并写入缓存
                    result = await func(*args, **kwargs)
                    await cache_instance.set(cache_key, result, ttl)
                    logging.info(f"缓存设置(无锁回退): {cache_key}")
                    return result
            finally:
                if acquired:
                    # 仅持有锁的工作者释放锁
                    await lock.release(cache_key)
        
        return wrapper
    return decorator

def generate_cache_key(func: Callable, args, kwargs, prefix: str, exclude_params: list = None) -> str:
    """生成缓存键"""
    # 排除不需要参与缓存键生成的参数
    filtered_kwargs = kwargs.copy()
    if exclude_params:
        for param in exclude_params:
            filtered_kwargs.pop(param, None)
    
    # 生成唯一标识
    key_data = {
        'func': f"{func.__module__}.{func.__name__}",
        'args': args,
        'kwargs': filtered_kwargs
    }
    
    # 使用hashlib生成固定长度的键
    key_str = f"{prefix}:{json.dumps(key_data, sort_keys=True, default=str)}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

# 缓存穿透防护
class CachePenetrationProtection:
    """缓存穿透防护"""
    
    def __init__(self, cache: RedisCache, null_ttl: int = 60):
        self.cache = cache
        self.null_ttl = null_ttl  # 空值缓存时间
    
    async def get_or_fetch(self, key: str, fetch_func: Callable, ttl: int = 300):
        """获取或获取并缓存，防止缓存穿透"""
        # 先尝试从缓存获取
        result = await self.cache.get(key)
        if result is not None:
            return result
        
        # 检查空值标记
        null_key = f"{key}:null"
        if await self.cache.exists(null_key):
            logging.info(f"空值标记命中: {key}")
            return None
        
        # 获取数据
        result = await fetch_func()
        
        if result is not None:
            # 缓存正常结果
            await self.cache.set(key, result, ttl)
        else:
            # 缓存空值标记，防止穿透
            await self.cache.set(null_key, "1", self.null_ttl)
        
        return result

# 分布式锁
class DistributedLock:
    """分布式锁"""
    
    def __init__(self, cache: RedisCache, lock_timeout: int = 30):
        self.cache = cache
        self.lock_timeout = lock_timeout
    
    async def acquire(self, resource: str, timeout: int = 10) -> bool:
        """获取锁"""
        lock_key = f"lock:{resource}"
        identifier = f"{os.getpid()}:{threading.current_thread().ident}"
        
        end_time = time.time() + timeout
        while time.time() < end_time:
            # 使用SETNX获取锁
            acquired = await self.cache.redis.set(lock_key, identifier, nx=True, ex=self.lock_timeout)
            if acquired:
                return True
            await asyncio.sleep(0.1)
        
        return False
    
    async def release(self, resource: str):
        """释放锁"""
        lock_key = f"lock:{resource}"
        # 使用Lua脚本确保原子性
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        identifier = f"{os.getpid()}:{threading.current_thread().ident}"
        # redis-py 的 eval 签名为 eval(script, numkeys, *keys_and_args)
        await self.cache.redis.eval(lua_script, 1, lock_key, identifier)

# 全局缓存实例
cache_instance = RedisCache()

# 使用 FastAPI lifespan 处理连接的建立与关闭，避免 on_event 的弃用警告
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时连接 Redis，关闭时释放连接"""
    await cache_instance.connect()
    try:
        yield
    finally:
        await cache_instance.close()

# 安装 lifespan 上下文（可在应用创建后设置）
app.router.lifespan_context = lifespan

# 缓存使用示例
@cache(ttl=600, key_prefix="predictions", exclude_params=["sleep_ms"])  # 排除 sleep_ms，不让它影响缓存键
async def get_prediction(model_name: str, features: list, sleep_ms: int = 1000):
    """带缓存的预测接口
    
    - 通过 sleep_ms 控制模拟计算耗时，首次请求更易观察缓存效果
    """
    # 模拟耗时的预测计算（毫秒）
    await asyncio.sleep(max(0, sleep_ms) / 1000.0)
    return {
        "prediction": random.uniform(0, 1),
        "model": model_name,
        "features_count": len(features)
    }

@app.get("/cached-prediction/{model_name}")
async def cached_prediction(model_name: str, features: str, bypass: bool = False, sleep_ms: int = 1000):
    """使用缓存的预测端点（支持旁路）
    
    - 通过查询参数 `bypass=true` 强制跳过缓存，适合调试或灰度测试
    """
    feature_list = json.loads(features)
    # 生成与装饰器一致的缓存键（排除 sleep_ms）以便判断是否已有缓存
    cache_key = generate_cache_key(
        get_prediction, (model_name, feature_list), {"sleep_ms": sleep_ms}, "predictions", exclude_params=["sleep_ms"]
    )
    pre_exists = await cache_instance.exists(cache_key)

    t0 = time.perf_counter()
    result = await get_prediction(model_name, feature_list, sleep_ms=sleep_ms, _cache_bypass=bypass)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    from_cache = (not bypass) and pre_exists
    # 打印可观测日志
    logging.info(f"[cached-prediction] from_cache={from_cache} elapsed_ms={elapsed_ms:.1f} key={cache_key}")

    return {
        "data": result,
        "from_cache": from_cache,
        "elapsed_ms": round(elapsed_ms, 1),
        "cache_key": cache_key,
    }

# =====================
# 大模型返回内容缓存示例
# =====================

@cache(ttl=300, key_prefix="llm_response")
async def generate_llm_response(
    model_name: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
):
    """调用 Qwen Turbo（DashScope 兼容模式）并缓存响应内容
    
    - 参考 websocket 与大模型示例：使用 httpx.AsyncClient 调用兼容模式 Chat Completions
    - 通过 `DASHSCOPE_API_KEY` 环境变量提供认证；`DASHSCOPE_COMPAT_URL` 可覆盖默认端点
    - 使用非流式调用（`stream=False`）以获取完整响应并便于做缓存
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        # 无 API Key 时返回错误信息（仍可缓存错误，避免穿透）
        return {
            "error": "未设置环境变量 DASHSCOPE_API_KEY",
            "model": model_name,
            "prompt": prompt,
        }

    url = os.getenv(
        "DASHSCOPE_COMPAT_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name or "qwen-turbo",
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    try:
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                # 读取错误信息并返回
                try:
                    msg = resp.text
                except Exception:
                    msg = ""
                return {
                    "error": f"HTTP {resp.status_code}",
                    "message": msg,
                    "model": model_name,
                    "prompt": prompt,
                }

            obj = resp.json()
            # 兼容 OpenAI 风格的返回结构
            choices = obj.get("choices") or []
            content = ""
            if choices:
                message = choices[0].get("message") or {}
                delta = choices[0].get("delta") or {}
                content = (
                    message.get("content")
                    or delta.get("content")
                    or ""
                )

            return {
                "model": payload["model"],
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "output": content,
                "raw": obj,
            }
    except httpx.RequestError as e:
        return {"error": f"网络请求失败 {e}", "model": model_name, "prompt": prompt}
    except Exception as e:
        return {"error": str(e), "model": model_name, "prompt": prompt}

@app.post("/cached-llm/generate")
async def cached_llm_generate(
    payload: Dict[str, Any] = Body(..., description="LLM请求体：model_name, prompt, system, temperature, top_p, max_tokens"),
    bypass: bool = False,
):
    """POST 端点：为大模型生成结果增加缓存（支持旁路）
    
    - 请求体示例：
      {
        "model_name": "gpt-4o-mini",
        "prompt": "请用一句话解释缓存击穿",
        "system": "你是资深工程师",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 128
      }
    - 查询参数 `bypass=true` 可绕过缓存
    """
    model_name = str(payload.get("model_name", ""))
    prompt = str(payload.get("prompt", ""))
    system = str(payload.get("system", ""))
    temperature = float(payload.get("temperature", 0.7))
    top_p = float(payload.get("top_p", 1.0))
    max_tokens = int(payload.get("max_tokens", 512))

    if not model_name or not prompt:
        return {"error": "model_name 与 prompt 为必填项"}
    # 生成与装饰器一致的缓存键（使用位置参数，与装饰器内一致）
    cache_key = generate_cache_key(
        generate_llm_response,
        (model_name, prompt, system, temperature, top_p, max_tokens),
        {},
        "llm_response",
    )
    pre_exists = await cache_instance.exists(cache_key)

    t0 = time.perf_counter()
    result = await generate_llm_response(
        model_name,
        prompt,
        system,
        temperature,
        top_p,
        max_tokens,
        _cache_bypass=bypass,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    from_cache = (not bypass) and pre_exists
    logging.info(f"[cached-llm] from_cache={from_cache} elapsed_ms={elapsed_ms:.1f} key={cache_key}")

    return {
        "data": result,
        "from_cache": from_cache,
        "elapsed_ms": round(elapsed_ms, 1),
        "cache_key": cache_key,
    }

if __name__ == "__main__":
    import uvicorn
    # 注意：当启用 reload 或 workers 时，需要以导入字符串形式传入应用
    # 这里使用当前工作目录下的模块文件名作为导入路径
    uvicorn.run("Redis异步客户端集成:app", host="0.0.0.0", port=8000, reload=True)

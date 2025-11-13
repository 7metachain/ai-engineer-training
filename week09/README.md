# Python高性能编程与并发工具

本目录汇集了第九周“Python 高性能并发”课程的全部示例，涵盖 **协程语法、事件循环深度解析、Future/Task 调度、多线程/多进程混合、网络 IO 工具链、LangChain/LangGraph 异步实践、性能剖析与压测、GPU/向量检索** 等主题。脚本既可以在课堂上逐段讲解，也可作为你在真实项目中排查性能瓶颈、设计高吞吐架构的参考。

## 目录速览
- `p5_底层IO多路复用过程.py` ～ `p8_2新风格.py`：从 selector 原理、`async/await` 语法糖到旧/新协程风格迁移的最小可运行示例。
- `p11_*.py` / `p12_*` / `p13_aiohttp.py` / `p14_concurrent.py`：日志调试、慢速回调定位、Future/Task/Executor 用法、`aiohttp` 客户端与 `concurrent.futures` 组合策略。
- `p16_*` / `p17_*`：GIL 影响、真实 IO/同步/异步/线程/进程对比实验，以及 `requests`、`psutil` 观测数据的实战脚本。
- `p18_*` / `p19_1上下文测量.py` / `p20_*`：IO 密集压测、负载压力脚本、上下文性能计量、`cProfile`、`py-spy` 的集成示例。
- `p21_多进程与协程混合/`：scheduler、processors、strategies、factories 全套示例，讲解如何在多进程 worker 与协程任务间编排工作流。
- `3/` 目录：FastAPI + WebSocket 服务、Pydantic 模型、限流中间件、asyncpg/Redis/缓存策略等“高并发 API 实战”模板。
- `4/` 目录：LangChain 回调追踪、LangGraph 异步执行、超时/重试测试案例（含 `pytest`）。
- `5/p36CUDA与异步GPU.py`：使用 `numpy` + `faiss` 的异步 GPU/CPU 向量检索示例，演示线程池 + 协程串联。

## 环境准备
### 基础依赖
- Python 3.11+（建议 `uv` 或 `venv` 管理虚拟环境）
- C 编译工具链 & `faiss-cpu` 对应的系统依赖（若需要 GPU，可改装 `faiss-gpu`）
- Redis / PostgreSQL / DashScope API / OpenAI 兼容接口（按需启用相应脚本）
- 可选：`pytest`, `psutil`, `py-spy`, `asyncpg`, `httpx`, `langchain*`, `langgraph`

常用环境变量：
- `DASHSCOPE_API_KEY`, `DASHSCOPE_COMPAT_URL`, `DASHSCOPE_MODEL`
- `WS_PORT`, `REDIS_URL`, `DATABASE_URL`（asyncpg/Redis 场景按需设置）

### 安装建议
```bash
# 进入项目根目录
cd week09

# 同步 pyproject 依赖（推荐）
uv sync --locked

# 或使用 pip
pip install -e .
```

> 首次添加/调整依赖后执行 `uv pip compile` 或 `uv lock` 以生成锁文件，再 `uv sync` 保持环境一致。

## 教学模块概览
### 1. 协程语法与事件循环剖析
- `p5_底层IO多路复用过程.py` 到 `p8_2新风格.py` 逐步展示 selector、`await` 展开、旧式协程 (`yield from`) 迁移，为理解事件循环提供可视化打印。
- `p7_await方法.py` 演示 `await` 如何针对不同 Awaitable 触发特定协议方法。

### 2. Future/Task/Executor 调试套路
- `p11_1日志调试代码.py`、`p11_2慢速回调调试代码.py`：提供结构化日志与慢回调定位技巧。
- `p12_1Future.py` ～ `p12_3Executor.py`：比较 `Future`、`Task`、执行器线程池/进程池的差异；配合 `p13_aiohttp.py`、`p14_concurrent.py` 说明如何把阻塞 IO 包装进协程世界。

### 3. 多线程/多进程/协程混合
- `p16_*`、`p17_*`：通过 `numpy`、`requests`、`psutil`、`asyncio`，构建 GIL 复现、同步/异步吞吐对比、线程 vs 进程 vs 协程性能基线。
- `p21_多进程与协程混合/` 目录提供 scheduler、processors、strategies、factories、`run_demo.py`，展示任务分发、子进程生命周期管理、异步回传等高级模式。

### 4. 高并发 API 与中间件实战（`3/`）
- `3/p24RESTful API设计规范与最佳实践/restful_api设计原则演示.py`、`3/p26WebSocket与大模型/websocket与大模型.py`：FastAPI + WebSocket + SSE 流式代理，配合 `httpx`, `uvicorn`.
- `3/p29限流中间件实现/限流中间件.py`、`3/p30缓存策略设计/Redis异步客户端集成.py`：演示 `redis`, `httpx`, `starlette` 中间件、缓存与限流策略。
- `3/p25使用Pydantic进行复杂数据校验与模型转换/pydantic1.py`：结合 `pydantic`, `pydantic-settings`, `openai`（DashScope 兼容），构建结构化输出校验。
- `3/p28数据库连接池配置/asyncpg异步数据库.py`：`asyncpg` + `SQLAlchemy` 双栈数据库访问示例。

### 5. LangChain / LangGraph 异步工作流（`4/`）
- `4/p32langchain调用/异步API.py`、`4/p33自定义回调处理器/回调过程跟踪_qwen.py`：自定义 `AsyncCallbackHandler`、WebSocket 进度跟踪、`langchain-core`/`langchain-community` 模型封装。
- `4/p34常见异步陷阱及规避/*.py`：针对 LangGraph 异步执行提供重试、超时、流水线调用示例，并通过 `test_retry.py`（`pytest`）验证容错逻辑。

### 6. 性能观测与压测
- `p18_1IO密集场景综合性能测试.py`、`p18_2负载压力测试.py`：封装 `aiohttp`, `psutil`, `time`，用于构造 IO 压测与系统负载采样。
- `p19_1上下文测量.py`：说明在上下文管理器中采集耗时/统计。
- `p20_1cProfile.py`, `p20_2PySpy.py`: 展示如何脚本化地启动 `cProfile`、`py-spy` 以获取火焰图（`profile.svg` 提供示例输出）。

### 7. GPU/向量检索与异步并行
- `5/p36CUDA与异步GPU.py`：结合 `numpy`, `faiss`, `asyncio`, `ThreadPoolExecutor` 构建 GPU/CPU 自适应索引，并通过 `async_search` 演示在 IO 循环中安全调用阻塞向量检索。

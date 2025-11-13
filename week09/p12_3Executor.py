import asyncio
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

# I/O密集型 - asyncio原生异步
async def io_task(task_id):
    await asyncio.sleep(1)  # 模拟I/O等待
    return f"I/O任务{task_id}完成"

# CPU密集型 - 多进程并行
def cpu_task(task_id):
    total = sum(i*i for i in range(1000000))  # CPU计算
    return f"CPU任务{task_id}完成"

async def main():
    print("=== Executor核心机制演示 ===")
    
    # I/O密集型：asyncio并发
    print("\n1. I/O密集型 (asyncio):")
    start = time.time()
    io_results = await asyncio.gather(*[io_task(i) for i in range(5)])
    io_time = time.time() - start
    print(f"5个I/O任务并发执行: {io_time:.2f}秒")
    
    # CPU密集型：ProcessPool并行
    print("\n2. CPU密集型 (ProcessPoolExecutor):")
    start = time.time()
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        loop = asyncio.get_event_loop()
        cpu_results = await asyncio.gather(*[
            loop.run_in_executor(pool, cpu_task, i) for i in range(5)
        ])
    cpu_time = time.time() - start
    print(f"5个CPU任务并行执行: {cpu_time:.2f}秒")
    
    print(f"\n核心原理:")
    print(f"- I/O密集型用asyncio: 单线程并发，避免阻塞")
    print(f"- CPU密集型用ProcessPool: 多进程并行，绕过GIL")

if __name__ == "__main__":
    asyncio.run(main())
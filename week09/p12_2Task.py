import asyncio
import time

async def simple_coroutine():
    """简单的协程函数"""
    print(" 协程开始执行")
    await asyncio.sleep(2)  # 模拟耗时操作
    print(" 协程执行完成")
    return "Hello from Task!"

async def demonstrate_create_task():
    """演示 asyncio.create_task() 的基本用法"""
    
    # 方法1: 直接创建任务
    print(" 方法1: 直接创建任务")
    task1 = asyncio.create_task(simple_coroutine())
    print(f"任务已创建: {task1}")
    print(f"任务是否完成: {task1.done()}")
    
    # 等待任务完成并获取结果
    result1 = await task1
    print(f"任务结果: {result1}")
    print(f"任务是否完成: {task1.done()}\n")
    
    # 方法2: 创建多个任务并发执行
    print(" 方法2: 创建多个任务并发执行")
    
    async def quick_task(name, delay):
        print(f"任务 {name} 开始执行")
        await asyncio.sleep(delay)
        print(f"任务 {name} 完成")
        return f"结果-{name}"
    
    # 同时创建多个任务
    task_a = asyncio.create_task(quick_task("A", 1))
    task_b = asyncio.create_task(quick_task("B", 2))
    task_c = asyncio.create_task(quick_task("C", 1.5))
    
    print("所有任务已创建，开始并发执行...")
    
    # 等待所有任务完成
    results = await asyncio.gather(task_a, task_b, task_c)
    print(f"所有任务结果: {results}\n")

async def main():
    await demonstrate_create_task()

if __name__ == "__main__":
    # 运行异步主函数
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"\n总执行时间: {end_time - start_time:.2f} 秒")
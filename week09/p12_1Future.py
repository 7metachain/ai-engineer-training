import asyncio
import time
from concurrent.futures import Future as SyncFuture


def callback_example(future):
    """Future完成时的回调函数示例"""
    try:
        result = future.result()
        print(f"回调函数收到结果: {result}")
    except asyncio.CancelledError:
        print("回调函数: Future已被取消")
    except Exception as e:
        print(f"回调函数捕获到异常: {e}")


async def basic_future_example():
    """基本Future使用示例"""
    print("=== 基本Future使用示例 ===")
    loop = asyncio.get_event_loop()
    
    # 创建Future并注册回调函数
    future = loop.create_future()
    future.add_done_callback(callback_example)
    
    def on_data_received(data):
        future.set_result(f"接收到数据: {data}")
    
    # 模拟异步数据接收
    loop.call_later(0.1, on_data_received, "Hello World")
    
    result = await future
    print(f"主程序收到结果: {result}")
    print()


async def concurrent_operations_example():
    """并发操作示例"""
    print("=== 并发操作示例 ===")
    loop = asyncio.get_event_loop()
    
    # 创建多个Future
    futures = [loop.create_future() for _ in range(3)]
    
    # 为每个Future添加回调
    for i, future in enumerate(futures):
        def make_callback(index):
            def callback(f):
                print(f"任务 {index} 完成: {f.result()}")
            return callback
        future.add_done_callback(make_callback(i))
    
    # 模拟异步完成
    def complete_task(index, value):
        futures[index].set_result(f"任务{index}的结果: {value}")
    
    # 安排任务完成（缩短时间）
    loop.call_later(0.1, complete_task, 0, "A")
    loop.call_later(0.2, complete_task, 1, "B")
    loop.call_later(0.15, complete_task, 2, "C")
    
    # 等待所有Future完成
    results = await asyncio.gather(*futures)
    print(f"所有任务完成，结果: {results}")
    print()

# 测试所有示例
async def main():
    """主函数：运行Future示例"""
    
    await basic_future_example()
    await concurrent_operations_example()

if __name__ == "__main__":
    asyncio.run(main())
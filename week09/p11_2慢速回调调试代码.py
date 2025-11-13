import asyncio
import time

# 简单的异步任务
async def demo_task():
    print("任务开始执行")
    await asyncio.sleep(0.1)  # 模拟异步操作
    print("任务执行完成")
    return "任务结果"

# 包含慢速回调的任务
async def task_with_slow_callback():
    loop = asyncio.get_running_loop()
    
    # 设置慢速回调的阈值为0.1秒
    loop.slow_callback_duration = 0.1
    
    def slow_callback():
        """这是一个会触发调试警告的慢速回调"""
        print("慢速回调开始")
        time.sleep(0.2)  # 故意阻塞超过阈值的时间
        print("慢速回调结束")
    
    # 注册回调
    loop.call_soon(slow_callback)
    
    # 等待回调执行完成
    await asyncio.sleep(0.3)
    
    return "包含慢速回调的任务完成"

# 演示普通模式
def run_without_debug():
    print("=== 普通模式 ===")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 不启用调试模式
    # loop.set_debug(False)  # 默认就是False
    
    try:
        result = loop.run_until_complete(task_with_slow_callback())
        print(f"结果: {result}")
    finally:
        loop.close()

# 演示调试模式
def run_with_debug():
    print("\n=== 调试模式 ===")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 启用调试模式 - 这会显示额外的调试信息
    loop.set_debug(True)
    
    try:
        result = loop.run_until_complete(task_with_slow_callback())
        print(f"结果: {result}")
    finally:
        loop.close()

# 主函数
if __name__ == "__main__":
    print("演示loop.set_debug(True)的调试信息差异：")
    print("=" * 50)
    
    # 先运行普通模式
    run_without_debug()
    
    print("")
    
    # 再运行调试模式
    run_with_debug()
    
    print("\n" + "=" * 50)
    print("注意观察调试模式下是否会显示关于慢速回调的警告信息")
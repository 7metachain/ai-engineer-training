import asyncio
import aiohttp
import threading
import time
import os
from typing import Dict, Any

class PySpyIntegration:
    """Py-Spy集成与使用指南"""
    
    @staticmethod
    def get_pyspy_commands():
        """获取常用的Py-Spy命令"""
        commands = {
            "实时监控": "py-spy top --pid <PID>",
            "生成火焰图": "py-spy record -o profile.svg --pid <PID>",
            "分析特定时间段": "py-spy record -d 30 -o profile.json --pid <PID>",
            "过滤特定函数": "py-spy top --pid <PID> --function '.*async.*'",
            "监控子进程": "py-spy top --pid <PID> --subprocesses"
        }
        return commands
    
    async def create_cpu_intensive_task(self):
        """创建CPU密集型任务用于分析"""
        print("开始CPU密集型任务...")
        
        # 模拟复杂计算
        for _ in range(10000):
            # 复杂数学运算
            result = 0
            for i in range(1000):
                result += i ** 2 + i ** 0.5
            await asyncio.sleep(0.01)  # 让出控制权
        
        print("CPU密集型任务完成")
    
    async def create_io_bottleneck(self):
        """创建I/O瓶颈用于分析"""
        print("开始I/O密集型任务...")
        
        connector = aiohttp.TCPConnector(limit=10)  # 限制连接数制造瓶颈
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i in range(50):
                task = asyncio.create_task(
                    session.get(f"https://httpbin.org/delay/2")
                )
                tasks.append(task)
                
                if len(tasks) % 10 == 0:
                    # 批量等待减少内存占用
                    await asyncio.gather(*tasks, return_exceptions=True)
                    tasks = []
                    await asyncio.sleep(0.1)
        
        print("I/O密集型任务完成")

# Py-Spy使用场景演示
async def demonstrate_pyspy_scenarios():
    """演示Py-Spy的使用场景"""
    
    pyspy = PySpyIntegration()
    
    # 场景1: CPU使用率过高
    print("场景1: CPU使用率过高分析")
    cpu_task = asyncio.create_task(pyspy.create_cpu_intensive_task())
    await asyncio.sleep(1)  # 给Py-Spy留出时间
    
    # 此时可以运行: py-spy top --pid <current_pid>
    await cpu_task
    
    # 场景2: I/O瓶颈
    print("\n场景2: I/O瓶颈分析")
    io_task = asyncio.create_task(pyspy.create_io_bottleneck())
    await asyncio.sleep(1)
    
    # 此时可以运行: py-spy record -o io_bottleneck.svg --pid <current_pid> -d 30
    await io_task

# 主函数，启动异步事件循环
if __name__ == "__main__":
    # 显示当前进程ID
    current_pid = os.getpid()
    print(f"启动Py-Spy集成演示程序...")
    print(f"当前进程ID: {current_pid}")
    print(f"注意: 运行过程中可以使用Py-Spy工具进行性能分析，例如: py-spy top --pid {current_pid}")
    
    # 获取并打印常用Py-Spy命令
    pyspy = PySpyIntegration()
    commands = pyspy.get_pyspy_commands()
    print("\n常用Py-Spy命令:")
    for desc, cmd in commands.items():
        print(f"- {desc}: {cmd}")
    
    # 运行演示场景
    asyncio.run(demonstrate_pyspy_scenarios())
    print("\n演示完成!")


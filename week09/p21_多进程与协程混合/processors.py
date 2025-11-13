"""任务处理器模块"""

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from strategies import IOStrategy
from utils import _process_cpu_task


class TaskProcessor(ABC):
    """
    任务处理器的抽象基类
    
    定义了所有任务处理器必须实现的接口，是处理器策略模式的基础。
    """
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个任务
        
        Args:
            task: 任务字典，包含任务类型、操作和数据等信息
            
        Returns:
            Dict[str, Any]: 任务处理结果，包含状态、结果和耗时等信息
        """
        pass


class IOProcessor(TaskProcessor):
    """
    IO任务处理器，直接在事件循环中处理IO密集型任务
    
    使用异步方式处理HTTP请求、文件读写等IO操作，充分利用事件循环的非阻塞特性。
    """
    
    def __init__(self):
        """初始化IO任务处理器"""
        self.strategy = IOStrategy()
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用异步策略处理IO任务
        
        Args:
            task: 任务字典
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            return await self.strategy.process(task)
        except Exception as e:
            # 捕获并记录处理过程中的异常
            task_id = task.get('id', 'unknown')
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': f"IO任务处理异常: {str(e)}",
                'original_error_type': type(e).__name__
            }


class CPUProcessor(TaskProcessor):
    """
    CPU任务处理器，在进程池中处理CPU密集型任务
    
    将计算密集型任务提交到进程池执行，避免阻塞事件循环，充分利用多核CPU资源。
    """
    
    def __init__(self, executor: concurrent.futures.ProcessPoolExecutor):
        """
        初始化CPU任务处理器
        
        Args:
            executor: 进程池执行器，用于提交CPU密集型任务
        """
        self.executor = executor
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        在进程池中异步执行CPU密集型任务
        
        Args:
            task: 任务字典，包含操作类型和数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        loop = asyncio.get_running_loop()
        try:
            # 在进程池中执行CPU密集型任务
            return await loop.run_in_executor(
                self.executor,
                _process_cpu_task,
                task
            )
        except Exception as e:
            # 捕获并记录进程池执行过程中的异常
            task_id = task.get('id', 'unknown')
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': f"CPU任务处理异常: {str(e)}",
                'original_error_type': type(e).__name__
            }
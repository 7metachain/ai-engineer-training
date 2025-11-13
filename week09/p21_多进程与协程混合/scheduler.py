"""任务调度器模块"""

import asyncio
from typing import List, Dict, Any, Optional

from factories import ProcessPoolFactory, ProcessorFactory


class TaskScheduler:
    """
    任务调度器，负责分发和处理任务
    
    该类是多进程与协程混合架构的核心组件，根据任务类型将任务分发到适当的执行器。
    IO密集型任务在事件循环中执行，CPU密集型任务在进程池中执行。
    """
    
    def __init__(self, pool_factory: ProcessPoolFactory):
        """
        初始化任务调度器
        
        Args:
            pool_factory: 进程池工厂实例，用于创建和管理进程池
        """
        self.pool_factory = pool_factory
        self.executor: Optional[asyncio.AbstractEventLoop] = None
        self._initialized = False
    
    def initialize(self, max_workers: Optional[int] = None):
        """
        初始化调度器和处理器
        
        Args:
            max_workers: 最大工作进程数，默认为CPU核心数
        """
        self.executor = self.pool_factory.create_pool(max_workers=max_workers)
        
        # 动态导入以避免循环导入
        from processors import IOProcessor, CPUProcessor
        
        # 注册处理器
        ProcessorFactory.register_processor('io', IOProcessor)
        ProcessorFactory.register_processor('cpu', CPUProcessor, executor=self.executor)
        
        self._initialized = True
    
    async def schedule_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        调度单个任务到适当的处理器
        
        Args:
            task: 任务字典，包含任务类型、参数等信息
            
        Returns:
            Dict[str, Any]: 任务处理结果
            
        Raises:
            RuntimeError: 如果调度器尚未初始化
        """
        if not self._initialized:
            raise RuntimeError("调度器尚未初始化，请先调用initialize方法")
        
        task_type = task.get('type')
        processor = ProcessorFactory.get_processor(task_type)
        return await processor.process_task(task)
    
    async def schedule_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        并行调度多个任务
        
        Args:
            tasks: 任务列表，每个任务是包含任务信息的字典
            
        Returns:
            List[Dict[str, Any]]: 任务处理结果列表，每个结果对应一个输入任务
        """
        # 并行处理所有任务
        futures = [self.schedule_task(task) for task in tasks]
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 处理可能的异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'task_id': tasks[i].get('id', i),
                    'status': 'error',
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def shutdown(self, wait: bool = True):
        """
        关闭调度器和相关资源
        
        Args:
            wait: 是否等待所有任务完成
        """
        if self._initialized:
            self.pool_factory.shutdown_pool(wait=wait)
            self._initialized = False
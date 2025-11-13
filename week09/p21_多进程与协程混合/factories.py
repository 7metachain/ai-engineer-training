"""工厂类模块，负责创建和管理处理器与进程池"""

import concurrent.futures
import multiprocessing as mp
from typing import Dict, Any, Optional, Tuple, Type
import logging

from processors import TaskProcessor

# 配置日志
logger = logging.getLogger(__name__)


class ProcessorFactory:
    """
    任务处理器工厂类
    
    使用工厂模式和注册机制，根据任务类型动态创建对应的处理器实例。
    支持不同类型的任务处理器注册和获取。
    """
    
    # 类变量，存储已注册的处理器类型
    _processors: Dict[str, Tuple[Type[TaskProcessor], Dict[str, Any]]] = {}
    
    @classmethod
    def register_processor(cls, task_type: str, processor_class: Type[TaskProcessor], **kwargs) -> None:
        """
        注册任务处理器
        
        Args:
            task_type: 任务类型标识符
            processor_class: 处理器类
            **kwargs: 创建处理器时传递的额外参数
        """
        if not issubclass(processor_class, TaskProcessor):
            raise TypeError(f"处理器类必须是TaskProcessor的子类，收到: {processor_class}")
        
        cls._processors[task_type] = (processor_class, kwargs)
        logger.debug(f"已注册任务处理器: {task_type} -> {processor_class.__name__}")
    
    @classmethod
    def get_processor(cls, task_type: str) -> TaskProcessor:
        """
        获取指定类型的任务处理器实例
        
        Args:
            task_type: 任务类型标识符
            
        Returns:
            TaskProcessor: 对应的任务处理器实例
            
        Raises:
            ValueError: 如果指定的任务类型不存在
        """
        if task_type not in cls._processors:
            available_types = ', '.join(cls._processors.keys())
            raise ValueError(
                f"未知任务类型: {task_type}。可用的任务类型有: {available_types}"
            )
        
        processor_class, kwargs = cls._processors[task_type]
        try:
            return processor_class(**kwargs)
        except Exception as e:
            logger.error(f"创建处理器实例失败: {task_type}, 错误: {e}")
            raise RuntimeError(f"创建处理器实例失败: {task_type}") from e
    
    @classmethod
    def get_available_processors(cls) -> Dict[str, Type[TaskProcessor]]:
        """
        获取所有可用的处理器类型
        
        Returns:
            Dict[str, Type[TaskProcessor]]: 任务类型到处理器类的映射
        """
        return {task_type: processor_class for task_type, (processor_class, _) in cls._processors.items()}


class ProcessPoolFactory:
    """
    进程池工厂类
    
    负责创建和管理进程池执行器，提供进程池的生命周期管理。
    确保进程池的正确创建和关闭，避免资源泄漏。
    """
    
    def __init__(self):
        """初始化进程池工厂"""
        self._executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
    
    def create_pool(self, max_workers: Optional[int] = None, start_method: str = 'spawn') -> concurrent.futures.ProcessPoolExecutor:
        """
        创建进程池执行器
        
        Args:
            max_workers: 最大工作进程数，默认使用CPU核心数
            start_method: 进程启动方法，Windows平台推荐使用'spawn'
            
        Returns:
            concurrent.futures.ProcessPoolExecutor: 进程池执行器实例
        """
        if self._executor is not None:
            logger.warning("进程池已存在，返回现有实例")
            return self._executor
        
        try:
            # 设置多进程启动方法
            ctx = mp.get_context(start_method)
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx
            )
            logger.info(f"进程池已创建，最大工作进程数: {self._executor._max_workers}")
            return self._executor
        except Exception as e:
            logger.error(f"创建进程池失败: {e}")
            raise RuntimeError(f"创建进程池失败: {str(e)}") from e
    
    def get_pool(self) -> Optional[concurrent.futures.ProcessPoolExecutor]:
        """
        获取现有的进程池执行器
        
        Returns:
            Optional[concurrent.futures.ProcessPoolExecutor]: 进程池实例，如果不存在则返回None
        """
        return self._executor
    
    def shutdown_pool(self, wait: bool = True) -> None:
        """
        关闭进程池执行器
        
        Args:
            wait: 是否等待所有任务完成后再关闭
        """
        if self._executor:
            try:
                self._executor.shutdown(wait=wait)
                logger.info(f"进程池已关闭，等待任务完成: {wait}")
            except Exception as e:
                logger.error(f"关闭进程池时出错: {e}")
            finally:
                self._executor = None
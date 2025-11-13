"""工厂模式与策略模式混合架构模块"""

from .strategies import TaskProcessingStrategy, IOStrategy, CPUProcessingStrategy
from .factories import ProcessorFactory, ProcessPoolFactory
from .processors import TaskProcessor, IOProcessor, CPUProcessor
from .scheduler import TaskScheduler
from .utils import _process_cpu_task, _fibonacci, _simulate_cpu_work
from .main import main

__all__ = [
    'TaskProcessingStrategy',
    'IOStrategy', 
    'CPUProcessingStrategy',
    'ProcessorFactory',
    'ProcessPoolFactory',
    'TaskProcessor',
    'IOProcessor',
    'CPUProcessor',
    'TaskScheduler',
    'main'
]
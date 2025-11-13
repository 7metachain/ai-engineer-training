"""工具函数模块，包含CPU密集型任务处理和辅助函数"""

import time
from typing import Dict, Any, List, Optional


def _process_cpu_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理CPU密集型任务
    
    在进程池中同步执行，根据任务类型执行不同的计算操作。
    支持斐波那契计算、数据分析和默认计算操作。
    
    Args:
        task: 任务字典，包含操作类型和数据
        
    Returns:
        Dict[str, Any]: 任务处理结果，包含状态、结果、操作类型和耗时等信息
    """
    start_time = time.time()
    task_id = task.get('id', 'unknown')
    
    try:
        # 获取操作类型和数据
        operation = task.get('operation', 'default')
        
        # 验证数据是否存在
        if 'data' not in task:
            raise ValueError("CPU任务缺少数据参数")
        
        data = task['data']
        
        # 根据操作类型执行不同的计算
        if operation == 'fibonacci':
            # 计算斐波那契数列
            n = data
            if not isinstance(n, int) or n < 0:
                raise ValueError(f"斐波那契计算需要非负整数，收到: {n}")
            
            result = _fibonacci(n)
            # 为了让迭代实现的斐波那契任务也有可见延迟，添加一些额外计算
            if n > 45:
                _simulate_cpu_work(1000000)  # 添加一些工作负载
                
        elif operation == 'data_analysis':
            # 数据分析 - 为了产生更明显的延迟，增加计算复杂度
            if not isinstance(data, list):
                raise TypeError(f"数据分析需要列表类型数据，收到: {type(data).__name__}")
            
            result = _perform_data_analysis(data)
            
        else:
            # 默认计算 - 增加一些循环以产生可见延迟
            if not isinstance(data, list):
                raise TypeError(f"默认计算需要列表类型数据，收到: {type(data).__name__}")
            
            result = sum(x**2 for x in data)
            # 对于大数据量，添加一些额外计算来延长处理时间
            if len(data) > 1000000:
                _simulate_cpu_work(500000)
        
        duration = time.time() - start_time
        return {
            'task_id': task_id,
            'status': 'completed',
            'result': result,
            'duration': duration,
            'operation': operation
        }
    
    except Exception as e:
        duration = time.time() - start_time
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(e),
            'error_type': type(e).__name__,
            'duration': duration,
            'operation': task.get('operation', 'unknown')
        }


def _fibonacci(n: int) -> int:
    """
    计算斐波那契数列的第n项（迭代实现）
    
    使用迭代方式计算斐波那契数列，避免递归实现的栈溢出问题。
    
    Args:
        n: 斐波那契数列的索引（从0开始）
        
    Returns:
        int: 斐波那契数列的第n项
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


def _simulate_cpu_work(iterations: int) -> int:
    """
    模拟CPU密集型工作，生成可见的处理时间
    
    执行一些简单但可累积的计算，用于模拟CPU负载。
    
    Args:
        iterations: 迭代次数，决定工作负载大小
        
    Returns:
        int: 计算结果
    """
    result = 0
    for i in range(iterations):
        # 执行一些简单但可累积的计算
        result += (i * i) % 10000
        result = result % 1000000  # 防止数值过大
    return result


def _perform_data_analysis(data: List[int]) -> Dict[str, Any]:
    """
    执行数据分析操作
    
    计算数据的基本统计信息，包括总和、平均值、最大值、最小值等。
    
    Args:
        data: 待分析的数据列表
        
    Returns:
        Dict[str, Any]: 包含分析结果的字典
    """
    count = len(data)
    
    if count == 0:
        return {
            'sum': 0,
            'mean': 0,
            'max': 0,
            'min': 0,
            'count': 0,
            'sum_squares': 0
        }
    
    # 计算基本统计量
    sum_data = sum(data)
    max_val = max(data)
    min_val = min(data)
    
    # 为了增加计算复杂度，只计算部分数据的平方和
    # 避免处理过大的数据量导致过度耗时
    sample_size = min(count, 100000)
    squares = [x*x for x in data[:sample_size]]
    sum_squares = sum(squares)
    
    return {
        'sum': sum_data,
        'mean': sum_data / count,
        'max': max_val,
        'min': min_val,
        'count': count,
        'sum_squares': sum_squares,
        'sample_size': sample_size
    }
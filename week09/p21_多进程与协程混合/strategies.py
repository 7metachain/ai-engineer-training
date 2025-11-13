"""任务处理策略模块"""

import asyncio
import aiohttp
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class TaskProcessingStrategy(ABC):
    """
    任务处理策略的抽象基类
    
    定义了任务处理的通用接口，是策略模式的核心组件。
    不同类型的任务可以实现各自的处理策略。
    """
    
    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理任务的抽象方法
        
        Args:
            task: 任务字典，包含任务相关信息
            
        Returns:
            Dict[str, Any]: 任务处理结果
        """
        pass


class IOStrategy(TaskProcessingStrategy):
    """
    IO密集型任务的处理策略
    
    处理HTTP请求、文件读写等异步IO操作，充分利用异步IO的非阻塞特性。
    支持多种IO类型的处理，包括HTTP请求和文件操作。
    """
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理IO密集型任务
        
        Args:
            task: 任务字典，包含IO操作相关参数
            
        Returns:
            Dict[str, Any]: 任务处理结果，包含状态、数据和耗时等信息
        """
        start_time = time.time()
        task_id = task.get('id', 'unknown')
        
        try:
            # 根据任务类型执行不同的IO操作
            io_type = task.get('io_type')
            
            if io_type == 'http':
                return await self._process_http_request(task, task_id, start_time)
            elif io_type == 'file':
                return await self._process_file_operation(task, task_id, start_time)
            else:
                raise ValueError(f"未知的IO类型: {io_type}")
                
        except Exception as e:
            duration = time.time() - start_time
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'duration': duration
            }
    
    async def _process_http_request(self, task: Dict[str, Any], task_id: str, start_time: float) -> Dict[str, Any]:
        """
        处理HTTP请求
        
        Args:
            task: 任务字典，包含HTTP请求参数
            task_id: 任务ID
            start_time: 任务开始时间
            
        Returns:
            Dict[str, Any]: HTTP请求结果
        """
        url = task.get('url')
        if not url:
            raise ValueError("HTTP任务缺少URL参数")
            
        method = task.get('method', 'GET').upper()
        params = task.get('params', {})
        headers = task.get('headers', {})
        timeout = task.get('timeout', 30)
        
        # 使用连接池优化
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.request(
                method, url, params=params, headers=headers
            ) as response:
                content_type = response.headers.get('Content-Type', '')
                
                # 根据内容类型解析响应
                try:
                    if 'application/json' in content_type:
                        content = await response.json()
                    else:
                        content = await response.text()
                except (ValueError, TypeError):
                    # 处理JSON解析错误
                    content = await response.text()
                
                duration = time.time() - start_time
                return {
                    'task_id': task_id,
                    'status': 'completed',
                    'status_code': response.status,
                    'url': url,
                    'content_length': len(str(content)),
                    'duration': duration,
                    'headers': dict(response.headers)
                }
    
    async def _process_file_operation(self, task: Dict[str, Any], task_id: str, start_time: float) -> Dict[str, Any]:
        """
        处理文件操作
        
        Args:
            task: 任务字典，包含文件操作参数
            task_id: 任务ID
            start_time: 任务开始时间
            
        Returns:
            Dict[str, Any]: 文件操作结果
        """
        file_path = task.get('file_path')
        if not file_path:
            raise ValueError("文件任务缺少文件路径")
            
        operation = task.get('file_operation', 'read')
        
        if operation == 'read':
            # 使用异步文件IO
            loop = asyncio.get_running_loop()
            try:
                content = await loop.run_in_executor(
                    None,  # 使用默认执行器
                    self._read_file_sync, file_path
                )
                
                duration = time.time() - start_time
                return {
                    'task_id': task_id,
                    'status': 'completed',
                    'file_path': file_path,
                    'content_length': len(content),
                    'duration': duration
                }
            except FileNotFoundError:
                raise FileNotFoundError(f"文件不存在: {file_path}")
            except PermissionError:
                raise PermissionError(f"没有权限访问文件: {file_path}")
        
        raise ValueError(f"不支持的文件操作: {operation}")
    
    def _read_file_sync(self, file_path: str) -> str:
        """
        同步读取文件内容（用于在执行器中运行）
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件内容
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


class CPUProcessingStrategy(TaskProcessingStrategy):
    """
    CPU密集型任务的处理策略
    
    注意：这个类主要用于接口一致性，实际的CPU密集型任务处理在utils模块中
    通过进程池同步执行。
    """
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        CPU任务处理的接口方法
        
        Args:
            task: 任务字典
            
        Returns:
            Dict[str, Any]: 任务处理结果
        """
        # 从utils导入以避免循环导入
        from utils import _process_cpu_task
        return _process_cpu_task(task)
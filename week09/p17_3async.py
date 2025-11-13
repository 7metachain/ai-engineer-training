import asyncio
import aiohttp
from typing import List, Dict, Any
from p17_1realIO import RealisticIOLoadGenerator, PerformanceBenchmark

class AsynchronousImplementation:
    """异步执行测试"""
    
    def __init__(self, load_generator: RealisticIOLoadGenerator):
        self.load_generator = load_generator
    
    async def create_session(self) -> aiohttp.ClientSession:
        """创建优化的aiohttp会话"""
        connector = aiohttp.TCPConnector(
            limit=4,  # 统一并行数量为4个
            limit_per_host=4,  # 统一并行数量为4个
            keepalive_timeout=30,
            force_close=False
        )
        
        return aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=10),
            raise_for_status=False
        )
    
    async def run_concurrent_tasks(self, num_tasks: int = 100) -> List[Dict[str, Any]]:  # 统一任务数量为100
        """并发执行异步任务"""
        session = await self.create_session()
        
        tasks = [
            self.load_generator.async_http_request(session, i)
            for i in range(num_tasks)
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = asyncio.get_event_loop().time() - start_time
        
        # 处理异常
        valid_results = []
        latencies = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                valid_results.append({
                    'task_id': i,
                    'error': str(result),
                    'success': False,
                    'duration': 0
                })
            else:
                valid_results.append(result)
                if result['success']:
                    latencies.append(result['duration'])
        
        await session.close()
        return valid_results, total_time, latencies
    
    async def run_streaming_tasks(self, duration: float = 10.0) -> List[Dict[str, Any]]:  # 统一测试时长为10秒
        """流式异步任务执行"""
        session = await self.create_session()
        results = []
        latencies = []
        start_time = asyncio.get_event_loop().time()
        task_id = 0
        
        # 使用信号量控制并发度
        semaphore = asyncio.Semaphore(4)  # 统一并行数量为4个
        
        async def bounded_task(tid: int):
            async with semaphore:
                return await self.load_generator.async_http_request(session, tid)
        
        # 动态创建和等待任务
        running_tasks = set()
        
        while asyncio.get_event_loop().time() - start_time < duration:
            # 创建新任务
            task = asyncio.create_task(bounded_task(task_id))
            running_tasks.add(task)
            task_id += 1
            
            # 清理已完成的任务（非阻塞）
            done_tasks, pending_tasks = await asyncio.wait(
                running_tasks, 
                timeout=0.01,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for done_task in done_tasks:
                try:
                    result = await done_task
                    results.append(result)
                    if result['success']:
                        latencies.append(result['duration'])
                except Exception as e:
                    results.append({
                        'task_id': task_id,
                        'error': str(e),
                        'success': False,
                        'duration': 0
                    })
            
            running_tasks = pending_tasks
            
            # 防止CPU占用过高
            await asyncio.sleep(0.001)
        
        # 等待剩余任务完成
        if running_tasks:
            finished_results = await asyncio.gather(*running_tasks, return_exceptions=True)
            for result in finished_results:
                if isinstance(result, Exception):
                    results.append({
                        'task_id': task_id,
                        'error': str(result),
                        'success': False,
                        'duration': 0
                    })
                elif isinstance(result, dict):
                    results.append(result)
                    if result['success']:
                        latencies.append(result['duration'])
        
        total_time = asyncio.get_event_loop().time() - start_time
        await session.close()
        return results, total_time, latencies

async def main():
    """主异步函数"""
    async_impl = AsynchronousImplementation(RealisticIOLoadGenerator())
    results, total_time, latencies = await async_impl.run_streaming_tasks()
    print(f"Streaming tasks results: {results}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average latency: {sum(latencies) / len(latencies):.4f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
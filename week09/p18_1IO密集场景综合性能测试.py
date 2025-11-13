import asyncio
import time
import concurrent.futures
import multiprocessing as mp
from typing import Dict, Any, List
import statistics

from p17_1realIO import RealisticIOLoadGenerator, PerformanceBenchmark, PerformanceResult

class SynchronousImplementation:
    """同步实现"""
    
    def __init__(self, load_generator: RealisticIOLoadGenerator):
        self.load_generator = load_generator
    
    def run_time_limited(self, duration: float):
        """在指定时间内运行同步测试"""
        results = []
        latencies = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            result = self.load_generator.sync_http_request(len(results))
            results.append(result)
            if result['success']:
                latencies.append(result['duration'])
        
        total_time = time.time() - start_time
        return results, total_time, latencies

class AsynchronousImplementation:
    """异步实现"""
    
    def __init__(self, load_generator: RealisticIOLoadGenerator):
        self.load_generator = load_generator
    
    async def run_streaming_tasks(self, duration: float):
        """在指定时间内运行异步测试"""
        import aiohttp
        results = []
        latencies = []
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            while time.time() - start_time < duration:
                task = self.load_generator.async_http_request(session, len(results))
                tasks.append(task)
                
                # 限制并发数
                if len(tasks) >= 10:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for r in batch_results:
                        if not isinstance(r, Exception):
                            results.append(r)
                            if r['success']:
                                latencies.append(r['duration'])
                    tasks = []
            
            # 处理剩余任务
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in batch_results:
                    if not isinstance(r, Exception):
                        results.append(r)
                        if r['success']:
                            latencies.append(r['duration'])
        
        total_time = time.time() - start_time
        return results, total_time, latencies

class MultithreadingImplementation:
    """多线程实现"""
    
    def __init__(self, load_generator: RealisticIOLoadGenerator, max_workers: int = 50):
        self.load_generator = load_generator
        self.max_workers = max_workers
    
    def run_dynamic_threading(self, duration: float):
        """在指定时间内运行多线程测试"""
        results = []
        latencies = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            while time.time() - start_time < duration:
                future = executor.submit(self.load_generator.thread_http_request, len(futures))
                futures.append(future)
                
                # 避免过多待处理的future
                if len(futures) > 2 * self.max_workers:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            if result['success']:
                                latencies.append(result['duration'])
                        except Exception as e:
                            results.append({
                                'task_id': len(results),
                                'error': str(e),
                                'success': False,
                                'duration': 0
                            })
                    futures = []
            
            # 处理剩余任务
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    if result['success']:
                        latencies.append(result['duration'])
                except Exception as e:
                    results.append({
                        'task_id': len(results),
                        'error': str(e),
                        'success': False,
                        'duration': 0
                    })
        
        total_time = time.time() - start_time
        return results, total_time, latencies

class MultiprocessingImplementation:
    """多进程实现"""
    
    def __init__(self, load_generator: RealisticIOLoadGenerator, max_workers: int = 8):
        self.load_generator = load_generator
        self.max_workers = max_workers
    
    def run_dynamic_processes(self, duration: float):
        """在指定时间内运行多进程测试"""
        results = []
        latencies = []
        start_time = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            while time.time() - start_time < duration:
                future = executor.submit(self.load_generator.process_http_request, len(futures))
                futures.append(future)
                
                # 避免过多待处理的future
                if len(futures) > 2 * self.max_workers:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            if result['success']:
                                latencies.append(result['duration'])
                        except Exception as e:
                            results.append({
                                'task_id': len(results),
                                'error': str(e),
                                'success': False,
                                'duration': 0
                            })
                    futures = []
            
            # 处理剩余任务
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    if result['success']:
                        latencies.append(result['duration'])
                except Exception as e:
                    results.append({
                        'task_id': len(results),
                        'error': str(e),
                        'success': False,
                        'duration': 0
                    })
        
        total_time = time.time() - start_time
        return results, total_time, latencies

class RealWorldPerformanceTest:
    """真实世界性能测试"""
    
    def __init__(self):
        self.load_generator = RealisticIOLoadGenerator()
        self.benchmark = PerformanceBenchmark()
    
    def analyze_results(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """分析测试结果"""
        successful = [r for r in results if r.get('success', False)]
        failed = len(results) - len(successful)
        
        if successful:
            latencies = [r['duration'] for r in successful]
            latency_mean = statistics.mean(latencies)
            latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 0 else 0
        else:
            latencies = []
            latency_mean = 0
            latency_std = 0
            p95_latency = 0
        
        throughput = len(results) / total_time if total_time > 0 else 0
        
        return {
            'total_requests': len(results),
            'successful': len(successful),
            'failed': failed,
            'throughput': throughput,
            'latency_mean': latency_mean,
            'latency_std': latency_std,
            'p95_latency': p95_latency,
            'latencies': latencies
        }
    
    def run_comprehensive_test(self) -> Dict[str, PerformanceResult]:
        """运行完整的性能对比测试"""
        print(f"=== 真实I/O场景性能测试 ===")
        print(f"测试时长: {self.benchmark.test_duration}s")
        print(f"目标URL: {self.load_generator.base_url}")
        
        results = {}
        
        # 1. 同步实现测试
        print("\n--- 同步实现 ---")
        sync_impl = SynchronousImplementation(self.load_generator)
        
        # 时间限制测试
        sync_results, sync_time, sync_latencies = sync_impl.run_time_limited(
            duration=self.benchmark.test_duration
        )
        
        sync_analysis = self.analyze_results(sync_results, sync_time)
        
        results['synchronous'] = PerformanceResult(
            scenario='Real_IO',
            implementation='Synchronous',
            total_time=sync_time,
            throughput=sync_analysis['throughput'],
            latency_mean=sync_analysis['latency_mean'],
            latency_std=sync_analysis['latency_std'],
            memory_usage=self.benchmark.measure_resources(),
            cpu_usage={'peak': 0, 'average': 0},
            details=sync_analysis
        )
        
        print(f"  完成: {sync_analysis['total_requests']} 请求 "
              f"({sync_analysis['successful']} 成功)")
        print(f"  时间: {sync_time:.2f}s, 吞吐量: {sync_analysis['throughput']:.1f} req/s")
        print(f"  平均延迟: {sync_analysis['latency_mean']*1000:.1f}ms")
        
        # 2. 异步实现测试
        print("\n--- 异步实现 ---")
        async_impl = AsynchronousImplementation(self.load_generator)
        
        async def run_async_test():
            async_results, async_time, async_latencies = await async_impl.run_streaming_tasks(
                duration=self.benchmark.test_duration
            )
            return async_results, async_time, async_latencies
        
        try:
            async_start = time.time()
            async_results, async_time, async_latencies = asyncio.run(run_async_test())
            actual_async_time = time.time() - async_start
            
            async_analysis = self.analyze_results(async_results, async_time)
            
            results['asynchronous'] = PerformanceResult(
                scenario='Real_IO',
                implementation='Asynchronous',
                total_time=actual_async_time,
                throughput=async_analysis['throughput'],
                latency_mean=async_analysis['latency_mean'],
                latency_std=async_analysis['latency_std'],
                memory_usage=self.benchmark.measure_resources(),
                cpu_usage={'peak': 0, 'average': 0},
                details=async_analysis
            )
            
            print(f"  完成: {async_analysis['total_requests']} 请求 "
                  f"({async_analysis['successful']} 成功)")
            print(f"  时间: {actual_async_time:.2f}s, 吞吐量: {async_analysis['throughput']:.1f} req/s")
            print(f"  平均延迟: {async_analysis['latency_mean']*1000:.1f}ms")
            
        except Exception as e:
            print(f"  异步测试失败: {e}")
        
        # 3. 多线程实现测试
        print("\n--- 多线程实现 ---")
        thread_impl = MultithreadingImplementation(
            self.load_generator, 
            max_workers=50
        )
        
        thread_results, thread_time, thread_latencies = thread_impl.run_dynamic_threading(
            duration=self.benchmark.test_duration
        )
        
        thread_analysis = self.analyze_results(thread_results, thread_time)
        
        results['multithreading'] = PerformanceResult(
            scenario='Real_IO',
            implementation='Multithreading',
            total_time=thread_time,
            throughput=thread_analysis['throughput'],
            latency_mean=thread_analysis['latency_mean'],
            latency_std=thread_analysis['latency_std'],
            memory_usage=self.benchmark.measure_resources(),
            cpu_usage={'peak': 0, 'average': 0},
            details=thread_analysis
        )
        
        print(f"  完成: {thread_analysis['total_requests']} 请求 "
              f"({thread_analysis['successful']} 成功)")
        print(f"  时间: {thread_time:.2f}s, 吞吐量: {thread_analysis['throughput']:.1f} req/s")
        print(f"  平均延迟: {thread_analysis['latency_mean']*1000:.1f}ms")
        
        # 4. 多进程实现测试
        print("\n--- 多进程实现 ---")
        process_impl = MultiprocessingImplementation(
            self.load_generator, 
            max_workers=8
        )
        
        try:
            process_results, process_time, process_latencies = process_impl.run_dynamic_processes(
                duration=self.benchmark.test_duration
            )
            
            process_analysis = self.analyze_results(process_results, process_time)
            
            results['multiprocessing'] = PerformanceResult(
                scenario='Real_IO',
                implementation='Multiprocessing',
                total_time=process_time,
                throughput=process_analysis['throughput'],
                latency_mean=process_analysis['latency_mean'],
                latency_std=process_analysis['latency_std'],
                memory_usage=self.benchmark.measure_resources(),
                cpu_usage={'peak': 0, 'average': 0},
                details=process_analysis
            )
            
            print(f"  完成: {process_analysis['total_requests']} 请求 "
                  f"({process_analysis['successful']} 成功)")
            print(f"  时间: {process_time:.2f}s, 吞吐量: {process_analysis['throughput']:.1f} req/s")
            print(f"  平均延迟: {process_analysis['latency_mean']*1000:.1f}ms")
            
        except Exception as e:
            print(f"  多进程测试失败: {e}")
        
        return results

if __name__ == "__main__":
    # 运行综合性能测试
    test = RealWorldPerformanceTest()
    test.run_comprehensive_test()
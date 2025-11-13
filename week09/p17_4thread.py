import concurrent.futures
import time
import threading
import multiprocessing as mp
from typing import Dict, List, Any
from p17_1realIO import RealisticIOLoadGenerator, PerformanceBenchmark

class MultithreadingImplementation:
    """多线程执行测试"""
    
    def __init__(self, load_generator: RealisticIOLoadGenerator, max_workers: int = 4):  # 统一并行数量为4个
        self.load_generator = load_generator
        self.max_workers = max_workers
    
    def create_thread_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """创建优化的线程池"""
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="IOThread"
        )
    
    def run_with_thread_pool(self, num_tasks: int = 100) -> List[Dict[str, Any]]:
        """使用线程池执行"""
        with self.create_thread_pool() as executor:
            futures = [
                executor.submit(self.load_generator.thread_http_request, i)
                for i in range(num_tasks)
            ]
            
            start_time = time.time()
            results = []
            latencies = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                    if result['success']:
                        latencies.append(result['duration'])
                except Exception as e:
                    print(f"Thread task failed: {e}")
            
            total_time = time.time() - start_time
            return results, total_time, latencies
    
    def run_dynamic_threading(self, duration: float = 10.0) -> List[Dict[str, Any]]:  # 统一测试时长为10秒
        """动态多线程执行"""
        results = []
        latencies = []
        start_time = time.time()
        task_id = 0
        lock = threading.Lock()
        
        def worker():
            nonlocal task_id, results, latencies
            local_results = []
            local_latencies = []
            
            while time.time() - start_time < duration:
                current_id = task_id
                task_id += 1
                
                result = self.load_generator.thread_http_request(current_id)
                local_results.append(result)
                if result['success']:
                    local_latencies.append(result['duration'])
                
                # 批量写入减少锁竞争
                if len(local_results) >= 20:
                    with lock:
                        results.extend(local_results)
                        latencies.extend(local_latencies)
                    local_results = []
                    local_latencies = []
            
            # 最后一批结果
            if local_results:
                with lock:
                    results.extend(local_results)
                    latencies.extend(local_latencies)
        
        # 创建多个工作者线程
        threads = []
        num_threads = 4  # 统一并行数量为4个
        for i in range(num_threads):
            t = threading.Thread(target=worker, name=f"Worker-{i}")
            threads.append(t)
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        total_time = time.time() - start_time
        return results, total_time, latencies

if __name__ == "__main__":
    thread_impl = MultithreadingImplementation(RealisticIOLoadGenerator())
    results, total_time, latencies = thread_impl.run_with_thread_pool()
    print(f"Thread pool results: {results}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average latency: {sum(latencies) / len(latencies):.4f} seconds")
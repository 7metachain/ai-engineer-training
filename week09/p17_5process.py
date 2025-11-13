from p17_1realIO import RealisticIOLoadGenerator, PerformanceBenchmark
import concurrent.futures
import multiprocessing as mp
import time
from typing import List, Dict, Any

class MultiprocessingImplementation:
    """多进程执行测试"""
    
    def __init__(self, load_generator: RealisticIOLoadGenerator, max_workers: int = 4):  # 统一并行数量为4个
        self.load_generator = load_generator
        self.max_workers = max_workers
    
    @staticmethod
    def process_wrapper(args):
        """进程包装函数（必须是顶层函数）"""
        task_id, url = args
        generator = RealisticIOLoadGenerator()
        return generator.process_http_request(task_id)
    
    def run_with_process_pool(self, num_tasks: int = 100) -> List[Dict[str, Any]]:
        """使用进程池执行"""
        # 准备参数
        args_list = [(i, self.load_generator.get_random_endpoint()) for i in range(num_tasks)]
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            start_time = time.time()
            results = []
            latencies = []
            
            # 使用map保持顺序，移除timeout参数以避免超时问题
            future_results = executor.map(self.process_wrapper, args_list)
            
            for result in future_results:
                results.append(result)
                if result['success']:
                    latencies.append(result['duration'])
            
            total_time = time.time() - start_time
            return results, total_time, latencies
    
    def run_dynamic_processes(self, duration: float = 10.0) -> List[Dict[str, Any]]:  # 统一测试时长为10秒
        """动态多进程执行"""
        manager = mp.Manager()
        results = manager.list()
        latencies = manager.list()
        start_time = time.time()
        task_id = mp.Value('i', 0)
        lock = mp.Lock()
        
        def worker():
            local_results = []
            local_latencies = []
            process_task_id = 0
            
            while time.time() - start_time < duration:
                with lock:
                    current_id = task_id.value
                    task_id.value += 1
                
                result = self.load_generator.process_http_request(current_id)
                local_results.append(result)
                if result['success']:
                    local_latencies.append(result['duration'])
                process_task_id += 1
                
                # 批量写入
                if len(local_results) >= 10:
                    results.extend(local_results)
                    latencies.extend(local_latencies)
                    local_results = []
                    local_latencies = []
            
            # 最后一批
            if local_results:
                results.extend(local_results)
                latencies.extend(local_latencies)
        
        # 启动多个进程
        processes = []
        num_processes = 4  # 统一并行数量为4个
        for i in range(num_processes):
            p = mp.Process(target=worker, name=f"Process-{i}")
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        total_time = time.time() - start_time
        return list(results), total_time, list(latencies)

if __name__ == "__main__":
    multiproc_impl = MultiprocessingImplementation(RealisticIOLoadGenerator())
    results, total_time, latencies = multiproc_impl.run_with_process_pool()
    print(f"Process pool results: {results}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average latency: {sum(latencies) / len(latencies):.4f} seconds")
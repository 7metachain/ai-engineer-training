from p17_1realIO import RealisticIOLoadGenerator, PerformanceBenchmark
import time
from typing import List, Dict, Any

class SynchronousImplementation:
    """同步执行测试"""
    
    def __init__(self, load_generator: RealisticIOLoadGenerator):
        self.load_generator = load_generator
    
    def run_fixed_tasks(self, num_tasks: int = 100) -> List[Dict[str, Any]]:  # 统一任务数量为100
        """固定数量任务的同步执行"""
        results = []
        latencies = []
        
        with PerformanceBenchmark().timer() as get_elapsed:
            for i in range(num_tasks):
                result = self.load_generator.sync_http_request(i)
                results.append(result)
                if result['success']:
                    latencies.append(result['duration'])
            
            total_time = get_elapsed()
        
        return results, total_time, latencies
    
    def run_time_limited(self, duration: float = 10.0) -> List[Dict[str, Any]]:  # 统一测试时长为10秒
        """时间限制内的同步执行"""
        results = []
        latencies = []
        start_time = time.time()
        
        task_id = 0
        while time.time() - start_time < duration:
            result = self.load_generator.sync_http_request(task_id)
            results.append(result)
            if result['success']:
                latencies.append(result['duration'])
            task_id += 1
        
        total_time = time.time() - start_time
        return results, total_time, latencies

if __name__ == "__main__":
    sync_impl = SynchronousImplementation(RealisticIOLoadGenerator())
    results, total_time, latencies = sync_impl.run_fixed_tasks()
    print(f"Fixed tasks results: {results}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average latency: {sum(latencies) / len(latencies):.4f} seconds")
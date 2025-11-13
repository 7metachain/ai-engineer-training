import asyncio
import time
import threading
from typing import Dict, Any


class ContextSwitchOverhead:
    """上下文切换开销测量"""
    
    def __init__(self):
        pass
    
    def measure_coroutine_switch(self, num_switches: int = 1000000) -> Dict[str, float]:
        """精确测量协程上下文切换开销"""
        print(f"精确测量 {num_switches:,} 次协程切换...")
        
        async def precise_switcher():
            # 预热
            for _ in range(1000):
                await asyncio.sleep(0)
            
            # 正式测量
            start_perf = time.perf_counter_ns()
            start_cpu = time.process_time_ns()
            
            for _ in range(num_switches):
                await asyncio.sleep(0)
            
            end_perf = time.perf_counter_ns()
            end_cpu = time.process_time_ns()
            
            return {
                'perf_time': (end_perf - start_perf) / 1e9,
                'cpu_time': (end_cpu - start_cpu) / 1e9  # 修复计算错误
            }
        
        timing = asyncio.run(precise_switcher())
        total_time = timing['perf_time']
        cpu_time = timing['cpu_time']
        
        overhead_per_switch = total_time / num_switches
        switches_per_second = num_switches / total_time
        
        print(f"协程切换: 总时间 {total_time:.6f}s, CPU时间 {cpu_time:.6f}s")
        print(f"每次切换 {overhead_per_switch*1e6:.3f}μs ({switches_per_second:,.0f} 次/秒)")
        
        return {
            'total_time': total_time,
            'cpu_time': cpu_time,
            'per_switch_us': overhead_per_switch * 1e6,
            'switches_per_second': switches_per_second
        }
    
    def measure_thread_switch(self, num_switches: int = 100000) -> Dict[str, float]:
        """精确测量线程上下文切换开销"""
        print(f"精确测量 {num_switches:,} 次线程切换...")
        
        event1 = threading.Event()
        event2 = threading.Event()
        counter = 0
        switch_count = 0
        lock = threading.Lock()
        
        def thread_a():
            nonlocal counter, switch_count
            for _ in range(num_switches // 2):
                with lock:
                    counter += 1
                    switch_count += 1
                event2.set()
                event1.wait()
                event1.clear()
        
        def thread_b():
            nonlocal counter, switch_count
            event1.set()  # 启动
            for _ in range(num_switches // 2):
                event2.wait()
                event2.clear()
                with lock:
                    counter += 1
                    switch_count += 1
                event1.set()
        
        # 预热
        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start(); t2.start(); t1.join(); t2.join()
        
        # 重置事件
        event1.clear()
        event2.clear()
        
        # 正式测量
        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        
        start_perf = time.perf_counter_ns()
        start_cpu = time.process_time_ns()
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        end_perf = time.perf_counter_ns()
        end_cpu = time.process_time_ns()
        
        total_time = (end_perf - start_perf) / 1e9
        cpu_time = (end_cpu - start_cpu) / 1e9
        
        overhead_per_switch = total_time / num_switches
        switches_per_second = num_switches / total_time
        
        print(f"线程切换: 总时间 {total_time:.6f}s, CPU时间 {cpu_time:.6f}s")
        print(f"每次切换 {overhead_per_switch*1e6:.3f}μs ({switches_per_second:,.0f} 次/秒)")
        
        return {
            'total_time': total_time,
            'cpu_time': cpu_time,
            'per_switch_us': overhead_per_switch * 1e6,
            'switches_per_second': switches_per_second
        }
    



class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, test_duration: float = 10.0):
        self.test_duration = test_duration
        self.context_switch = ContextSwitchOverhead()
        self.results = {}
    
    def run_benchmark(self):
        """运行完整的基准测试"""
        print("=== 上下文切换开销基准测试 ===")
        print(f"测试持续时间: {self.test_duration}秒")
        print()
        
        # 运行所有测试
        self.results['coroutine'] = self.context_switch.measure_coroutine_switch()
        print()
        self.results['thread'] = self.context_switch.measure_thread_switch()
        print()
        
        # 打印比较结果
        self.print_comparison()
    
    def print_comparison(self):
        """打印比较结果"""
        print("=== 上下文切换开销比较 ===")
        print(f"{'类型':<10} {'每次切换(μs)':<15} {'每秒切换次数':<15} {'CPU时间(s)':<12}")
        print("-" * 60)
        
        # 按开销排序
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['per_switch_us'])
        
        for switch_type, data in sorted_results:
            switch_type_cn = {
                'coroutine': '协程',
                'thread': '线程'
            }.get(switch_type, switch_type)
            
            print(f"{switch_type_cn:<10} {data['per_switch_us']:<15.3f} "
                  f"{data['switches_per_second']:<15,.0f} "
                  f"{data['cpu_time']:<12.6f}")
        
        print()
        # 计算相对开销
        base_overhead = sorted_results[0][1]['per_switch_us']
        print("相对开销比较 (以最快方式为基准):")
        for switch_type, data in sorted_results:
            switch_type_cn = {
                'coroutine': '协程',
                'thread': '线程'
            }.get(switch_type, switch_type)
            
            relative = data['per_switch_us'] / base_overhead
            print(f"{switch_type_cn}: {relative:.1f}x")


if __name__ == "__main__":
    # 运行基准测试
    benchmark = PerformanceBenchmark(test_duration=10.0)
    benchmark.run_benchmark()
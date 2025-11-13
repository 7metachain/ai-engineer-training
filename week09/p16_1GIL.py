import threading
import time

def io_task(task_id):
    """I/O密集型任务 - 模拟I/O等待"""
    print(f"I/O任务{task_id} 开始")
    time.sleep(1)  # 模拟I/O等待，会释放GIL
    print(f"I/O任务{task_id} 完成")
    return time.time()

def cpu_task(task_id):
    """CPU密集型任务 - 纯计算"""
    print(f"CPU任务{task_id} 开始")
    # CPU密集型计算
    total = 0
    for i in range(500000):
        total += i * i + i ** 0.5
    print(f"CPU任务{task_id} 完成")
    return total

def run_comparison(num_threads=3):
    """运行I/O和CPU任务对比实验"""
    print("=== I/O vs CPU密集型任务对比 ===")
    
    # 测试I/O密集型任务
    print(f"\n--- I/O密集型任务 ({num_threads}线程) ---")
    io_start = time.time()
    io_threads = []
    
    for i in range(num_threads):
        t = threading.Thread(target=io_task, args=(i+1,))
        io_threads.append(t)
        t.start()
    
    for t in io_threads:
        t.join()
    
    io_total_time = time.time() - io_start
    print(f"I/O任务总耗时: {io_total_time:.2f}s")
    
    # 测试CPU密集型任务
    print(f"\n--- CPU密集型任务 ({num_threads}线程) ---")
    cpu_start = time.time()
    cpu_threads = []
    
    for i in range(num_threads):
        t = threading.Thread(target=cpu_task, args=(i+1,))
        cpu_threads.append(t)
        t.start()
    
    for t in cpu_threads:
        t.join()
    
    cpu_total_time = time.time() - cpu_start
    print(f"CPU任务总耗时: {cpu_total_time:.2f}s")
    
    # 计算效率
    io_efficiency = (1.0 * num_threads) / io_total_time  # 理想情况下应该是num_threads倍
    cpu_efficiency = (cpu_total_time / num_threads) / cpu_total_time if num_threads > 1 else 1.0
    
    print(f"\n对比分析:")
    print(f"  I/O任务效率: {io_efficiency:.2f}x (理想: {num_threads}x)")
    print(f"  CPU任务效率: {cpu_efficiency:.2f}x (理想: 1.0x)")
    print(f"  I/O任务可以真正并发: {'✓' if io_efficiency > 0.8 else '✗'}")
    print(f"  CPU任务受GIL限制: {'✓' if cpu_efficiency < 0.8 else '✗'}")
    
if __name__ == "__main__":
    run_comparison()
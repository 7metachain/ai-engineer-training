"""
concurrent.futures 模块示例
演示 ThreadPoolExecutor 和 ProcessPoolExecutor 的使用
"""

import concurrent.futures
import time
import math
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def cpu_intensive_task(n):
    """CPU密集型任务 - 计算大数的平方根"""
    result = 0
    for i in range(1, n * 1000):
        result += math.sqrt(i)
    return result


def io_intensive_task(task_id, duration):
    """IO密集型任务 - 模拟网络请求或文件操作"""
    print(f"任务 {task_id} 开始，线程ID: {threading.current_thread().ident}")
    time.sleep(duration)
    print(f"任务 {task_id} 完成")
    return f"任务 {task_id} 结果"


def thread_pool_example():
    """ThreadPoolExecutor 示例 - 适合IO密集型任务"""
    print("\n=== ThreadPoolExecutor 示例 ===")
    
    # 创建线程池，最大线程数为3
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交多个IO密集型任务
        futures = []
        for i in range(5):
            future = executor.submit(io_intensive_task, i, 1)
            futures.append(future)
        
        # 获取结果
        print("等待所有任务完成...")
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"获取结果: {result}")
            except Exception as e:
                print(f"任务执行出错: {e}")


def process_pool_example():
    """ProcessPoolExecutor 示例 - 适合CPU密集型任务"""
    print("\n=== ProcessPoolExecutor 示例 ===")
    
    # 创建进程池，最大进程数等于CPU核心数
    max_processes = multiprocessing.cpu_count()
    print(f"CPU核心数: {max_processes}")
    
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        # 提交多个CPU密集型任务
        numbers = [100, 200, 300, 400, 500]
        
        # 使用map方法
        print("使用map方法:")
        results = executor.map(cpu_intensive_task, numbers)
        for i, result in enumerate(results):
            print(f"数字 {numbers[i]} 的计算结果: {result:.2f}")


def future_callback_example():
    """Future回调函数示例"""
    print("\n=== Future回调函数示例 ===")
    
    def callback_function(future):
        """回调函数"""
        try:
            result = future.result()
            print(f"回调函数获取结果: {result}")
        except Exception as e:
            print(f"回调函数捕获异常: {e}")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交任务并添加回调
        future = executor.submit(io_intensive_task, "callback_task", 2)
        future.add_done_callback(callback_function)
        
        # 等待任务完成
        time.sleep(3)


def exception_handling_example():
    """异常处理示例"""
    print("\n=== 异常处理示例 ===")
    
    def task_with_exception(task_id):
        if task_id == 2:
            raise ValueError(f"任务 {task_id} 故意抛出异常")
        return f"任务 {task_id} 成功完成"
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i in range(5):
            future = executor.submit(task_with_exception, i)
            futures.append(future)
        
        # 处理结果和异常
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"成功: {result}")
            except Exception as e:
                print(f"捕获异常: {e}")


def executor_comparison():
    """比较不同Executor的性能"""
    print("\n=== Executor性能比较 ===")
    
    # 测试数据
    test_numbers = [1000, 2000, 3000, 4000, 5000]
    
    # ThreadPoolExecutor测试
    print("ThreadPoolExecutor执行CPU密集型任务:")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_intensive_task, test_numbers))
    thread_time = time.time() - start_time
    print(f"线程池耗时: {thread_time:.2f}秒")
    
    # ProcessPoolExecutor测试
    print("ProcessPoolExecutor执行CPU密集型任务:")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_intensive_task, test_numbers))
    process_time = time.time() - start_time
    print(f"进程池耗时: {process_time:.2f}秒")
    
    print(f"进程池比线程池快: {thread_time - process_time:.2f}秒")


def main():
    """主函数"""
    print("concurrent.futures 模块演示")
    print("=" * 40)
    
    # 演示不同功能
    thread_pool_example()
    process_pool_example()
    # future_callback_example()
    # exception_handling_example()
    # executor_comparison()

if __name__ == "__main__":
    main()
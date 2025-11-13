import numpy as np
import threading
import time

def numpy_gil_comparison():
    """NumPy绕过GIL前后对比演示"""
    print("=== NumPy绕过GIL对比演示 ===")
    
    # 创建大数组用于计算
    size = 10000000
    arr1 = np.random.rand(size)
    arr2 = np.random.rand(size)
    
    def numpy_operation(name: str):
        """NumPy计算操作 - 会在C层释放GIL"""
        print(f"{name} 开始")
        start_time = time.time()
        
        # NumPy在C层执行，会释放GIL，允许多线程并行
        result = np.sqrt(arr1 ** 2 + arr2 ** 2)
        
        elapsed = time.time() - start_time
        print(f"{name} 完成，耗时: {elapsed:.2f}s")
        return elapsed
    
    def pure_python_operation(name: str):
        """纯Python计算 - 受GIL限制"""
        print(f"{name} 开始")
        start_time = time.time()
        
        # 纯Python计算，受GIL限制，无法真正并行
        result = []
        for i in range(1000000):
            result.append(i ** 2 + i ** 0.5)
        
        elapsed = time.time() - start_time
        print(f"{name} 完成，耗时: {elapsed:.2f}s")
        return elapsed
    
    print("\n--- NumPy计算（绕过GIL）---")
    print("单线程执行:")
    single_time = numpy_operation("NumPy单线程")
    
    print("\n多线程并发:")
    start_time = time.time()
    
    # NumPy多线程 - 由于释放GIL，可以真正并行
    threads = []
    for i in range(4):
        t = threading.Thread(target=numpy_operation, args=(f"NumPy线程{i+1}",))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    multi_time = time.time() - start_time
    
    print(f"\nNumPy性能对比:")
    print(f"  单线程: {single_time:.2f}s")
    print(f"  4线程并发: {multi_time:.2f}s")
    print(f"  加速比: {single_time*4/multi_time:.2f}x")
    
    print("\n--- 纯Python计算（受GIL限制）---")
    print("单线程执行:")
    py_single_time = pure_python_operation("Python单线程")
    
    print("\n多线程并发:")
    start_time = time.time()
    
    # 纯Python多线程 - 受GIL限制，无法真正并行
    py_threads = []
    for i in range(4):
        t = threading.Thread(target=pure_python_operation, args=(f"Python线程{i+1}",))
        py_threads.append(t)
        t.start()
    
    for t in py_threads:
        t.join()
    
    py_multi_time = time.time() - start_time
    
    print(f"\n纯Python性能对比:")
    print(f"  单线程: {py_single_time:.2f}s")
    print(f"  4线程并发: {py_multi_time:.2f}s")
    print(f"  加速比: {py_single_time*4/py_multi_time:.2f}x")
    

if __name__ == "__main__":
    numpy_gil_comparison()
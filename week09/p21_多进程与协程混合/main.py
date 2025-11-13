"""主函数模块，包含测试和演示代码"""

import asyncio
import time
import random
import multiprocessing as mp
import traceback
from typing import List, Dict, Any

from factories import ProcessPoolFactory
from scheduler import TaskScheduler


async def run_practical_scenario() -> List[Dict[str, Any]]:
    """
    运行一个混合任务场景
    包含网站爬虫、数据处理等应用场景
    
    Returns:
        List[Dict[str, Any]]: 任务执行结果列表
    """
    # 初始化进程池工厂和任务调度器
    pool_factory = ProcessPoolFactory()
    scheduler = TaskScheduler(pool_factory)
    scheduler.initialize(max_workers=4)
    
    try:
        # 准备混合任务列表
        tasks = []
        
        # 1. 网站监控任务（IO密集型）
        print("\n准备网站监控任务...")
        for i, url in enumerate([
            "https://httpbin.org/get",
            "https://httpbin.org/delay/2",
            "https://httpbin.org/status/200",
            "https://httpbin.org/status/404"
        ]):
            tasks.append({
                'id': f'http-{i}',
                'type': 'io',
                'io_type': 'http',
                'url': url,
                'method': 'GET'
            })
        
        # 2. 数据处理任务（CPU密集型）
        print("准备数据处理任务...")
        # 生成一些测试数据
        test_data = [random.randint(1, 1000) for _ in range(500000)]
        
        tasks.append({
            'id': 'cpu-analysis-1',
            'type': 'cpu',
            'operation': 'data_analysis',
            'data': test_data
        })
        
        # 添加一个复杂计算任务
        tasks.append({
            'id': 'cpu-complex-calc',
            'type': 'cpu',
            'operation': 'default',
            'data': [random.randint(1, 1000) for _ in range(200000)]  # 减少数据量
        })
        
        # 添加几个不同复杂度的计算任务
        tasks.append({
            'id': 'cpu-work-1',
            'type': 'cpu',
            'operation': 'default',
            'data': [random.randint(1, 1000) for _ in range(5000000)]  # 较大数据量，产生明显延迟
        })
        
        # 斐波那契计算任务
        for i in range(3):
            tasks.append({
                'id': f'cpu-fib-{i+1}',
                'type': 'cpu',
                'operation': 'fibonacci',
                'data': 50 + i  # 迭代实现下这些值计算很快
            })
        
        # 3. 更多IO任务 - 模拟API调用
        print("准备API调用任务...")
        for i in range(3):
            tasks.append({
                'id': f'api-call-{i}',
                'type': 'io',
                'io_type': 'http',
                'url': 'https://httpbin.org/anything',
                'method': 'POST',
                'params': {'query': f'test{i}'}
            })
        
        # 执行任务并测量性能
        print(f"\n提交 {len(tasks)} 个任务...")
        start_time = time.time()
        
        # 处理任务
        results = await scheduler.schedule_tasks(tasks)
        
        total_time = time.time() - start_time
        print(f"\n任务处理完成！总耗时: {total_time:.2f}秒")
        
        # 分析结果
        completed = [r for r in results if r.get('status') == 'completed']
        failed = [r for r in results if r.get('status') in ['failed', 'error']]
        
        print(f"完成任务数: {len(completed)}")
        print(f"失败任务数: {len(failed)}")
        print(f"总任务数: {len(results)}")
        
        # 显示一些详细信息
        io_tasks = [r for r in results if 'status_code' in r]
        cpu_tasks = [r for r in results if 'operation' in r]
        
        if io_tasks:
            io_duration = sum(r['duration'] for r in io_tasks)
            print(f"\nIO任务统计:")
            print(f"  平均响应时间: {io_duration/len(io_tasks):.2f}秒")
            
            # 统计HTTP状态码
            status_codes = {}
            for r in io_tasks:
                code = r.get('status_code')
                status_codes[code] = status_codes.get(code, 0) + 1
            print(f"  HTTP状态码分布: {status_codes}")
        
        if cpu_tasks:
            cpu_duration = sum(r['duration'] for r in cpu_tasks)
            print(f"\nCPU任务统计:")
            print(f"  平均执行时间: {cpu_duration/len(cpu_tasks):.2f}秒")
            
            # 显示每个CPU任务的详情
            for r in cpu_tasks:
                if r.get('status') == 'completed':
                    # 使用更多小数位显示执行时间
                    print(f"  任务 {r['task_id']} ({r['operation']}): {r['duration']:.4f}秒")
                    if 'result' in r:
                        # 避免打印过大的结果
                        result_str = str(r['result'])
                        if len(result_str) > 50:
                            result_str = result_str[:50] + "..."
                        print(f"    结果: {result_str}")
            
            # 打印详细的CPU任务耗时信息
            print("\n详细CPU任务耗时分析:")
            for r in cpu_tasks:
                if r.get('status') == 'completed':
                    print(f"  {r['task_id']}: {r['duration']*1000:.2f} 毫秒")
        
        return results
        
    finally:
        # 确保关闭调度器和进程池
        scheduler.shutdown()


async def run_unit_tests() -> bool:
    """
    运行单元测试
    
    Returns:
        bool: 测试是否全部通过
    """
    print("\n=== 运行单元测试 ===")
    
    # 初始化测试环境
    pool_factory = ProcessPoolFactory()
    scheduler = TaskScheduler(pool_factory)
    scheduler.initialize(max_workers=2)
    
    try:
        # 测试1: 简单IO任务
        io_task = {
            'id': 'test-io-1',
            'type': 'io',
            'io_type': 'http',
            'url': 'https://httpbin.org/get',
            'method': 'GET'
        }
        
        print("\n测试1: 执行IO任务...")
        io_result = await scheduler.schedule_task(io_task)
        print(f"IO任务测试结果: {io_result['status']}")
        assert io_result['status'] == 'completed', f"IO任务测试失败: {io_result}"
        
        # 测试2: 简单CPU任务
        cpu_task = {
            'id': 'test-cpu-1',
            'type': 'cpu',
            'operation': 'fibonacci',
            'data': 10
        }
        
        print("\n测试2: 执行CPU任务...")
        cpu_result = await scheduler.schedule_task(cpu_task)
        print(f"CPU任务测试结果: {cpu_result['status']}, 计算结果: {cpu_result['result']}")
        assert cpu_result['status'] == 'completed', f"CPU任务测试失败: {cpu_result}"
        assert cpu_result['result'] == 55, f"斐波那契计算错误: {cpu_result['result']} != 55"
        
        # 测试3: 任务异常处理
        error_task = {
            'id': 'test-error-1',
            'type': 'io',
            'io_type': 'unknown',
            'url': 'invalid-url'
        }
        
        print("\n测试3: 异常任务处理...")
        error_result = await scheduler.schedule_task(error_task)
        print(f"异常任务测试结果: {error_result['status']}")
        assert error_result['status'] == 'failed', f"异常处理测试失败: {error_result}"
        
        print("\n所有单元测试通过!")
        return True
        
    except AssertionError as e:
        print(f"\n单元测试失败: {e}")
        return False
    finally:
        scheduler.shutdown()


async def run_performance_test() -> List[Dict[str, Any]]:
    """
    运行性能基准测试
    
    Returns:
        List[Dict[str, Any]]: 性能测试结果
    """
    print("\n=== 运行性能基准测试 ===")
    
    # 初始化测试环境
    pool_factory = ProcessPoolFactory()
    scheduler = TaskScheduler(pool_factory)
    scheduler.initialize(max_workers=4)
    
    try:
        # 创建不同数量级的任务进行测试
        test_scenarios = [
            {'name': '小规模测试', 'io_count': 5, 'cpu_count': 3},
            {'name': '中等规模测试', 'io_count': 15, 'cpu_count': 10},
        ]
        
        results = []
        
        for scenario in test_scenarios:
            print(f"\n执行 {scenario['name']}...")
            
            # 准备任务
            tasks = []
            
            # 添加IO任务
            for i in range(scenario['io_count']):
                delay = 0.5 if i % 3 == 0 else 0  # 部分任务添加延迟
                tasks.append({
                    'id': f'perf-io-{i}',
                    'type': 'io',
                    'io_type': 'http',
                    'url': f'https://httpbin.org/delay/{delay}',
                    'method': 'GET'
                })
            
            # 添加CPU任务
            for i in range(scenario['cpu_count']):
                # 增加计算复杂度
                n = 38 if i % 3 == 0 else (40 if i % 3 == 1 else 42)  # 更高复杂度的斐波那契计算
                tasks.append({
                    'id': f'perf-cpu-{i}',
                    'type': 'cpu',
                    'operation': 'fibonacci',
                    'data': n
                })
            
            # 执行测试
            start_time = time.time()
            scenario_results = await scheduler.schedule_tasks(tasks)
            total_time = time.time() - start_time
            
            # 分析结果
            completed = sum(1 for r in scenario_results if r.get('status') == 'completed')
            throughput = len(tasks) / total_time
            
            print(f"  总任务数: {len(tasks)}")
            print(f"  完成任务数: {completed}")
            print(f"  总耗时: {total_time:.2f}秒")
            print(f"  吞吐量: {throughput:.2f} 任务/秒")
            
            results.append({
                'scenario': scenario['name'],
                'tasks_count': len(tasks),
                'completed': completed,
                'total_time': total_time,
                'throughput': throughput
            })
        
        print("\n性能测试完成!")
        return results
        
    finally:
        scheduler.shutdown()


async def main() -> None:
    """
    主函数，协调运行单元测试、性能测试和实用场景
    """
    try:
        # 运行单元测试
        if not await run_unit_tests():
            print("\n单元测试失败，中止程序")
            return
        
        # 运行性能测试
        await run_performance_test()
        
        # 运行实用场景
        await run_practical_scenario()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Windows上需要设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    print("多进程启动方法已设置为'spawn'")
    
    asyncio.run(main())
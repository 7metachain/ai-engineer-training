import cProfile
import pstats
import asyncio
import aiohttp
from typing import Dict, Any, List, Tuple
import io
import os
from dataclasses import dataclass

class Profiler:
    """性能分析器 - 专注于调用图分析"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
    
    async def profile_async_function(self, coro):
        """分析异步函数的执行"""
        self.profiler.enable()
        try:
            return await coro
        finally:
            self.profiler.disable()
    
    def generate_call_graph_analysis(self) -> str:
        """生成调用图分析报告 - cProfile最强大的功能
        
        调用图分析可以帮助我们：
        1. 识别性能瓶颈（热点函数）
        2. 理解函数调用关系和层级
        3. 发现意外的函数调用和递归
        4. 分析每个函数的调用次数、总时间和累计时间
        """
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        
        print("="*80)
        print("          🚀 cProfile 调用图分析 - 性能瓶颈识别")
        print("="*80)
        print()
        
        # 1. 识别热点函数（按累计时间排序 - 最能反映性能瓶颈）
        print(" 热点函数识别（按累计时间排序）")
        print("这些函数占用了最多的执行时间，是性能优化的首要目标")
        print("-" * 80)
        
        # 获取按累计时间排序的统计信息
        ps_cumtime = pstats.Stats(self.profiler, stream=s)
        ps_cumtime.sort_stats('cumulative')
        
        # 打印详细的函数调用信息
        print(f"{' ncalls':>10} {'tottime':>10} {'percall':>10} {'cumtime':>10} {'percall':>10} {'filename:lineno(function)':<40}")
        print("-" * 80)
        
        # 提取并显示前15个热点函数
        for func, (cc, nc, tt, ct, callers) in list(ps_cumtime.stats.items())[:15]:
            filename, line_num, func_name = func
            # 格式化函数名，处理匿名函数等情况
            if not func_name:
                func_name = "<anonymous>"
            
            # 计算平均调用时间
            per_call_tottime = tt / nc if nc != 0 else 0
            per_call_cumtime = ct / nc if nc != 0 else 0
            
            # 提取文件名（去除路径）
            short_filename = os.path.basename(filename)
            
            print(f"{nc:>10} {tt:>10.4f} {per_call_tottime:>10.4f} {ct:>10.4f} {per_call_cumtime:>10.4f} {short_filename}:{line_num}({func_name})")
        
        print()
        # 2. 调用关系分析
        print(" 关键函数调用关系分析")
        print("-" * 80)
        
        # 找出累计时间最长的3个函数，分析它们的调用关系
        top_functions = list(ps_cumtime.stats.items())[:3]
        for i, (func, stats) in enumerate(top_functions, 1):
            filename, line_num, func_name = func
            cc, nc, tt, ct, callers = stats
            short_filename = os.path.basename(filename)
            
            print(f"\n{i}. 热点函数: {short_filename}:{line_num}({func_name})")
            print(f"   累计时间: {ct:.4f}秒, 调用次数: {nc}")
            
            # 分析调用者
            if callers:
                print(f"   主要调用者:")
                # 按调用次数排序
                sorted_callers = sorted(callers.items(), key=lambda x: sum(x[1][:2]), reverse=True)
                for caller_func, caller_stats in sorted_callers[:3]:  # 显示前3个主要调用者
                    caller_filename, caller_line, caller_name = caller_func
                    caller_ncalls = sum(caller_stats[:2])
                    short_caller_filename = os.path.basename(caller_filename)
                    print(f"     - {short_caller_filename}:{caller_line}({caller_name}) 调用了 {caller_ncalls} 次")
            else:
                print("   无直接调用者（可能是入口函数）")
        
        # 3. 性能优化建议
        print("\n 性能优化建议")
        print("-" * 80)
        
        # 找出调用次数较多的函数
        ps_ncalls = pstats.Stats(self.profiler, stream=s)
        ps_ncalls.sort_stats('ncalls')
        most_called = list(ps_ncalls.stats.items())[-5:]  # 最后5个是调用次数最多的
        
        suggestions_found = False
        
        for func, stats in reversed(most_called):  # 从多到少显示
            filename, line_num, func_name = func
            cc, nc, tt, ct, _ = stats
            short_filename = os.path.basename(filename)
            
            # 降低调用次数阈值，让更多函数显示在建议中
            if nc > 100:  # 降低到100次
                print(f"- {short_filename}:{line_num}({func_name}) 被调用了 {nc:,} 次，考虑缓存结果或优化算法")
                suggestions_found = True
        
        # 找出内部时间较长的函数（可能有计算密集型操作）
        ps_tottime = pstats.Stats(self.profiler, stream=s)
        ps_tottime.sort_stats('tottime')
        most_time_consuming = list(ps_tottime.stats.items())[-3:]  # 最后3个是内部时间最长的
        
        for func, stats in reversed(most_time_consuming):
            filename, line_num, func_name = func
            cc, nc, tt, ct, _ = stats
            short_filename = os.path.basename(filename)
            
            # 降低平均调用时间阈值，让更多函数显示在建议中
            avg_time = tt / nc if nc > 0 else 0
            if avg_time > 0.0001:  # 降低到0.1ms
                print(f"- {short_filename}:{line_num}({func_name}) 单次调用平均耗时 {avg_time*1000:.2f}ms，考虑优化计算逻辑")
                suggestions_found = True
        
        # 如果没有找到符合条件的函数，添加一些通用的优化建议
        if not suggestions_found:
            print("- 本次运行中未发现明显的性能瓶颈，但可以考虑以下通用优化方向：")
            print("  1. 使用异步IO处理并发请求")
            print("  2. 实现适当的缓存机制减少重复计算")
            print("  3. 优化数据结构和算法复杂度")
            print("  4. 考虑使用多进程处理CPU密集型任务")
            print("  5. 减少不必要的对象创建和内存分配")
        
        return s.getvalue()

# 实际应用示例 - 展示cProfile调用图分析能力
async def sample_async_task(session: aiohttp.ClientSession, url: str):
    """示例异步任务 - 模拟网络IO和计算"""
    async with session.get(url) as response:
        # 模拟网络IO操作
        data = await response.read()
        # 模拟一些计算密集型操作
        result = sum(i*i for i in range(1000))
        # 模拟数据处理
        if len(data) > 0:
            _ = data[:100]  # 处理部分数据
        return result

async def nested_task():
    """嵌套任务 - 用于展示调用层级关系"""
    # 模拟一些计算操作
    total = 0
    for i in range(5000):
        total += i * i
    return total

async def complex_workflow(session: aiohttp.ClientSession):
    """复杂工作流 - 包含多个嵌套调用"""
    # 第一阶段：网络请求
    result1 = await sample_async_task(session, "https://httpbin.org/delay/1")
    
    # 第二阶段：嵌套计算
    result2 = await nested_task()
    
    # 第三阶段：并发请求
    sub_tasks = [
        sample_async_task(session, f"https://httpbin.org/delay/{i%2+0.5}")
        for i in range(5)
    ]
    sub_results = await asyncio.gather(*sub_tasks)
    
    return result1 + result2 + sum(sub_results)

async def run_profiling_demo():
    """运行性能分析演示 - 突出cProfile调用图分析能力"""
    print(" cProfile 调用图分析演示")
    print("目标：识别性能瓶颈和调用关系")
    print("="*80)
    print()
    
    profiler = Profiler()
    
    # 创建aiohttp会话
    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        print("正在执行复杂工作流...")
        print("这将触发多层次的函数调用和网络IO操作")
        print()
        
        # 启用profiler（只启用一次，覆盖所有任务）
        profiler.profiler.enable()
        try:
            # 执行多次复杂工作流以获得有意义的统计数据
            for i in range(3):  # 执行3次以获得足够的调用数据
                print(f"\n执行第 {i+1}/3 次工作流...")
                await complex_workflow(session)
        finally:
            # 禁用profiler
            profiler.profiler.disable()
        
        print("\n" + "="*80)
        print("分析完成！正在生成调用图分析报告...")
        print("="*80)
        print()
        
        # 生成调用图分析报告（cProfile最强大的功能）
        profiler.generate_call_graph_analysis()
        
        print("\n" + "="*80)
        print("调用图分析完成")
        print("通过分析结果，您可以：")
        print("1. 快速定位最耗时的函数（热点函数）")
        print("2. 理解函数之间的调用关系和层级")
        print("3. 发现优化机会（高频调用、耗时操作）")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(run_profiling_demo())
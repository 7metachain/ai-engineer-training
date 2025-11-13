import asyncio
import statistics
import aiohttp
from p17_1realIO import RealisticIOLoadGenerator, PerformanceBenchmark, PerformanceResult

class LoadStressTest:
    """负载压力测试"""
    
    def __init__(self):
        self.load_generator = RealisticIOLoadGenerator()
    
    def run_load_test(self, max_concurrency: int = 100, step: int = 20):
        """运行负载压力测试"""
        print(f"=== 负载压力测试 (最大并发: {max_concurrency}) ===")
        
        results = {}
        
        for concurrency in range(step, max_concurrency + 1, step):
            print(f"\n并发级别: {concurrency}")
            
            # 异步测试
            async def run_concurrent_test():
                session = aiohttp.ClientSession()
                
                tasks = [
                    self.load_generator.async_http_request(session, i)
                    for i in range(concurrency)
                ]
                
                start_time = asyncio.get_event_loop().time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                elapsed = asyncio.get_event_loop().time() - start_time
                
                await session.close()
                return results, elapsed
            
            try:
                async_results_raw, async_time = asyncio.run(run_concurrent_test())
                
                # 过滤有效结果
                async_results = []
                latencies = []
                for r in async_results_raw:
                    if isinstance(r, dict):
                        async_results.append(r)
                        if r['success']:
                            latencies.append(r['duration'])
                
                if latencies:
                    throughput = len(async_results) / async_time
                    latency_mean = statistics.mean(latencies)
                else:
                    throughput = 0
                    latency_mean = 0
                
                results[concurrency] = {
                    'async': {
                        'throughput': throughput,
                        'latency_mean': latency_mean,
                        'total_time': async_time,
                        'success_rate': len([r for r in async_results if r['success']]) / len(async_results) if async_results else 0
                    }
                }
                
                print(f"  吞吐量: {throughput:.1f} req/s, "
                      f"平均延迟: {latency_mean*1000:.1f}ms")
                
            except Exception as e:
                print(f"  测试失败: {e}")
                results[concurrency] = {'async': {'throughput': 0, 'latency_mean': 0, 'success_rate': 0}}
        
        return results

if __name__ == "__main__":
    # 运行负载压力测试
    load_test = LoadStressTest()
    load_results = load_test.run_load_test(max_concurrency=100, step=20)
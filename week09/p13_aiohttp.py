import aiohttp
import asyncio
import time
from typing import Dict, Any, Optional, List
import json

class HighPerformanceHttpClient:
    """高性能HTTP客户端 - 演示aiohttp的高级特性"""
    
    def __init__(self, max_connections=100, max_keepalive_connections=30, timeout=30):
        # 配置TCP连接器 - 连接池管理
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,                    # 总连接数限制
            limit_per_host=30,                       # 每主机连接数限制  
            enable_cleanup_closed=True,              # 清理已关闭连接
            keepalive_timeout=30,                    # 保持连接超时
            force_close=False,                       # 重用连接
            ttl_dns_cache=300,                       # DNS缓存时间
            use_dns_cache=True                       # 启用DNS缓存
        )
        
        # 超时配置
        timeout_config = aiohttp.ClientTimeout(
            total=timeout,                           # 总超时时间
            connect=10,                              # 连接超时
            sock_read=timeout                        # 读取超时
        )
        
        # 创建会话
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout_config,
            raise_for_status=False,                  # 手动处理状态码
            headers={
                'User-Agent': 'HighPerformanceClient/1.0',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        )
    
    async def close(self):
        """优雅关闭会话"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
        except Exception:
            pass
        try:
            if self.connector and not self.connector.closed:
                await self.connector.close()
        except Exception:
            pass
    
    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """GET请求封装 - 带错误处理"""
        try:
            async with self.session.get(url, **kwargs) as response:
                result = {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'url': str(response.url),
                    'content_type': response.content_type
                }
    

                
                # 根据内容类型处理响应
                if response.content_type == 'application/json':
                    result['data'] = await response.json()
                elif response.content_type == 'text/html':
                    result['data'] = await response.text()
                else:
                    result['data'] = await response.read()
                
                return result
        except aiohttp.ClientError as e:
            return {'error': f'Client error: {str(e)}', 'status': 0}
        except asyncio.TimeoutError:
            return {'error': 'Request timeout', 'status': 0}
        except Exception as e:
            return {'error': f'Unexpected error: {str(e)}', 'status': 0}
    
    async def post(self, url: str, data=None, json_data=None, **kwargs) -> Dict[str, Any]:
        """POST请求封装 - 支持多种数据格式"""
        try:
            async with self.session.post(url, data=data, json=json_data, **kwargs) as response:
                result = {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'url': str(response.url)
                }
                
                if response.content_type == 'application/json':
                    result['data'] = await response.json()
                else:
                    result['data'] = await response.text()
                
                return result
        except Exception as e:
            return {'error': str(e), 'status': 0}
    
    async def get_with_retry(self, url: str, max_retries=3, **kwargs) -> Dict[str, Any]:
        """带重试机制的GET请求"""
        for attempt in range(max_retries):
            result = await self.get(url, **kwargs)
            if 'error' not in result and result['status'] < 500:
                return result
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return result

# 演示1: 连接池性能对比实验
async def connection_pool_performance_test():
    """测试不同连接池配置的性能"""
    print("=== 连接池性能对比实验 ===")
    
    urls = [f"https://httpbin.org/delay/1"] * 50
    
    # 测试1: 无连接复用
    print("测试1: 无连接复用...")
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        await asyncio.gather(*tasks, return_exceptions=True)
    no_pool_time = time.time() - start
    
    # 测试2: 启用连接复用
    print("测试2: 启用连接复用...")
    connector = aiohttp.TCPConnector(limit=50, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.time()
        tasks = [session.get(url) for url in urls]
        await asyncio.gather(*tasks, return_exceptions=True)
    pooled_time = time.time() - start
    
    print(f"无连接复用: {no_pool_time:.2f}s")
    print(f"启用连接复用: {pooled_time:.2f}s")
    print(f"性能提升: {no_pool_time/pooled_time:.2f}x")
    print()

# 演示2: 并发请求演示
async def concurrent_requests_demo():
    """演示并发请求能力"""
    print("=== 并发请求演示 ===")
    
    client = HighPerformanceHttpClient()
    
    # 不同类型的API端点
    urls = [
        'https://httpbin.org/json',
        'https://httpbin.org/html',
        'https://httpbin.org/delay/1',
        'https://httpbin.org/status/200',
        'https://httpbin.org/headers'
    ]
    
    print(f"并发请求 {len(urls)} 个不同的端点...")
    start = time.time()
    
    # 并发执行所有请求
    tasks = [client.get(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    print(f"总耗时: {elapsed:.2f}s")
    
    # 显示结果
    for i, (url, result) in enumerate(zip(urls, results)):
        status = result.get('status', 'ERROR')
        if 'error' in result:
            print(f"  {i+1}. {url} - 错误: {result['error']}")
        else:
            print(f"  {i+1}. {url} - 状态码: {status}")
    
    await client.close()
    print()

# 演示3: POST请求和JSON处理
async def post_request_demo():
    """演示POST请求和JSON数据处理"""
    print("=== POST请求演示 ===")
    
    client = HighPerformanceHttpClient()
    
    # POST JSON数据
    json_data = {
        'name': '张三',
        'age': 25,
        'city': '北京',
        'skills': ['Python', 'asyncio', 'aiohttp']
    }
    
    print("POST JSON数据到 httpbin.org...")
    result = await client.post(
        'https://httpbin.org/post',
        json_data=json_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if 'error' not in result and result['status'] == 200:
        response_data = result['data']
        print(f"  状态码: {result['status']}")
        print(f"  接收到的JSON数据: {response_data.get('json', {})}")
        print(f"  请求头: {list(response_data.get('headers', {}).keys())}")
    else:
        print(f"  请求失败: {result.get('error', 'Unknown error')}")
    
    await client.close()
    print()

# 演示4: 错误处理和重试机制
async def error_handling_demo():
    """演示错误处理和重试机制"""
    print("=== 错误处理和重试机制演示 ===")
    
    client = HighPerformanceHttpClient(timeout=5)
    
    # 测试各种错误情况
    test_urls = [
        'https://httpbin.org/status/404',  # 404错误
        'https://httpbin.org/status/500',  # 500错误
        'https://httpbin.org/delay/10',    # 超时
        'https://invalid-domain-12345.com' # 无效域名
    ]
    
    for url in test_urls:
        print(f"\n测试URL: {url}")
        
        # 普通请求
        result = await client.get(url)
        if 'error' in result:
            print(f"  普通请求错误: {result['error']}")
        else:
            print(f"  普通请求状态码: {result['status']}")
        
        # 带重试的请求
        retry_result = await client.get_with_retry(url, max_retries=2)
        if 'error' not in retry_result:
            print(f"  重试请求成功: 状态码 {retry_result['status']}")
        else:
            print(f"  重试后仍然失败: {retry_result['error']}")
    
    await client.close()
    print()

# 演示5: 会话管理和连接池统计
async def session_management_demo():
    """演示会话管理和连接池统计"""
    print("=== 会话管理和连接池统计演示 ===")
    
    client = HighPerformanceHttpClient(max_connections=10)
    
    print("连接池配置:")
    print(f"  总连接数限制: {client.connector.limit}")
    print(f"  每主机连接数限制: {client.connector.limit_per_host}")
    # print(f"  Keep-alive超时: {client.connector.keepalive_timeout}s")  # 该属性不存在
    
    # 执行一些请求
    urls = ['https://httpbin.org/json'] * 5
    tasks = [client.get(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    successful = sum(1 for r in results if 'error' not in r and r['status'] == 200)
    print(f"\n成功请求数: {successful}/{len(urls)}")
    
    # 显示连接池状态
    print(f"连接池已关闭: {client.connector.closed}")
    print(f"会话已关闭: {client.session.closed}")
    
    await client.close()
    print("会话已优雅关闭")
    print()

async def main():
    """主函数：运行所有演示"""
    try:
        # 运行所有演示
        await connection_pool_performance_test()
        await concurrent_requests_demo()
        await post_request_demo()
        await error_handling_demo()
        await session_management_demo()
        
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序执行错误: {e}")

if __name__ == "__main__":
    # 使用推荐的asyncio运行方式
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")

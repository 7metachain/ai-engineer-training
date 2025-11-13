# 旧风格（Python 3.4-3.6）协程示例
# 注意：在较新的Python版本中，asyncio.coroutine已被弃用
# 此代码通过注释和结构展示旧风格协程的概念

import asyncio
from functools import wraps

# 模拟@asyncio.coroutine装饰器的功能
def coroutine(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 创建协程对象
        gen = func(*args, **kwargs)
        # 在现代Python中，我们使用ensure_future确保它在事件循环中运行
        return asyncio.ensure_future(_execute_coroutine(gen))
    return wrapper

# 简单的协程执行器
async def _execute_coroutine(gen):
    result = None
    try:
        # 启动生成器
        value = gen.send(None)
        while True:
            try:
                # 等待异步操作完成
                if asyncio.iscoroutine(value) or asyncio.isfuture(value):
                    result = await value
                else:
                    # 对于非协程对象，直接传递
                    result = value
                # 发送结果回生成器
                value = gen.send(result)
            except StopIteration as e:
                # 生成器结束，返回最终结果
                return e.value
    except StopIteration as e:
        return e.value

# 使用自定义的coroutine装饰器，模拟旧风格
@coroutine
def fetch_data(url):
    print(f"Fetching {url}")
    # 模拟网络请求 - 在旧风格中这里会是: yield from asyncio.sleep(5)
    # 但在现代Python中我们需要特殊处理
    yield asyncio.sleep(5)  # 通过yield传递异步任务
    return f"Data from {url}"

@coroutine
def process_multiple_urls(urls):
    # 创建所有任务
    tasks = [fetch_data(url) for url in urls]
    # 并发执行所有任务 - 在旧风格中这里会是: yield from asyncio.gather(*tasks)
    results = yield asyncio.gather(*tasks)  # 通过yield传递异步任务
    return results

# 使用旧风格的事件循环处理方式
print("演示旧风格@coroutine装饰器协程")
loop = asyncio.get_event_loop()
try:
    result = loop.run_until_complete(process_multiple_urls(['a.com', 'b.com']))
    print("结果:", result)
except Exception as e:
    print(f"错误: {e}")
finally:
    loop.close()

print("\n注意: 在实际的Python 3.4-3.6环境中，您可以直接使用@asyncio.coroutine装饰器和yield from语法")
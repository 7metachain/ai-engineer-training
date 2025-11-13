# 新风格（Python 3.7+）
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}")
    await asyncio.sleep(5)
    return f"Data from {url}"

async def main():
    urls = ['a.com', 'b.com']
    # 并发执行 - 直接在main中实现核心逻辑
    start = asyncio.get_event_loop().time()
    # 这行代码实现了并发执行多个异步任务：
    # 1. [fetch_data(url) for url in urls] 创建一个包含所有协程对象的列表
    # 2. * 运算符将列表解包为位置参数传递给asyncio.gather
    # 3. asyncio.gather 会并发执行所有协程，等待它们全部完成
    # 4. 返回的results是一个与输入顺序一致的结果列表
    results = await asyncio.gather(*[fetch_data(url) for url in urls])
    print(results)
    print(f"Concurrent took: {asyncio.get_event_loop().time() - start:.2f}s")

# 运行
asyncio.run(main())
import asyncio

class CustomAwaitable:
    def __init__(self, value):
        self.value = value
    
    def __await__(self):
        """必须返回一个迭代器"""
        # 第一步：准备异步操作
        print(f"Preparing to resolve {self.value}")
        
        # 第二步：yield控制权给事件循环
        yield
        
        # 第三步：操作完成，返回结果
        print(f"Resolved {self.value}")
        return self.value

async def use_custom_awaitable():
    result = await CustomAwaitable("Hello")
    print(result)


# 等价的手动实现
def manual_implementation():
    awaitable = CustomAwaitable("Hello").__await__()
    iterator = iter(awaitable)
    
    try:
        # 第一次next：开始操作
        next(iterator)
        # 此时控制权回到事件循环...
        
        # 模拟I/O完成后继续
        result = next(iterator)  # 获取最终结果
        print(result)
    except StopIteration as e:
        print(e.value)
# 执行异步函数
async def main():
    print("\n=== 执行异步函数 ===")
    
    # 调用自定义awaitable
    result = await use_custom_awaitable()
    print(f"异步函数：获取到结果 - {result}")
    
    # print("\n=== 执行手动实现 ===")
    # manual_implementation()

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())
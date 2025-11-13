import asyncio
import logging

# 配置日志功能
def setup_logging():
    """配置详细的日志记录"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s:%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )

# 演示异步函数
async def demo_task(name, delay):
    """模拟一个异步任务"""
    logger = logging.getLogger(f"task.{name}")
    logger.debug(f"任务 {name} 开始，延迟 {delay}秒")
    print(f"任务 {name} 开始")
    
    await asyncio.sleep(delay)
    
    logger.debug(f"任务 {name} 完成")
    print(f"任务 {name} 完成，耗时 {delay}秒")
    return f"任务 {name} 的结果"

# 主异步函数
async def main_async():
    """主异步函数，使用日志进行调试"""
    logger = logging.getLogger("main")
    logger.debug("主函数开始执行")
    print("主函数开始执行")
    
    # 创建并运行多个异步任务
    tasks = [
        demo_task("A", 1),
        demo_task("B", 2),
        demo_task("C", 1.5)
    ]
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    logger.debug(f"所有任务完成，结果: {results}")
    print("所有任务完成")
    print(f"结果: {results}")

# 主入口
def main():
    """主程序入口 - 使用日志功能进行异步调试"""
    setup_logging()
    print("\n=== 使用日志功能进行异步调试 ===")
    print("- 日志级别设置为DEBUG")
    
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        print("\n=== 调试演示完成 ===")
        print("提示：查看日志输出了解异步任务的执行细节")

if __name__ == "__main__":
    main()
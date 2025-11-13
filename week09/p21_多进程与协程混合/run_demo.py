import asyncio
import multiprocessing as mp
from main import main

if __name__ == "__main__":
    # Windows上需要设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    print("多进程启动方法已设置为'spawn'")
    
    # 运行主程序
    asyncio.run(main())
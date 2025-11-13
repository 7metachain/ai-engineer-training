import asyncio
import selectors
import socket
import time

"""
本代码演示了CPython的asyncio如何通过selectors模块集成操作系统级I/O多路复用机制

selectors模块是Python标准库，它会根据不同操作系统自动选择最高效的I/O多路复用实现：
- Linux: epoll
- macOS/BSD: kqueue
- Windows: 通常是select，但在较新版本中可能使用Windows IOCP
- 其他系统: select

这种抽象层设计使得Python代码可以跨平台高效处理大量并发连接。
"""

class EventLoopIntegration:
    """模拟CPython asyncio事件循环的I/O多路复用集成机制
    
    这个类展示了事件循环如何使用selectors模块管理文件描述符并处理I/O事件，
    这是Python异步编程的底层基础机制之一。
    """
    
    def __init__(self):
        # 创建默认选择器，会根据操作系统自动选择最优的I/O多路复用实现
        # 这正是CPython asyncio事件循环的核心组件之一
        self.selector = selectors.DefaultSelector()
        # 存储文件描述符到回调函数的映射，这是事件分发的关键
        self._fd_to_callback = {}
    
    def add_reader(self, fd, callback):
        """注册文件描述符读事件
        
        此方法模拟了asyncio事件循环的add_reader方法，将文件描述符与回调函数关联起来。
        在实际的asyncio中，这通常用于注册socket等I/O对象的可读事件。
        
        参数:
            fd: 文件描述符（整数）
            callback: 当文件描述符可读时要调用的回调函数
        """
        # 保存回调函数映射
        self._fd_to_callback[fd] = callback
        # 向选择器注册读事件，这会通知操作系统开始监控此文件描述符
        self.selector.register(fd, selectors.EVENT_READ)
    
    def remove_reader(self, fd):
        """移除文件描述符监控
        
        此方法模拟了asyncio事件循环的remove_reader方法，停止监控指定的文件描述符。
        
        参数:
            fd: 要移除监控的文件描述符
        """
        # 从选择器中注销文件描述符，停止操作系统级监控
        self.selector.unregister(fd)
        # 删除对应的回调函数映射
        del self._fd_to_callback[fd]
    
    def _poll_for_events(self, timeout=None):
        """轮询I/O事件
        
        此方法展示了asyncio事件循环的核心工作原理：
        1. 调用selector.select()阻塞等待I/O事件
        2. 当事件发生时，遍历所有就绪的文件描述符
        3. 调用对应的回调函数处理事件
        
        在实际的asyncio实现中，这些回调通常会恢复被挂起的协程执行。
        
        参数:
            timeout: 超时时间（秒），None表示无限等待
        """
        # 调用select方法，这是对操作系统I/O多路复用API的封装调用
        # 例如在Linux上调用epoll_wait，在Windows上可能调用select或IOCP相关API
        events = self.selector.select(timeout)
        
        # 遍历所有就绪的文件描述符和事件掩码
        for key, mask in events:
            # 获取对应的回调函数
            callback = self._fd_to_callback[key.fd]
            # 调用回调函数，这在asyncio中会恢复等待该I/O事件的协程
            callback()  # 触发回调，恢复协程执行
    
    async def sleep(self, delay):
        """真正的异步sleep实现
        
        此方法演示了asyncio.sleep的基本原理，虽然它不直接使用selectors，
        但展示了如何通过事件循环创建可等待的future对象，这是协程调度的基础。
        
        参数:
            delay: 延迟时间（秒）
        """
        # 获取当前运行的事件循环
        loop = asyncio.get_running_loop()
        # 创建future对象，这是asyncio中表示异步操作结果的基本单元
        future = loop.create_future()
        
        def wakeup():
            """延迟到期后的回调函数"""
            if not future.done():
                # 设置future结果，这会唤醒等待该future的协程
                future.set_result(None)
        
        # 注册定时器，在指定延迟后调用wakeup函数
        # 这是事件循环中定时器事件的处理方式
        timer_handle = loop.call_later(delay, wakeup)
        try:
            # 挂起当前协程，等待future完成
            return await future
        finally:
            # 确保定时器被取消，防止内存泄漏
            timer_handle.cancel()

# 测试代码

async def test_sleep():
    """测试异步sleep功能
    
    演示了事件循环如何处理定时器事件，这是CPython asyncio中除I/O事件外的另一种事件类型。
    虽然定时器事件不直接使用selectors，但它展示了事件循环如何管理和调度异步任务。
    """
    print("开始测试异步sleep功能...")
    # 创建事件循环集成实例，这模拟了CPython中事件循环的核心组件
    integration = EventLoopIntegration()
    
    start_time = time.time()
    print(f"当前时间: {time.strftime('%H:%M:%S')}")
    
    # 测试异步sleep 2秒
    # 这里调用的是我们实现的sleep方法，它使用了asyncio的future机制
    await integration.sleep(2)
    
    end_time = time.time()
    print(f"2秒后时间: {time.strftime('%H:%M:%S')}")
    print(f"实际耗时: {end_time - start_time:.2f}秒")

async def test_socket_integration():
    """测试套接字集成功能
    
    这是本示例的核心，演示了CPython的asyncio如何通过selectors模块与操作系统的I/O多路复用机制集成。
    具体展示了：
    1. 如何注册文件描述符到selector
    2. 如何通过事件回调处理I/O事件
    3. 如何将底层的I/O事件与高层的协程机制结合
    """
    print("\n开始测试套接字集成功能...")
    # 创建事件循环集成实例
    integration = EventLoopIntegration()
    
    # 创建一个简单的TCP回显服务器
    # 这里使用非阻塞socket，这是与I/O多路复用配合使用的关键
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', 8888))
    server_socket.listen(1)
    # 设置为非阻塞模式，这样accept()等操作不会阻塞整个线程
    server_socket.setblocking(False)
    
    print(f"服务器启动在 127.0.0.1:8888")
    
    # 服务器接受连接的协程
    # 创建future对象，用于在连接建立时通知主协程
    # 这正是CPython asyncio中将回调与协程连接起来的常用模式
    accepted_connection = asyncio.get_running_loop().create_future()
    
    def accept_connection():
        """连接接受回调函数
        
        当服务器socket可读时（有新连接到来），此回调会被调用。
        这模拟了CPython asyncio中socket连接处理的底层机制。
        """
        # 因为socket是非阻塞的，所以此处的accept()不会阻塞
        conn, addr = server_socket.accept()
        # 确保新连接也是非阻塞的
        conn.setblocking(False)
        print(f"接受到来自 {addr} 的连接")
        # 设置future结果，唤醒等待此future的协程
        accepted_connection.set_result((conn, addr))
    
    # 注册服务器socket到事件循环集成
    # 这一步骤展示了如何将文件描述符与回调函数关联并交给selector管理
    # 在CPython的asyncio实现中，这是通过loop.add_reader()方法完成的
    integration.add_reader(server_socket.fileno(), accept_connection)
    
    # 启动一个简单的客户端
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setblocking(False)
    # 使用connect_ex避免阻塞（非阻塞模式下connect可能返回EINPROGRESS）
    client_socket.connect_ex(('127.0.0.1', 8888))
    print("客户端尝试连接...")
    
    # 轮询事件直到连接建立
    # 这模拟了CPython事件循环的事件循环过程
    # 在实际的asyncio中，这个过程由事件循环的run_forever()或run_until_complete()方法管理
    start_time = time.time()
    while not accepted_connection.done() and time.time() - start_time < 5:
        # 调用poll方法检查I/O事件，这是selectors模块的核心功能
        # 在实际的asyncio中，这对应于事件循环的_run_once()方法
        integration._poll_for_events(timeout=0.1)
        # 让出控制权，允许其他协程运行
        await asyncio.sleep(0.01)
    
    if accepted_connection.done():
        conn, addr = accepted_connection.result()
        
        # 测试发送和接收数据
        test_data = b"Hello, IO Multiplexing!"
        client_socket.sendall(test_data)
        print(f"客户端发送: {test_data.decode()}")
        
        # 接收数据的协程
        # 再次使用future模式将回调与协程连接
        received_data = asyncio.get_running_loop().create_future()
        
        def receive_data():
            """数据接收回调函数
            
            当连接socket可读时（有数据到来），此回调会被调用。
            展示了asyncio如何处理socket数据接收事件。
            """
            # 非阻塞模式下的recv
            data = conn.recv(1024)
            if data:
                print(f"服务器接收: {data.decode()}")
                # 回显数据
                conn.sendall(data)
                # 设置future结果
                received_data.set_result(data)
        
        # 注册连接socket到事件循环集成
        # 这展示了如何动态添加新的文件描述符到监控中
        integration.add_reader(conn.fileno(), receive_data)
        
        # 轮询事件直到接收到数据
        start_time = time.time()
        while not received_data.done() and time.time() - start_time < 5:
            integration._poll_for_events(timeout=0.1)
            await asyncio.sleep(0.01)
        
        # 客户端接收回显数据
        client_data = client_socket.recv(1024)
        print(f"客户端接收回显: {client_data.decode()}")
        
        # 清理资源
        # 移除文件描述符监控，这是资源管理的重要步骤
        integration.remove_reader(conn.fileno())
        conn.close()
    
    # 清理资源
    integration.remove_reader(server_socket.fileno())
    server_socket.close()
    client_socket.close()
    print("连接已关闭，测试完成")

async def main():
    """主测试函数
    
    整合所有测试用例，展示CPython的asyncio通过selectors集成操作系统I/O多路复用的完整流程。
    """
    print("=== IO多路复用集成测试 ===")
    
    # 测试异步sleep功能
    await test_sleep()
    
    # 测试套接字集成功能
    await test_socket_integration()
    
    print("\n=== 所有测试完成 ===")

if __name__ == "__main__":
    # 运行主测试函数
    # asyncio.run()会创建一个事件循环并运行指定的协程
    # 这是CPython 3.7+推荐的运行异步代码的方式
    asyncio.run(main())
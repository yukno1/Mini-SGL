"""
总体说明：
本模块基于 ZMQ + msgpack 封装了一组进程间通信（IPC）"队列"，
用于多进程推理架构中各角色（tokenizer / scheduler / engine / server）
之间的消息传递。提供两种通信模式：

  1. PUSH/PULL 模式（点对点、负载均衡）：
     ZmqPushQueue  -> ZmqPullQueue（同步版）
     ZmqAsyncPushQueue -> ZmqAsyncPullQueue（异步版）
     每条消息只会被一个 PULL 端消费。

  2. PUB/SUB 模式（一对多广播）：
     ZmqPubQueue -> ZmqSubQueue
     一条消息会广播给所有订阅者。

序列化协议：msgpack（比 JSON 更快、体积更小）。
消息流程：对象 T --encoder--> dict --msgpack.packb--> bytes 发送；
         bytes --msgpack.unpackb--> dict --decoder--> 对象 T。

通用约定：
  - addr: ZMQ 地址，如 "ipc:///tmp/xxx.sock" 或 "tcp://127.0.0.1:5555"；
  - create: True 表示本端是"服务端"，调用 socket.bind(addr) 监听地址；
            False 表示本端是"客户端"，调用 socket.connect(addr) 连接对端；
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

# 泛型类型变量：表示队列中传输的对象类型
T = TypeVar("T")


class ZmqPushQueue(Generic[T]):
    """基于 ZeroMQ PUSH-PUSH 模式的同步消息队列（生产者端）。

    典型用法：多个生产者 PUSH 消息到同一个消费者（PULL 端），
    类似进程间通信中的任务队列。消息使用 msgpack 序列化。

    属性:
        context: ZeroMQ 上下文对象，管理底层 IO 线程和 socket 资源。
        socket: ZeroMQ PUSH 套接字，用于发送消息。
        encoder: 用户自定义编码函数，将对象 T 转换为可序列化的字典。
    """

    __slots__ = ("context", "socket", "encoder")

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """初始化 PUSH 套接字。

        Args:
            addr: ZeroMQ 连接地址，如 "tcp://127.0.0.1:5555" 或 "ipc:///tmp/xxx"。
            create: 为 True 时绑定（bind）该地址，作为队列的服务端；
                    为 False 时连接（connect）该地址，作为客户端。
            encoder: 编码函数，在发送前将对象 T 转换为字典以便 msgpack 序列化。
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        """编码并阻塞发送一个对象（若底层发送缓冲已满则阻塞）。"""
        # 先用自定义 encoder 转成字典，再用 msgpack 序列化为字节
        # use_bin_type=True 使 bytes 类型以 bin 类型编码，解码时可还原
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        # copy=False 避免 Python 层的内存拷贝，直接引用缓冲区发送
        self.socket.send(event, copy=False)

    def stop(self):
        """关闭套接字并销毁上下文，释放所有底层资源。"""
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    """基于 asyncio 的异步 PUSH 消息队列（生产者端）。

    与 ZmqPushQueue 功能相同，但使用 zmq.asyncio，
    发送操作为协程，不会阻塞事件循环。
    """

    __slots__ = ("context", "socket", "encoder")

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """初始化异步 PUSH 套接字。"""
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        """异步编码并发送一个对象；缓冲区满时挂起当前协程而非阻塞线程。"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        """关闭套接字并销毁上下文，释放所有底层资源。"""
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    """基于 ZeroMQ PULL 模式的同步消息队列（消费者端）。

    与 ZmqPushQueue 配对使用：接收并解码 PUSH 端发送的消息。
    额外提供了原始字节访问和解码接口，便于延迟解码场景。
    """

    __slots__ = ("context", "socket", "decoder")

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """初始化 PULL 套接字。

        Args:
            addr: ZeroMQ 连接地址。
            create: 为 True 时绑定（bind），通常消费端作为服务端；
                    为 False 时连接（connect）。
            decoder: 解码函数，将 msgpack 解包出的字典还原为对象 T。
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        """阻塞接收一条消息并解码为对象 T。"""
        event = self.socket.recv()
        # raw=False 使 bytes/str 字段解码为 Python 字符串而非 bytes
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        """阻塞接收一条消息，返回未经解码的原始字节。"""
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        """将先前通过 get_raw() 获得的原始字节解码为对象 T。

        适用于"先取消息、稍后在需要时再解码"的场景，
        可避免不必要的解码开销。
        """
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        """非阻塞检查队列当前是否为空（无待接收消息）。

        使用 timeout=0 的 poll 轮询；返回 True 表示无消息可取。
        """
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        """关闭套接字并销毁上下文，释放所有底层资源。"""
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    """基于 asyncio 的异步 PULL 消息队列（消费者端）。

    与 ZmqAsyncPushQueue 配对使用，接收操作为协程，
    等待消息时挂起协程而不阻塞事件循环。
    """

    __slots__ = ("context", "socket", "decoder")

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """初始化异步 PULL 套接字，参数含义同 ZmqPullQueue.__init__。"""
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        """异步接收一条消息并解码为对象 T。"""
        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        """关闭套接字并销毁上下文，释放所有底层资源。"""
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    """基于 ZeroMQ PUB-SUB 模式的同步发布者（publisher）。

    发布的消息会被所有订阅者（ZmqSubQueue）收到，
    适合一对多的广播场景，如向多个 worker 广播控制命令。
    """

    __slots__ = ("context", "socket", "encoder")

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """初始化 PUB 套接字，参数含义同 ZmqPushQueue.__init__。"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        """直接发送原始字节，不做任何编码（用于转发或预编码消息）。"""
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        """编码并发布一个对象，所有已订阅的 SUB 端都会收到该消息。"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        """关闭套接字并销毁上下文，释放所有底层资源。"""
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    """基于 ZeroMQ SUB 模式的同步订阅者（subscriber）。

    与 ZmqPubQueue 配对使用，接收发布者广播的消息。
    注意：SUB 端默认不接收任何消息，必须设置订阅前缀。
    """

    __slots__ = ("context", "socket", "decoder")

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """初始化 SUB 套接字。

        参数含义同 ZmqPullQueue.__init__；
        额外通过订阅空字符串 "" 表示接收所有消息（不过滤主题前缀）。
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # 订阅所有消息：空字符串前缀匹配任何消息
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        """阻塞接收一条广播消息并解码为对象 T。"""
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        """非阻塞检查当前是否有待接收的消息。"""
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        """关闭套接字并销毁上下文，释放所有底层资源。"""
        self.socket.close()
        self.context.term()

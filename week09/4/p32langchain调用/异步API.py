import asyncio
import time
import os

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def _make_llm():
    """
    初始化 DashScope（通义千问）聊天模型。
    依赖环境变量：
        DASHSCOPE_API_KEY  - 必需，你的 DashScope API Key
        DASHSCOPE_MODEL    - 可选，默认 qwen-turbo
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY 环境变量")
    return ChatTongyi(model=model, temperature=0.9)


def _make_chain():
    """
    构建 Prompt → LLM → 字符串输出 的可组合链。
    提示词模板：
        system: You are a helpful assistant.
        human:  What is a good name for a company that makes {product}?
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "What is a good name for a company that makes {product}?",
            ),
        ]
    )
    llm = _make_llm()
    return prompt | llm | StrOutputParser()


def generate_serially():
    """
    串行调用：循环 5 次，每次调用一次链。
    用于对比并发性能。
    """
    chain = _make_chain()
    for _ in range(5):
        try:
            resp = chain.invoke({"product": "toothpaste"})
            # print(resp)
        except Exception as e:
            print(f"调用出错: {e}")
            raise


async def async_generate(chain):
    """单次异步调用，供并发任务使用。"""
    try:
        resp = await chain.ainvoke({"product": "toothpaste"})
        # print(resp)
    except Exception as e:
        print(f"调用出错: {e}")
        raise


async def generate_concurrently():
    """
    并发调用：一次性生成 5 个任务，并发执行。
    使用 asyncio.gather 等待全部完成。
    """
    chain = _make_chain()
    tasks = [async_generate(chain) for _ in range(5)]
    await asyncio.gather(*tasks)


def main():
    """
    主函数：先并发执行，再串行执行，并打印耗时对比。
    """
    s = time.perf_counter()
    asyncio.run(generate_concurrently())
    elapsed = time.perf_counter() - s
    print("\033[1m" + f"并发执行花费了 {elapsed:0.2f} 秒." + "\033[0m")

    s = time.perf_counter()
    generate_serially()
    elapsed = time.perf_counter() - s
    print("\033[1m" + f"串行执行花费了 {elapsed:0.2f} 秒." + "\033[0m")


if __name__ == "__main__":
    # 脚本入口：并发 vs 串行性能对比
    main()

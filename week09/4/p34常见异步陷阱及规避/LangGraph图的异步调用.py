try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    from langgraph.graph import StateGraph
    START = "__start__"
    END = "__end__"
import asyncio

# 模拟异步天气工具
async def weather(city: str) -> str:
    return f"{city} 晴，25°C"

# 定义异步节点
async def get_weather(state):
    city = state.get("city", "北京")
    result = await weather(city)
    return {"result": result}

# 构建极简 LangGraph 工作流
workflow = StateGraph(dict)
workflow.add_node("get_weather", get_weather)
workflow.add_edge(START, "get_weather")
workflow.add_edge("get_weather", END)

# 运行
async def main():
    app = workflow.compile()
    inputs = {"city": "上海"}
    async for event in app.astream(inputs):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())

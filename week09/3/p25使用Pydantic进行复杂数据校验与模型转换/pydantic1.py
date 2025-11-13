import os
import json
import re
import logging
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvSettings(BaseSettings):
    """环境配置"""
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# 初始化API客户端
env_settings = EnvSettings()
client = OpenAI(
    api_key=env_settings.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY"),
    base_url=env_settings.DASHSCOPE_API_BASE,
)


class ProductItem(BaseModel):
    """产品项模型 - 主要验证模型"""
    id: int = Field(gt=0)
    name: str
    description: str
    price: float = Field(gt=0)
    quantity: int = Field(ge=0)
    category: str

    model_config = {
        "extra": "forbid",  # 禁止多余字段，保证结构严格
    }

    @field_validator("name", "category")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


def build_prompt(product_description: str, schema: dict) -> str:
    """构造严格的提示词，要求输出纯 JSON。"""
    return (
        "请根据以下产品描述，严格输出符合 JSON Schema 的纯 JSON 数据：\n\n"
        f"产品描述：{product_description}\n\n"
        "JSON Schema:\n"
        f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n\n"
        "输出要求：\n"
        "- 仅输出可以被直接解析的纯 JSON 字符串。\n"
        "- 禁止输出 Markdown 代码块（例如 ```json ）。\n"
        "- 不要包含任何额外文字、解释或前后缀。\n"
        "- 对描述中未明确给出的字段合理补全：\n"
        "  - id 使用任意正整数。\n"
        "  - quantity 为库存数量（整数）。若描述为‘库存充足’，请设为 100。\n"
    )


def call_qwen_model(prompt: str) -> str:
    """调用模型并返回原始文本结果。"""
    try:
        logger.info("正在调用模型生成结构化 JSON...")

        # 确保API密钥已配置
        if not env_settings.DASHSCOPE_API_KEY and not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("未配置 DASHSCOPE_API_KEY 环境变量")

        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个结构化数据的处理器，精通 JSON。"
                        "请严格按给定 JSON Schema 输出纯 JSON。"
                        "输出将被直接解析，禁止代码块与解释文本。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        result = response.choices[0].message.content
        logger.info("模型调用成功")
        return result

    except Exception as e:
        logger.error(f"调用模型时发生错误: {e}")
        raise


def extract_pure_json(text: str) -> str:
    """去除可能的代码块或前后缀，仅保留 JSON 字符串。"""
    if not text:
        raise ValueError("空响应，无法解析 JSON")

    # 捕获 ```json ... ``` 或 ``` ... ``` 中的内容
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # 若包含多余文字，尝试截取第一个 '{' 到最后一个 '}'
    if text.strip()[0] != "{" or text.strip()[-1] != "}":
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    return text.strip()


def main() -> None:
    product_description = (
        "新款智能手机，6.1英寸OLED屏幕，A15仿生芯片，128GB存储，售价4999元，"
        "库存充足，属于电子产品类别。"
    )
    schema = ProductItem.model_json_schema()
    prompt = build_prompt(product_description, schema)

    # 调用模型生成 JSON 格式的产品信息
    raw_text = call_qwen_model(prompt)
    logger.info("模型返回的原始文本：%s", raw_text)

    # 清洗并校验
    cleaned_json = extract_pure_json(raw_text)
    logger.info("清洗后的 JSON 文本：%s", cleaned_json)

    try:
        product = ProductItem.model_validate_json(cleaned_json)
    except Exception:
        # 回退：若为 Python 字典，先 loads 后再校验
        data = json.loads(cleaned_json)
        product = ProductItem.model_validate(data)

    logger.info(
        "校验后的产品信息：%s",
        json.dumps(product.model_dump(), ensure_ascii=False, indent=2),
    )


if __name__ == "__main__":
    main()

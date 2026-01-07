import json
from typing import Optional
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from util.my_llm import llm_gpt


# 使用pydantic定义一个类
class Joke(BaseModel) :
    """笑话的结构类   数据结构类  类似POVO"""
    setup: str = Field(description="笑话的开头部分") # 笑话的铺垫部分
    punchline: str = Field(description="笑话的包袱/笑点") # 笑话的爆笑部分
    rating: Optional[int] = Field(description="笑话的有趣程度评分：范围在1到10") # 可选地笑话评分字段

prompt_template = PromptTemplate.from_template("帮我生成一个关于{topic}的笑话")

#
runnable = llm_gpt.with_structured_output(Joke)

chain = prompt_template | runnable

resp = chain.invoke({"topic":"路易十六"})

# 直接输出结果
print(resp)

# 结果转为对象
print(resp.__dict__)

# 将结果转为JSON字符串
json_str = json.dumps(resp.__dict__)
print(json_str)
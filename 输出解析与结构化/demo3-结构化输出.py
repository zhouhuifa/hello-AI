import json
from typing import Optional

from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from util.my_llm import llm_gpt


prompt = ChatPromptTemplate.from_template(
    # 基本指令
    "尽你可能回答用户的问题。"
    # 输出格式的要求
    '你必须始终输出一个包含"answer"和"followup_question"键的JSON对象。其中”answer“代表：对用户问题的回答；"followup_quest"代表：用户可能提出的后续问题'
    "{question}" # 用户问题的占位符
)

# SimpleJsonOutputParser 直接在模版中指定输出格式，可以不用定义数据模型类
chain = prompt | llm_gpt | SimpleJsonOutputParser()

resp = chain.invoke({"question":"细胞的动力源是什么？"})

# 直接输出结果
print(resp)

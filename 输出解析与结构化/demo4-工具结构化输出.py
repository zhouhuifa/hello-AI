import json
from typing import Optional
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from util.my_llm import llm_gpt


# 使用pydantic定义一个类
class ResponseFormatter(BaseModel) :
    """ 数据结构类  类似POVO"""
    answer: str = Field(description="对用户问题的回答") #
    followup_question: str = Field(description="用户可能提出的后续问题") #

#
runnable = llm_gpt.bind_tools([ResponseFormatter])

resp = runnable.invoke("细胞的动力源是什么")

# 直接输出结果
print(resp)

resp.pretty_print()

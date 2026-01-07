from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from demo3提示词模版 import prompt_template
from util.my_llm import llm_gpt


# 步骤一


# 步骤二

# {topic} 变量占位符
prompt_template1 = ChatPromptTemplate([
    ("system","你是一个幽默的电视台主持人"),
    ("user","帮我生成一个简短的，关于{topic}的报幕词。")
])
# print(prompt_template1.invoke({"topic":"相声"}))


# 消息占位符 MessagesPlaceholder
prompt_template2 = ChatPromptTemplate([
    ("system","你是一个幽默的电视台主持人"),
    MessagesPlaceholder("msgs")
])
prompt_template2.invoke({"msgs":[HumanMessage(content="你好，主持人！")]})


# 问题模版和模型组成链
chain = prompt_template2 | llm_gpt

res = chain.invoke({"msgs":[HumanMessage(content="你好，主持人！")]})
print(res)


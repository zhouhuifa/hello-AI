from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import  ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate

from util.my_llm import llm_gpt


examples = [
    {"input":"2 @ 2", "output":"4"},
    {"input":"2 @ 3", "output":"5"},

]

base_prompt = ChatPromptTemplate.from_messages(
    [
        ('human','{input}'),
        ('ai', '{output}')
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=base_prompt,
)

final_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能机器人AI助手"),
    few_shot_prompt,
    MessagesPlaceholder("msgs"),
])

# 字符串解析器 StrOutputParser()
chain = final_template | llm_gpt | StrOutputParser()

print(chain.invoke({"msgs":[HumanMessage(content="中国第一个皇帝是谁")]}))
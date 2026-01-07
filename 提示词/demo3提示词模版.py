from langchain_core.prompts import PromptTemplate

from util.my_llm import llm_gpt

prompt_template = PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的报幕词。")
chain = prompt_template | llm_gpt
# res = prompt_template.invoke({"topic":"相声"})
res = chain.invoke({"topic":"相声"})
print(res)
from langchain_core.prompts import PromptTemplate

from util.my_llm import llm_gpt


prompt = (
    PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的报幕词。")
    + "，要求：1.内容搞笑一点"
    + "，输出的内容采用{language}"
)

chain = prompt | llm_gpt

print(chain.invoke({"topic":"相声", "language":"English"}))
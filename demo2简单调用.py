from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from langchain_openai import ChatOpenAI

from env_utils import XIAOAI_API_KEY, XIAOAI_BASE_URL, LOCAL_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 官方的大模型
llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.8,
    api_key=XIAOAI_API_KEY,
    base_url=XIAOAI_BASE_URL,
)
 
# llm = ChatOpenAI(
#     model='claude-3-7-sonnet-20250219',
#     temperature=0.8,
#     api_key=XIAOAI_API_KEY,
#     base_url=XIAOAI_BASE_URL,
# )

# llm = ChatOpenAI(
#     model='deepseek-chat',
#     temperature=0.8,
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
# )

# 本地部署的大模型
# llm_local = ChatOpenAI(
#     model='qwen3-8b',
#     temperature=0.8,
#     api_key='xx',
#     base_url=LOCAL_BASE_URL,
# )

message = [
    ('system', '你是一个智能助手'),
    ('human', '请介绍什么是深度学习？')
]

resp = llm.invoke(message)
print(resp)
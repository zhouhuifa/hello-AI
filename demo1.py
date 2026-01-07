from openai import OpenAI


client = OpenAI(base_url = "")
# 测试本地部署的大模型
resp = client.chat.completions.create(
    model='qwen3-8b',
    messages=[{'role':'user', 'content': '请介绍什么是深度学习？'}],
    temperature=0.8,
    presence_penalty=1.5,
    extra_body={'chat_template_kwargs': {'enable_thinking': True}},
)
print(resp)
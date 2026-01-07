from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from util.my_llm import llm_gpt

# 1. 提示词模版
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个智能助手，尽可能的调用工具回答用户的问题.提供的聊天历史包含与你对话用户的相关信息。'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}'),
    # MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),
])

chain = prompt | llm_gpt

# 存储聊天记录 （可以存储到：内存、关系型数据库、Redis）

store = {} # 用来保存历史消息， key:会话ID session_id

def get_session_history(session_id: str):
    """从关系型数据库中的历史消息列表中 返回当前会话的所有历史消息"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string='sqlite:///chat_history.db',
    )

# langchain中所有的消息类型：SystemMessage HumanMessage AIMessage ToolMessage

# 3. 创建带历史记录功能的处理链
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_message_key='input',
    history_messages_key='chat_history',
)

# 4. 剪辑和摘要上下文，历史记录。 保留最近的前2条消息，把之前的所有消息形成摘要

def summarize_messages(current_input):
    """剪辑和摘要上下文，历史记录"""
    session_id = current_input['config']["configurables"]["session_id"]
    if not session_id:
        raise ValueError("必须通过Config参数提供session_id")

    # 获取当前会话ID的所有历史聊天记录
    chat_history = get_session_history(session_id)
    stored_messages = chat_history.messages
    if len(stored_messages) <= 2: # 保留最近2条消息
        return False

    # 剪辑消息列表
    last_tow_messages = stored_messages[-2:] # 保留的2条消息
    messages_to_summarize = stored_messages[:-2] # 需要进行摘要的消息列表

    summarization_promat = ChatPromptTemplate.format_messages([
        ("system","请将以下对话历史压缩为一条保留关键信息的摘要信息。"),
        ("placeholder","{chat_history}"),
        ("human","请生成包含上述对话核心内容的摘要，保留重要实事和决策。")
    ])
    summarization_chain = summarization_promat | llm_gpt
    # 生成摘要
    summary_message = summarization_chain.invoke({'chat_history': messages_to_summarize})

    # 重建历史记录： 摘要+最后2条原始消息
    chat_history.clear()
    chat_history.add_message(summary_message)
    for msg in last_tow_messages:
        chat_history.add_message(msg)
    return True

# result1 = chain_with_message_history.invoke({'input':'你好，我是周辉发'},config={"configurable":{"session_id": "user123"}})
# print(result1)

result2 = chain_with_message_history.invoke({'input':'我的名字是什么'},config={"configurable":{"session_id": "user123"}})
print(result2)
#
# result3 = chain_with_message_history.invoke({'input':'历史上和我同名的人有哪些？'},config={"configurable":{"session_id": "user124"}})
# print(result3)

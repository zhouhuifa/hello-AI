from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
import gradio as gr
from util.my_llm import llm_gpt

# 1. 提示词模版
prompt = ChatPromptTemplate.from_messages([
    ('system', "{system_message}"),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}'),
])
chain = prompt | llm_gpt

# 2. 存储聊天记录 （可以存储到：内存、关系型数据库、Redis）

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
    session_id = current_input['config']["configurable"]["session_id"]
    if not session_id:
        raise ValueError("必须通过Config参数提供session_id")

    # 获取当前会话ID的所有历史聊天记录
    chat_history = get_session_history(session_id)
    stored_messages = chat_history.messages
    if len(stored_messages) <= 2: # 保留最近2条消息
        return { "original_messages": stored_messages,  "summary": "" }

    # 剪辑消息列表
    last_tow_messages = stored_messages[-2:] # 保留的2条消息
    messages_to_summarize = stored_messages[:-2] # 需要进行摘要的消息列表

    summarization_promat = ChatPromptTemplate.from_messages([
        ("system","请将以下对话历史压缩为一条保留关键信息的摘要信息。"),
        ("placeholder","{chat_history}"),
        ("human","请生成包含上述对话核心内容的摘要，保留重要实事和决策。")
    ])
    summarization_chain = summarization_promat | llm_gpt
    # 生成摘要
    summary_message = summarization_chain.invoke({'chat_history': messages_to_summarize})
    return {
        "original_messages": last_tow_messages, # 原始的前两天消息
        "summary": summary_message, # 生成的摘要
    }

# 5. 最终的链
# RunnablePassthrough 默认会将输入数据原样传递到下游，而assign()方法允许在保留原始输入的同时，用过指定键值对(如 messages_summarized=summarize_messages)
final_chain = RunnablePassthrough.assign(messages_summarized=summarize_messages) | RunnablePassthrough.assign(
    input=lambda x: x['input'],
    chat_history=lambda x: x['messages_summarized']['original_messages'],
    system_message=lambda x: f"你是一个智能助手，尽可能的调用工具回答用户的问题.提供的聊天历史包含与你对话用户的相关信息。摘要：{x['messages_summarized']['summary']}"
        if x['messages_summarized'].get("summary") else "无摘要",
) | chain_with_message_history

def get_last_user_after_assistant(history):
    # 反向遍历找到最后一个assistant的位置，并返回后面的所有user消息
    if not history:
        return None
    if history[-1]['role'] == 'assistant':
        return None
    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i]['role'] == 'assistant':
            last_assistant_idx = i
            break
    # 如果没有找到assistant
    if last_assistant_idx == -1:
        return history
    else:
        # 从assistant位置向后找第一个user
        return history[last_assistant_idx+1:]

# web页面中的核心函数
def add_message(chat_history, user_messages):
    # 将用户输入的消息添加到聊天记录中
    for m in user_messages['files']:
        print(m)
        chat_history.append({'role': 'user', 'content': {'path': m}})
    # 处理文本消息
    if user_messages['text'] is not None:
        chat_history.append({'role': 'user', 'content': user_messages['text']})
    # 返回更新后的历史和重置的输入框
    return chat_history, gr.MultimodalTextbox(value=None, interactive=False)

def submit_messages(history):
    # 提交用户输入的消息，生成机器人回复
    user_messages = get_last_user_after_assistant(history)
    print(user_messages)

def execute_chain(chat_history):
    input = chat_history[-1]
    result = final_chain.invoke({'input': input['content'], "config": {"configurable": {"session_id": "user123"}}},
                                 config={"configurable": {"session_id": "user123"}})
    chat_history.append({'role': 'assistant', 'content': result.content})
    return chat_history

with gr.Blocks(title='多模态聊天机器人', theme=gr.themes.Soft()) as block:
    # 聊天历史记录的组件
    chatbot = gr.Chatbot(height=500, label='聊天机器人')
    # 创建多模态输入框
    chat_input = gr.MultimodalTextbox(
        interactive=True, # 可交互
        file_types=['image', '.wav', '.mp4'],
        file_count='multiple',
        placeholder="请输入信息或者上传文件...",
        show_label=False,
        sources=['microphone', 'upload']
    )
    chat_input.submit(
        add_message,
        [chatbot, chat_input],
        [chatbot, chat_input],
    ).then(
        submit_messages,
        [chatbot],
        [chatbot],
    )


if __name__ == '__main__':
    block.launch()
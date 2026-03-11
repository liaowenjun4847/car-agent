import sys

# --- 💡 关键补丁：必须放在最顶部，甚至在 import streamlit 之前 ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
    
import streamlit as st
# 导入你 final 代码里的 run_agent 函数
from car_agent_final import run_agent

# --- 网页页面配置 ---
st.set_page_config(page_title="秦PLUS 智能管家", page_icon="🚗")

st.title("🚗 秦PLUS DM-i 智能技术顾问")
st.markdown("---")

# --- 初始化网页聊天记录 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 在网页上显示历史对话 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 用户输入框 ---
if prompt := st.chat_input("您可以问我：怎么开启定速巡航？或者：今天深圳天气适合洗车吗？"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用你的 final 代码进行思考和检索
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在查阅手册并分析天气..."):
            try:
                # 直接调用你 final 代码里的核心逻辑
                response = run_agent(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"发生了一点小意外: {e}")
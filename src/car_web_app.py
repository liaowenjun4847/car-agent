import sys
import os
import json
import logging
import random
import time

# --- 1. 核心补丁 (必须在最顶端) ---
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    try:
        __import__('pysqlite3_binary')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3_binary')
    except:
        pass

import streamlit as st

# --- 2. 网页配置 (防止白屏的关键) ---
st.set_page_config(
    page_title="车灵 AI - 全品牌智能专家",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. 资源初始化 (使用缓存防止重复连接) ---
@st.cache_resource
def init_clients():
    from openai import OpenAI
    from zhipuai import ZhipuAI
    import chromadb
    
    # 从 Secrets 安全读取
    try:
        dk = st.secrets["DEEPSEEK_KEY"]
        zk = st.secrets["ZHIPU_KEY"]
        # 云端部署建议使用相对路径
        db_p = "data/vector_db" 
        
        ai = OpenAI(api_key=dk, base_url="https://api.deepseek.com", timeout=60.0)
        zp = ZhipuAI(api_key=zk)
        
        # 数据库连接
        chroma = chromadb.PersistentClient(path=db_p)
        coll = chroma.get_or_create_collection(name="car_manual_zhipu")
        return ai, zp, coll
    except Exception as e:
        st.error(f"❌ 初始化失败: 请检查 Streamlit Secrets 是否配置正确。错误: {e}")
        st.stop()

# 执行初始化
ai_client, zp_client, collection = init_clients()

# 尝试导入工具类 (加入容错)
try:
    from tools import get_weather, TOOLS_DEFINITION
except ImportError:
    st.warning("⚠️ tools.py 导入失败，请检查该文件是否还引用了已删除的 config.py")
    get_weather = None
    TOOLS_DEFINITION = []

# --- 4. 核心 Agent 逻辑 ---
def run_agent(user_query):
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = [
            {"role": "system", "content": "你是一位精通多品牌的智能车机专家。你可以根据用户提到的品牌自动检索手册，并支持对比。"}
        ]
    
    history = st.session_state.conversation_history
    if len(history) > 10:
        history = [history[0]] + history[-9:]
    history.append({"role": "user", "content": user_query})

    # 第一步：意图识别
    intent_prompt = f"提取问题中的汽车品牌标签(如: tesla, qin_plus)，无品牌返'all'。只需返回标签名。问题: {user_query}"
    intent_res = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": intent_prompt}],
        temperature=0
    )
    detected_tag = intent_res.choices[0].message.content.strip().lower()

    # 第二步：思考与工具 (天气)
    response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        tools=TOOLS_DEFINITION if TOOLS_DEFINITION else None,
        tool_choice="auto" if TOOLS_DEFINITION else None
    )
    
    # ... (省略中间工具执行代码以保持简洁，逻辑同你之前的一致) ...

    # 第三步：RAG 检索
    try:
        res = zp_client.embeddings.create(model="embedding-3", input=user_query)
        q_vector = res.data[0].embedding
        search_params = {"query_embeddings": [q_vector], "n_results": 4}
        if detected_tag != 'all':
            search_params["where"] = {"car_model": detected_tag} # 简化匹配
        
        results = collection.query(**search_params)
        context = "\n---\n".join(results['documents'][0]) if results['documents'] else "未找到相关手册内容。"
    except:
        context = "检索服务暂不可用。"

    # 第四步：汇总
    final_messages = history.copy()
    final_messages.append({"role": "system", "content": f"参考资料：\n{context}"})
    final_res = ai_client.chat.completions.create(model="deepseek-chat", messages=final_messages)
    
    answer = final_res.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    return answer

# --- 5. UI 界面 ---
with st.sidebar:
    st.title("⚡ 车灵 AI 智库")
    st.markdown("---")
    st.info("🚗 已接入：比亚迪、特斯拉")
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

st.markdown("# ⚡ 车灵 **全品牌** 智能汽车专家")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("问我关于特斯拉或比亚迪的问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🚀 深度思考中..."):
            ans = run_agent(prompt)
            st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})

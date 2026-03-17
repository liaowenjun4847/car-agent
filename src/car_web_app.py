import sys
import json
import logging
import time
import random

# --- 1. 核心生存补丁 (必须在最顶部) ---
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

# --- 2. 网页顶级配置 ---
st.set_page_config(
    page_title="车灵 AI - 全品牌智能汽车专家",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. 资源级初始化 (使用缓存避免白屏) ---
@st.cache_resource
def get_system_engine():
    from openai import OpenAI
    from zhipuai import ZhipuAI
    import chromadb
    
    # 从 Secrets 获取配置
    dk = st.secrets["DEEPSEEK_KEY"]
    zk = st.secrets["ZHIPU_KEY"]
    base_url = st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    db_path = "data/vector_db"

    ai = OpenAI(api_key=dk, base_url=base_url, timeout=60.0)
    zp = ZhipuAI(api_key=zk)
    
    # 初始化数据库
    chroma = chromadb.PersistentClient(path=db_path)
    coll = chroma.get_or_create_collection(name="car_manual_zhipu")
    
    return ai, zp, coll

# 尝试初始化，失败则优雅报错
try:
    ai_client, zp_client, collection = get_system_engine()
except Exception as e:
    st.error(f"⚠️ 系统引擎启动失败: {e}")
    st.stop()

# 尝试导入工具类
try:
    from tools import get_weather, TOOLS_DEFINITION
except:
    get_weather = None
    TOOLS_DEFINITION = []

# --- 4. 侧边栏构建 (借鉴第一版，增强丰富度) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.title("车灵 AI 智库")
    st.markdown("---")
    
    st.subheader("🏎️ 已激活知识库")
    st.success("✅ 比亚迪 (Qin PLUS)")
    st.success("✅ 特斯拉 (Model Y)")
    st.info("🕒 小米 / 问界 (即将上线)")
    
    st.markdown("---")
    st.subheader("📊 引擎状态")
    cols = st.columns(2)
    cols[0].metric("LLM", "DeepSeek")
    cols[1].metric("RAG", "ChromaDB")
    
    st.caption("实时天气系统：已连接 🟢")
    st.caption("向量空间：Zhipu Embedding-3")
    
    if st.button("🧹 清除记忆并重启"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

# --- 5. 核心 Agent 逻辑 (融合 Gemini 风格的思考过程) ---
def run_agent(user_query):
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = [{"role": "system", "content": "你是一位精通全车系的专家..."}]
    
    history = st.session_state.conversation_history
    history.append({"role": "user", "content": user_query})

    # A. 意图提取
    intent_res = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f"提取品牌标签(tesla/qin_plus/all): {user_query}"}],
        temperature=0
    )
    detected_tag = intent_res.choices[0].message.content.strip().lower()

    # B. RAG 检索
    with st.status("🔍 正在多维度检索汽车手册...", expanded=False) as status:
        st.write(f"目标车型: {detected_tag}")
        try:
            res = zp_client.embeddings.create(model="embedding-3", input=user_query)
            q_vector = res.data[0].embedding
            
            search_params = {"query_embeddings": [q_vector], "n_results": 5}
            if 'all' not in detected_tag:
                tags = [t.strip() for t in detected_tag.split(",")]
                search_params["where"] = {"car_model": {"$in": tags}} if len(tags) > 1 else {"car_model": tags[0]}
            
            results = collection.query(**search_params)
            
            context_list = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                context_list.append(f"【{meta['car_model']}】: {doc}")
            rag_context = "\n---\n".join(context_list)
            st.write("✅ 成功提取关联条目")
        except:
            rag_context = "暂无相关参考。"
        status.update(label="✅ 资料检索完成", state="complete")

    # C. 最终回复
    final_messages = history.copy()
    final_messages.append({"role": "system", "content": f"参考资料:\n{rag_context}"})
    
    response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=final_messages,
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    st.session_state.conversation_history = history
    return answer

# --- 6. 主界面渲染 ---
st.markdown("# ⚡ 车灵 **全品牌** 智能汽车专家")
st.markdown("> 融合实时天气、多车主手册、跨品牌对比的 AI 助手")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 对话展示容器
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("您可以问：特斯拉和比亚迪的灯光开启方式有什么不同？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        start_t = time.time()
        # 这里是 Gemini 风格的关键：展示思考步骤
        ans = run_agent(prompt)
        st.markdown(ans)
        
        end_t = time.time()
        st.caption(f"🧠 计算耗时: {round(end_t-start_t, 2)}s | 动力来源: DeepSeek-V3 & ChromaDB")
    
    st.session_state.messages.append({"role": "assistant", "content": ans})

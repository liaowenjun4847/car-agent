import sys
import json
import time
import streamlit as st

# --- 1. 核心生存补丁 ---
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    try:
        __import__('pysqlite3_binary')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3_binary')
    except:
        pass

# --- 2. 顶级配置 ---
st.set_page_config(page_title="车灵 AI - 智慧全能专家", page_icon="⚡", layout="wide")

# --- 3. 资源初始化 ---
@st.cache_resource
def get_system_engine():
    from openai import OpenAI
    from zhipuai import ZhipuAI
    import chromadb
    ai = OpenAI(api_key=st.secrets["DEEPSEEK_KEY"], base_url="https://api.deepseek.com")
    zp = ZhipuAI(api_key=st.secrets["ZHIPU_KEY"])
    chroma = chromadb.PersistentClient(path="data/vector_db")
    coll = chroma.get_or_create_collection(name="car_manual_zhipu")
    return ai, zp, coll

ai_client, zp_client, collection = get_system_engine()

# --- 4. 核心功能函数 ---

def get_dynamic_suggestions(history):
    """Gemini 风格：根据对话历史预测后续问题"""
    try:
        context = [h for h in history if h['role'] != 'system'][-3:] # 取最近3轮
        prompt = f"根据以下对话内容，预测用户接下来最可能问的3个简洁问题（汽车相关）。只需返回JSON列表，格式: ['问题1', '问题2', '问题3']。对话记录: {context}"
        res = ai_client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.7)
        return json.loads(res.choices[0].message.content)
    except:
        return ["特斯拉和比亚迪的续航对比", "如何保养新能源车电池？", "介绍一下最新的自动驾驶技术"]

def run_agent(user_query):
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = [{"role": "system", "content": "你是一位百科全书式的汽车专家。优先使用检索资料，若资料不足，请调用你自有的知识库给出专业解答。"}]
    
    history = st.session_state.conversation_history
    history.append({"role": "user", "content": user_query})

    # A. RAG 检索过程
    with st.status("🔍 正在跨时空检索资料...", expanded=False) as status:
        try:
            # 意图识别
            intent_res = ai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"提取品牌(tesla/qin_plus/all): {user_query}"}],
                temperature=0
            )
            tag = intent_res.choices[0].message.content.strip().lower()
            
            # 向量检索
            emb = zp_client.embeddings.create(model="embedding-3", input=user_query).data[0].embedding
            search_args = {"query_embeddings": [emb], "n_results": 4}
            if 'all' not in tag:
                search_args["where"] = {"car_model": tag}
            
            results = collection.query(**search_args)
            rag_context = "\n".join(results['documents'][0]) if results['documents'][0] else "知识库未覆盖此细节。"
            status.update(label="✅ 知识库检索完成", state="complete")
        except:
            rag_context = "正在切换至全能大模型模式..."
            status.update(label="🌐 正在使用大模型通用知识", state="complete")

    # B. 最终生成 (整合 RAG + LLM 知识)
    final_messages = history.copy()
    final_messages.append({"role": "system", "content": f"已知手册资料（仅供参考）: {rag_context}\n如果资料不足，请用你的通用汽车知识回答，不要说'我不知道'。"})
    
    response = ai_client.chat.completions.create(model="deepseek-chat", messages=final_messages, temperature=0.5)
    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    
    # C. 更新建议词
    st.session_state.suggestions = get_dynamic_suggestions(history)
    return answer

# --- 5. UI 界面 ---
with st.sidebar:
    st.title("⚡ 车灵 AI 智库")
    st.info("🚗 核心库：比亚迪 / 特斯拉")
    st.metric("核心大脑", "DeepSeek-V3")
    if st.button("🗑️ 开启新对话"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.suggestions = ["比亚迪秦PLUS的优点是什么？", "特斯拉Model Y的辅助驾驶怎么用？", "今天适合洗车吗？"]
        st.rerun()

st.markdown("# ⚡ 车灵 **全能** 汽车助手")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.suggestions = ["比亚迪秦PLUS的优点是什么？", "特斯拉Model Y的辅助驾驶怎么用？", "北京现在的天气适合开车吗？"]

# 渲染对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 渲染动态建议按钮 (Gemini 标志性设计)
cols = st.columns(len(st.session_state.suggestions))
for i, suggestion in enumerate(st.session_state.suggestions):
    if cols[i].button(f"💡 {suggestion}", use_container_width=True):
        st.session_state.pushed_suggestion = suggestion

# 处理输入
input_placeholder = "您可以问："+st.session_state.suggestions[0]+"..."
if prompt := (st.chat_input(input_placeholder) or st.session_state.get("pushed_suggestion")):
    st.session_state.pushed_suggestion = None # 清空临时点击
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        ans = run_agent(prompt)
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun() # 强制刷新以更新输入框的 placeholder 和建议按钮

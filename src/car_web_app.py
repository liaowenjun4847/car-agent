import sys
import json
import logging
import os
import tempfile
import streamlit as st

# --- 1. 环境补丁与初始化 ---
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # 如果找不到 pysqlite3，尝试导入 pysqlite3-binary
    try:
        __import__('pysqlite3_binary')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3_binary')
    except ImportError:
        pass

from openai import OpenAI
from zhipuai import ZhipuAI
import chromadb
from chromadb.config import Settings
from tools import get_weather, TOOLS_DEFINITION
DEEPSEEK_KEY = st.secrets["DEEPSEEK_KEY"]
DEEPSEEK_BASE_URL = st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
ZHIPU_KEY = st.secrets["ZHIPU_KEY"]
DB_PATH = "data/vector_db"

ai_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE_URL, timeout=60.0)
zp_client = ZhipuAI(api_key=ZHIPU_KEY)

# 💡 使用原本的路径或持久化路径，确保能读到你刚才跑完的 pro 数据库
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="car_manual_zhipu")

# --- 2. 全局对话记忆 ---
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "你是一位精通多品牌的智能车机专家。你可以根据用户提到的品牌（如比亚迪、特斯拉）自动检索对应手册，并支持多车型对比。"}
    ]

def run_agent(user_query):
    history = st.session_state.conversation_history
    if len(history) > 10:
        history = [history[0]] + history[-9:]

    history.append({"role": "user", "content": user_query})

    # --- 第一步：车型意图识别 (核心新增) ---
    # 让 DeepSeek 帮忙判断用户是在问哪辆车，或者是想对比
    # 比如用户问“特斯拉怎么开”，它会提取出 "tesla"
    intent_prompt = f"""
    请从用户的问题中提取提到的汽车品牌或车型标签（例如: tesla, Qin_PLUS, xiaomi）。
    如果提到多个，请用逗号分隔。如果没有提到具体品牌，请返回 'all'。
    用户问题："{user_query}"
    只需返回标签名，不要任何解释。
    """
    
    intent_res = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": intent_prompt}],
        temperature=0
    )
    detected_tag = intent_res.choices[0].message.content.strip().lower()
    logging.info(f"🔍 意图识别结果: {detected_tag}")

    # --- 第二步：思考与工具调用 (天气等) ---
    response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        tools=TOOLS_DEFINITION,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    if response_message.tool_calls:
        history.append(response_message)
        for tool_call in response_message.tool_calls:
            function_args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "get_weather":
                weather_info = get_weather(function_args.get("city"))
                history.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(weather_info, ensure_ascii=False)})

    # --- 第三步：多维度 RAG 检索 ---
    try:
        res = zp_client.embeddings.create(model="embedding-3", input=user_query)
        q_vector = res.data[0].embedding
        
        search_params = {"query_embeddings": [q_vector], "n_results": 4}
        
        # 💡 根据意图决定检索范围
        if detected_tag != 'all':
            # 如果识别到了具体车型，精准搜索该车型
            # 支持多车型对比，如果是 "tesla, Qin_PLUS"，用 $in 操作符
            tags = [t.strip() for t in detected_tag.split(",")]
            if len(tags) > 1:
                search_params["where"] = {"car_model": {"$in": tags}}
            else:
                search_params["where"] = {"car_model": tags[0]}
        
        results = collection.query(**search_params)
        
        if results['documents'] and results['documents'][0]:
            # 💡 在 context 中加入标签信息，让 AI 知道哪段话是谁说的
            context_list = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                context_list.append(f"【车型:{meta['car_model']}】: {doc}")
            rag_context = "\n---\n".join(context_list)
        else:
            rag_context = "知识库中暂无相关具体说明。"
            
    except Exception as e:
        logging.error(f"检索失败: {e}")
        rag_context = "检索服务繁忙。"

    # --- 第四步：最终汇总 (跨语言/跨车总结) ---
    final_messages = history.copy()
    final_messages.append({
        "role": "system",
        "content": f"请结合以下检索到的多车型参考资料回答。如果是英文资料请翻译成中文。如果是对比问题，请清晰指出不同车型的差异。\n{rag_context}"
    })

    final_response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=final_messages,
        temperature=0.3
    )
    
    answer = final_response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    st.session_state.conversation_history = history
    return answer

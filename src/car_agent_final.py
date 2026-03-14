import sys
import json
import logging
import os
import tempfile
import streamlit as st

# --- 1. 解决环境补丁 ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from openai import OpenAI
from zhipuai import ZhipuAI
import chromadb
from chromadb.config import Settings
from config import DEEPSEEK_KEY, DEEPSEEK_BASE_URL, ZHIPU_KEY, DB_PATH
from tools import get_weather, TOOLS_DEFINITION

# 💡 关键：引入 read_manual 中的路径，实现单一源管理
from read_manual import pdf_dir

# --- 2. 配置日志系统 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- 3. 初始化引擎 ---
ai_client = OpenAI(
    api_key=DEEPSEEK_KEY, 
    base_url=DEEPSEEK_BASE_URL,
    timeout=60.0
)
zp_client = ZhipuAI(api_key=ZHIPU_KEY)

# 使用临时目录解决 Permission denied
tmp_db_path = os.path.join(tempfile.gettempdir(), "chroma_db_storage")

chroma_client = chromadb.PersistentClient(
    path=tmp_db_path,
    settings=Settings(anonymized_telemetry=False)
)

# 💡 注意：这里必须和 build_knowledge_base_pro.py 中的名字一致
collection = chroma_client.get_or_create_collection(name="car_manual_zhipu")

# --- 4. 全局对话记忆 ---
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "你是一位专业的智能车机管家。你擅长结合实时天气工具和车主手册为用户提供精准建议。"}
    ]

def run_agent(user_query):
    # 💡 自动识别当前车型标签 (如 Qin_PLUS)
    current_model_tag = os.path.basename(os.path.normpath(pdf_dir))
    
    history = st.session_state.conversation_history
    
    # 记忆管理
    if len(history) > 10:
        history = [history[0]] + history[-9:]

    history.append({"role": "user", "content": user_query})
    logging.info(f"用户提问: {user_query} (针对车型: {current_model_tag})")

    # --- 第一步：思考阶段 ---
    response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        tools=TOOLS_DEFINITION,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # --- 第二步：执行阶段 ---
    if tool_calls:
        logging.info("🤖 Agent 决策：调用外部工具...")
        history.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "get_weather":
                city = function_args.get("city")
                weather_info = get_weather(city)
                
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(weather_info, ensure_ascii=False)
                })

    # --- 第三步：检索阶段 (RAG) ---
    try:
        res = zp_client.embeddings.create(model="embedding-3", input=user_query)
        q_vector = res.data[0].embedding
        
        # 💡 增强后的检索逻辑：加入 where 过滤
        if collection.count() > 0:
            results = collection.query(
                query_embeddings=[q_vector], 
                n_results=2,
                # 💡 关键：只检索当前设置的车型数据
                where={"car_model": current_model_tag} 
            )
            
            if results['documents'] and results['documents'][0]:
                rag_context = "\n---\n".join(results['documents'][0])
            else:
                rag_context = f"未找到关于 {current_model_tag} 的具体手册说明。"
        else:
            rag_context = "知识库为空，请根据通用知识回答。"
    except Exception as e:
        logging.error(f"RAG检索失败: {e}")
        rag_context = "检索服务暂时不可用。"

    # --- 第四步：汇总阶段 ---
    final_messages = history.copy()
    final_messages.append({
        "role": "system",
        "content": f"【当前车型】：{current_model_tag}\n【车主手册参考】：\n{rag_context}\n请结合车型信息给出最终回答。"
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
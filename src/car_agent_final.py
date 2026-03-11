import sys
import json
import logging
import os

# --- 1. 解决 Rocky Linux 环境补丁 ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from openai import OpenAI
from zhipuai import ZhipuAI
import chromadb
# 💡 导入 Settings 用于关闭干扰请求
from chromadb.config import Settings
from config import DEEPSEEK_KEY, DEEPSEEK_BASE_URL, ZHIPU_KEY, DB_PATH
from tools import get_weather, TOOLS_DEFINITION

# --- 2. 配置专业日志系统 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("project.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- 3. 初始化所有引擎 ---
# 💡 增加 timeout 设置，应对网络抖动造成的连接错误
ai_client = OpenAI(
    api_key=DEEPSEEK_KEY, 
    base_url=DEEPSEEK_BASE_URL,
    timeout=60.0  # 增加超时限制到 60 秒
)
zp_client = ZhipuAI(api_key=ZHIPU_KEY)

# 💡 核心修改：增加 Settings 强制关闭 ChromaDB 匿名统计请求，解决 Errno -2 报错
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_collection(name="car_manual_zhipu")

# --- 4. 初始化全局对话记忆 ---
conversation_history = [
    {
        "role": "system", 
        "content": "你是一位专业的秦PLUS DM-i智能管家。你擅长结合实时天气工具和车主手册为用户提供精准建议。"
    }
]

def run_agent(user_query):
    global conversation_history
    
    # 💡 记忆管理：如果记忆过长，清理早期对话（保留 System 指令），防止请求包过大导致超时
    if len(conversation_history) > 10:
        conversation_history = [conversation_history[0]] + conversation_history[-9:]

    conversation_history.append({"role": "user", "content": user_query})
    logging.info(f"用户提问: {user_query}")

    # --- 第一步：思考阶段 ---
    response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=conversation_history,
        tools=TOOLS_DEFINITION,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # --- 第二步：执行阶段 ---
    if tool_calls:
        logging.info("🤖 Agent 决策：需要调用外部工具获取信息...")
        conversation_history.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "get_weather":
                city = function_args.get("city")
                weather_info = get_weather(city)
                
                # 将工具执行结果存入记忆
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(weather_info, ensure_ascii=False)
                })
                logging.info(f"工具返回结果: {weather_info}")

    # --- 第三步：检索阶段 ---
    res = zp_client.embeddings.create(model="embedding-3", input=user_query)
    q_vector = res.data[0].embedding
    
    results = collection.query(query_embeddings=[q_vector], n_results=2)
    rag_context = "\n---\n".join(results['documents'][0])
    
    temp_messages = conversation_history.copy()
    temp_messages.append({
        "role": "system",
        "content": f"【车主手册参考资料】：\n{rag_context}\n请结合实时工具信息和手册资料给出最终回答。"
    })

    # --- 第四步：汇总阶段 ---
    final_response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=temp_messages,
        temperature=0.3
    )
    
    answer = final_response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})
    
    return answer

if __name__ == "__main__":
    print("="*40)
    print("🚗 秦PLUS 工业级 Agent 助手 (V3.1 稳定版) 已上线！")
    print("已关闭外部遥测干扰，优化了网络超时处理")
    print("="*40)
    
    while True:
        try:
            q = input("\n👤 用户: ")
            if q.lower() in ['exit', 'quit']:
                print("再见，文隽！祝你面试顺利！")
                break
            if not q.strip(): continue
            
            print("🔍 系统思考中...")
            result = run_agent(q)
            print(f"\n🤖 [秦PLUS管家]: {result}")
            
        except Exception as e:
            logging.error(f"发生错误: {e}")
            # 如果是连接报错，给出更友好的提示
            if "Connection error" in str(e):
                print("❌ 提示：网络连接超时，请检查您的虚拟机网络环境或重试。")
            else:
                print(f"❌ 抱歉，遇到了一点小麻烦: {e}")
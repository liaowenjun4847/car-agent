import sys
import json
# 解决 Rocky Linux 环境补丁
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from openai import OpenAI
from zhipuai import ZhipuAI
import chromadb
from config import DEEPSEEK_KEY, DEEPSEEK_BASE_URL, ZHIPU_KEY, DB_PATH
from tools import get_weather, TOOLS_DEFINITION

# 1. 初始化引擎
ai_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE_URL)
zp_client = ZhipuAI(api_key=ZHIPU_KEY)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="car_manual_zhipu")

def run_agent(user_query):
    # 第一步：初步对话，把工具定义传给 DeepSeek
    messages = [
        {"role": "system", "content": "你是一位秦PLUS智能管家。你可以查询天气并结合手册给出建议。"},
        {"role": "user", "content": user_query}
    ]
    
    # 询问 DeepSeek：是否需要调用工具？
    response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS_DEFINITION, # 传入工具说明书
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 第二步：如果模型决定调用工具
    if tool_calls:
        print("🤖 Agent 思考中：需要获取实时信息...")
        for tool_call in tool_calls:
            # 解析模型想查询的城市
            function_args = json.loads(tool_call.function.arguments)
            city = function_args.get("city")
            
            # 真正执行 Python 函数
            weather_data = get_weather(city)
            
            # 将工具结果存入消息记录，准备二次回复
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(weather_data)
            })

    # 第三步：结合工具结果和 RAG 知识库做最终回答
    # 先做一次 RAG 检索（这一步让 Agent 更有深度）
    res = zp_client.embeddings.create(model="embedding-3", input=user_query)
    q_vector = res.data[0].embedding
    docs = collection.query(query_embeddings=[q_vector], n_results=2)['documents'][0]
    rag_context = "\n".join(docs)

    messages.append({
        "role": "system", 
        "content": f"这是从车主手册检索到的相关知识：\n{rag_context}\n请结合以上知识和刚才查询到的工具信息给用户最终回复。"
    })

    final_response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    return final_response.choices[0].message.content

if __name__ == "__main__":
    print("🚀 秦PLUS Agent 智能体版已上线！")
    while True:
        q = input("\n用户咨询: ")
        if q.lower() in ['exit', 'quit']: break
        print(f"\n[Agent]: {run_agent(q)}")
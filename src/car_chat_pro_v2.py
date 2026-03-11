import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
from openai import OpenAI
from zhipuai import ZhipuAI
import chromadb
from config import DEEPSEEK_KEY, DEEPSEEK_BASE_URL, ZHIPU_KEY, DB_PATH

# 1. 初始化两个客户端
ai_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE_URL)
zp_client = ZhipuAI(api_key=ZHIPU_KEY)

# 2. 连接智谱版向量库
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="car_manual_zhipu")

def get_answer(question):
    # 3. 核心升级：检索时先将提问向量化
    response = zp_client.embeddings.create(
        model="embedding-3",
        input=question
    )
    query_vector = response.data[0].embedding

    # 4. 执行语义搜索
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )
    
    context = "\n---\n".join(results['documents'][0])

    # 5. 组合 Prompt 调用 DeepSeek
    prompt = f"你是一位秦PLUS专家。根据手册内容回答问题：\n{context}\n问题：{question}"
    
    chat_response = ai_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat_response.choices[0].message.content

if __name__ == "__main__":
    print("🚗 秦PLUS 语义级 RAG 助手已就位！")
    while True:
        q = input("\n用户咨询: ")
        if q.lower() in ['exit', 'quit']: break
        print("🔍 正在语义检索并生成回答...")
        print(f"\n[AI技术顾问]: {get_answer(q)}")
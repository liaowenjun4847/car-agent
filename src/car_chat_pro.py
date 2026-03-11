import sys
# 1. 解决 Rocky Linux 的 sqlite3 版本问题
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
from openai import OpenAI
import chromadb
from config import DEEPSEEK_KEY, DEEPSEEK_BASE_URL, DB_PATH

# 初始化 DeepSeek 客户端
ai_client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE_URL)

# 2. 获取数据库集合（必须保持 embedding_function=None）
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(
    name="car_manual",
    embedding_function=None 
)

def get_answer(question):
    # --- 核心修改：手动提供检索向量 ---
    # 既然数据库里存的是 384 维的 [0.0...], 我们检索时也传一个同样的假向量
    # 这样 Chroma 就会跳过下载，直接进行内部的“文本回退”匹配
    dummy_query_embedding = [[0.0] * 384]
    
    results = collection.query(
        query_embeddings=dummy_query_embedding, # 传向量而不是传 query_texts
        query_texts=[question],                 # 同时传文字，触发文本搜索
        n_results=5
    )
    
    # 后面提取背景知识和调用 DeepSeek 的逻辑保持不变
    relevant_texts = results['documents'][0] if results['documents'] else []
    context = "\n---\n".join(relevant_texts) if relevant_texts else "手册中未找到直接匹配的描述。"

    # 4. 调用 DeepSeek
    system_prompt = "你是一位秦PLUS DM-i 技术专家。请根据参考内容严谨回答。如果内容无关，请告知并给出常识性建议。"
    user_content = f"【手册参考内容】:\n{context}\n\n【用户问题】: {question}"

    try:
        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"接口连接失败: {e}"

if __name__ == "__main__":
    print("🚗 秦PLUS 智能助手（纯文本版）已上线！")
    print("提示：当前使用关键词检索模式，建议提问时带有具体词汇（如：强制纯电、胎压）。")
    while True:
        query = input("\n用户咨询: ")
        if query.lower() in ['quit', 'exit']: break
        if not query.strip(): continue
        
        print("🔍 正在查阅手册...")
        answer = get_answer(query)
        print(f"\n[AI技术顾问]: {answer}")
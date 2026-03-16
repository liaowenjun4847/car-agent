import sys
import os

# 1. 解决 Rocky Linux 环境补丁
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from zhipuai import ZhipuAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import ZHIPU_KEY, DB_PATH
# 💡 引入 read_manual 里的自动扫描函数
from read_manual import get_all_car_manuals

# 初始化智谱客户端
client = ZhipuAI(api_key=ZHIPU_KEY)

def get_embedding(text):
    """调用智谱 embedding-3 模型生成向量"""
    response = client.embeddings.create(
        model="embedding-3", 
        input=text
    )
    return response.data[0].embedding

def build_pro_db():
    # 初始化 ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    # 创建智谱专用集合
    collection = chroma_client.get_or_create_collection(name="car_manual_zhipu")

    # 💡 核心改动：获取所有手册的清单
    manual_list = get_all_car_manuals()
    
    if not manual_list:
        print("❌ 错误：没有发现任何可入库的 PDF 文件")
        return

    print(f"📚 准备开始批量入库，共发现 {len(manual_list)} 本手册。")

    # 💡 循环处理每一本手册
    for manual in manual_list:
        car_model_tag = manual['tag']
        pdf_path = manual['path']
        filename = manual['filename']

        print(f"\n--- 🚀 正在处理品牌/车型: [{car_model_tag}] ---")
        print(f"📂 文件路径: {pdf_path}")

        try:
            reader = PdfReader(pdf_path)
            # 提取全文
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            # 文本切分
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
            chunks = splitter.split_text(text)

            print(f"📝 成功切分为 {len(chunks)} 个知识块，准备生成向量...")

            for i, chunk in enumerate(chunks):
                try:
                    vector = get_embedding(chunk)
                    collection.add(
                        documents=[chunk],
                        embeddings=[vector],
                        # 💡 核心：每块内容都打上所属品牌的标签
                        metadatas=[{"car_model": car_model_tag}],
                        # 💡 ID 也要带上标签，防止不同车型的 ID 冲突
                        ids=[f"id_{car_model_tag}_{i}"]
                    )
                    if i % 50 == 0:
                        print(f"⏳ [{car_model_tag}] 进度: {i}/{len(chunks)}")
                except Exception as e:
                    print(f"❌ 块 {i} 处理失败: {e}")
            
            print(f"✅ 完成！车型 {car_model_tag} 已存入数据库。")

        except Exception as e:
            print(f"❌ 处理文件 {filename} 时发生崩溃: {e}")
            continue

    print("\n✨✨✨ 所有品牌数据构建完成！全库已就绪。 ✨✨✨")

if __name__ == "__main__":
    build_pro_db()
import sys
# 1. 解决 Rocky Linux 环境补丁
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import chromadb
from zhipuai import ZhipuAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import ZHIPU_KEY, DB_PATH, PDF_DIR

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
    # 创建智谱专用集合，避免与之前的混淆
    collection = chroma_client.get_or_create_collection(name="car_manual_zhipu")

    # 2. 读取并解析 PDF
    # 确保文件名与你 data/raw_docs/ 下的完全一致
    pdf_path = os.path.join(PDF_DIR, "2026款秦PLUS DM-i用户手册20260120.pdf") 
    
    if not os.path.exists(pdf_path):
        print(f"❌ 错误：找不到文件 {pdf_path}")
        return

    print("正在解析 PDF...")
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # 3. 文本切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    chunks = splitter.split_text(text)

    # 4. 批量生成向量并入库
    print(f"开始为 {len(chunks)} 个知识块生成语义向量（使用智谱 API）...")
    
    for i, chunk in enumerate(chunks):
        try:
            vector = get_embedding(chunk)
            collection.add(
                documents=[chunk],
                embeddings=[vector],
                ids=[f"id_zp_{i}"]
            )
            if i % 20 == 0:
                print(f"🚀 进度: {i}/{len(chunks)}")
        except Exception as e:
            # 记录详细错误，不再盲目跳过
            print(f"❌ 块 {i} 处理失败: {e}")
            continue

    print("✅ 大厂标准：基于智谱语义向量的数据库构建完成！")

if __name__ == "__main__":
    build_pro_db()
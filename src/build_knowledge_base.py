import sys
# 1. 解决 Rocky Linux 的 sqlite3 版本问题
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from config import DB_PATH, PDF_DIR

def build_pure_text_db():
    # 初始化 Chroma 客户端
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 【关键点】显式设置 embedding_function=None，禁止下载模型
    collection = client.get_or_create_collection(
        name="car_manual",
        embedding_function=None
    )

    # 读取 PDF 并切分（复用你之前的逻辑）
    all_chunks = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    print(f"正在处理 {len(pdf_files)} 个 PDF 文件...")
    for file in pdf_files:
        reader = PdfReader(os.path.join(PDF_DIR, file))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)

    # 存入数据库
    if all_chunks:
        print(f"成功切分出 {len(all_chunks)} 个知识块，正在存入数据库...")
    
    # 核心变动：手动构造一串“假向量”
    # 我们给每个块都配一个长度为 384 的 [0.0, 0.0...] 列表
    # 这样 ChromaDB 发现你已经提供了向量，就不会去下载模型了！
    dummy_embeddings = [[0.0] * 384 for _ in range(len(all_chunks))]
    
    collection.add(
        documents=all_chunks,
        embeddings=dummy_embeddings,  # 喂给它假向量，堵住它的下载嘴
        ids=[f"id_{i}" for i in range(len(all_chunks))]
    )
    print("✅ 纯文本知识库构建完成！已通过虚拟向量绕过模型下载。")

if __name__ == "__main__":
    build_pure_text_db()
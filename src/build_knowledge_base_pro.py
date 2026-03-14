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
# 💡 引入 read_manual 里的 pdf_dir，实现单一路径管理
from read_manual import pdf_dir

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

    # 💡 自动识别当前车型标签 (取子文件夹的名字，如 Qin_PLUS)
    car_model_tag = os.path.basename(os.path.normpath(pdf_dir))

    # 2. 读取并解析 PDF
    files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not files:
        print(f"❌ 错误：在 {pdf_dir} 下找不到任何 PDF 文件")
        return

    pdf_path = os.path.join(pdf_dir, files[0])
    print(f"📂 正在解析车型 [{car_model_tag}] 的手册: {files[0]}...")

    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # 3. 文本切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    chunks = splitter.split_text(text)

    # 4. 批量生成向量并入库
    print(f"🚀 开始为 {len(chunks)} 个知识块生成向量，标签为: {car_model_tag}")
    
    for i, chunk in enumerate(chunks):
        try:
            vector = get_embedding(chunk)
            collection.add(
                documents=[chunk],
                embeddings=[vector],
                # 💡 增加 metadata，以后多车型检索全靠它
                metadatas=[{"car_model": car_model_tag}],
                ids=[f"id_{car_model_tag}_{i}"]
            )
            if i % 20 == 0:
                print(f"⏳ 进度: {i}/{len(chunks)}")
        except Exception as e:
            print(f"❌ 块 {i} 处理失败: {e}")
            continue

    print(f"✅ 构建完成！车型 {car_model_tag} 的数据已成功打标入库。")

if __name__ == "__main__":
    build_pro_db()

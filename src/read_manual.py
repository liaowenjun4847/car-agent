import os
from pypdf import PdfReader

# 💡 以后要换车型或改路径，只需要改下面这一行！
pdf_dir = "/home/LiaoWenjun/car_ai_project/data/raw_docs/Qin_PLUS/"

def check_and_read():
    # 检查文件夹是否存在 
    if not os.path.exists(pdf_dir):
        print(f"❌ 错误：找不到目录 {pdf_dir}，请检查路径是否正确。")
        return

    files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    # 检查文件夹内是否有文件 
    if not files:
        print(f"⚠️ 警告：目录 {pdf_dir} 下没有 PDF 文件，请上传手册。")
        return

    pdf_file = os.path.join(pdf_dir, files[0])
    
    try:
        # 鲁棒性体现：增加异常捕捉
        reader = PdfReader(pdf_file)
        print(f"✅ 成功加载手册: {files[0]}")
        print(f"📄 总页数: {len(reader.pages)}")
        
        # 尝试读取内容
        page_text = reader.pages[15].extract_text()
        print("\n--- 内容预览 ---")
        print(page_text[:200])
        
    except Exception as e:
        print(f"❌ 解析 PDF 时发生错误: {e}")

if __name__ == "__main__":
    check_and_read()

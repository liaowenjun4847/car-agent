import os
from pypdf import PdfReader

# 💡 现在我们定义总的根目录，代码会自动往下钻
base_raw_dir = "/home/LiaoWenjun/car_ai_project/data/raw_docs/"

def get_all_car_manuals():
    """
    扫描 base_raw_dir 下的所有 PDF，并根据文件夹名字自动打标签
    返回一个列表，包含：文件路径、文件名、所属品牌/车型(标签)
    """
    manuals = []
    
    if not os.path.exists(base_raw_dir):
        print(f"❌ 错误：找不到总目录 {base_raw_dir}")
        return manuals

    # 使用 os.walk 递归遍历所有子文件夹
    for root, dirs, files in os.walk(base_raw_dir):
        for file in files:
            if file.endswith('.pdf'):
                full_path = os.path.join(root, file)
                # 🏷️ 自动提取文件夹名字作为标签
                # 例如：/data/raw_docs/tesla/model_y.pdf -> 标签就是 tesla
                tag = os.path.basename(root)
                
                manuals.append({
                    "path": full_path,
                    "filename": file,
                    "tag": tag
                })
    
    return manuals

def check_all_manuals():
    manuals = get_all_car_manuals()
    
    if not manuals:
        print(f"⚠️ 警告：在 {base_raw_dir} 下没找到任何 PDF 文件。")
        return

    print(f"🔍 扫描完成！共发现 {len(manuals)} 本手册：")
    for m in manuals:
        try:
            reader = PdfReader(m['path'])
            print(f"✅ 品牌标签: [{m['tag']}] | 文件: {m['filename']} | 页数: {len(reader.pages)}")
        except Exception as e:
            print(f"❌ 无法读取 {m['filename']}: {e}")

if __name__ == "__main__":
    # 运行测试，看看它能不能把你现在的 byd 和 tesla 都扫描出来
    check_all_manuals()
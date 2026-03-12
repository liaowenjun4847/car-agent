🚗 秦PLUS DM-i 智能技术顾问
本项目是一款集成车主手册检索、实时天气分析与大语言模型 (LLM) 交互的智能车载助手系统。用户可以通过自然语言咨询车辆操作规范，并获取结合实时环境的驾驶建议。
在线演示地址: [👉 点击这里立即访问 (https://auto-mind.streamlit.app/)]

🌟 核心亮点
智能知识检索 (RAG)：针对《秦PLUS DM-i 车主手册》构建向量数据库，支持精准的说明书内容定位。
RAG 容错机制：设计了双层检索架构。当数据库无法精准匹配时，自动触发 AI 专家模式，结合模型内置知识库为用户提供购车建议。
动态可视化分析：采用 Streamlit + Matplotlib 实时生成销量趋势与价格分布图表。
安全工程化设计：通过 secrets.toml 实现 API 密钥与数据库凭证的解耦，符合企业级安全开发规范。
🚗 秦PLUS DM-i 智能技术顾问
本项目是一款集成车主手册检索、实时天气分析与大语言模型 (LLM) 交互的智能车载助手系统。用户可以通过自然语言咨询车辆操作规范，并获取结合实时环境的驾驶建议。
在线演示地址: [👉 点击这里立即访问 ()]

🌟 核心亮点



实时工具联动：自主触发天气接口调用，根据降雨、温度等实时数据提供洗车及出行建议。

云端兼容性设计：采用临时文件存储方案 (tempfile)，完美解决 Streamlit 云端的权限冲突问题。

工业级安全设计：通过 secrets.toml 实现 API 密钥的加密解密，符合企业级安全开发规范。

🛠️ 技术栈

前端展示：Streamlit (Python 原生 Web 框架)

大模型后端：DeepSeek API (OpenAI 兼容协议)

向量存储：ChromaDB (Persistent 模式)

语义分析：ZhipuAI Embedding-3 引擎

数据处理：Pandas, PyPDF2

📁 项目结构

├── src/
│   ├── car_web_app.py      # 主程序：Streamlit 网页逻辑与 AI 交互
│   ├── car_agent_final.py  # Agent 核心大脑：负责 RAG 检索与工具调度
│   ├── tools.py            # 工具脚本：负责外部 API 数据抓取
│   └── config.py           # 配置管理：环境变量与路径设置
├── data/                   # 数据仓库：存放手册 PDF 与预处理语料
└── requirements.txt        # 项目依赖清单

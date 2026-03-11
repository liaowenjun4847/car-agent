# src/tools.py

def get_weather(city: str):
    """查询指定城市的天气情况"""
    # 这里我们模拟一个返回结果。在实际生产中，这里会调用高德或心知天气的 API。
    # 为了演示效果，我们假设现在的天气是“大雨”
    print(f"系统提示：Agent 正在调用外部工具查询 {city} 的天气...")
    return {
        "city": city,
        "temperature": "18°C",
        "condition": "大雨",
        "suggestion": "路面湿滑，请减速慢行"
    }

# 定义工具的描述（这是给 DeepSeek 看的说明书）
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气预报",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如：广州"}
                },
                "required": ["city"]
            }
        }
    }
]
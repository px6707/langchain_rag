from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """返回当前时间。当用户询问时间、日期时使用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

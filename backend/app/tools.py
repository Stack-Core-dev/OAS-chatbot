from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """Get the current system date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


TOOLS = [get_current_datetime]

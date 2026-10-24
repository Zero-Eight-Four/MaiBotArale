"""终端交互共用的 Rich Console。"""

from rich.console import Console
from rich.theme import Theme

import sys

from src.common.console_output import get_console_stream

custom_theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "muted": "dim",
        "accent": "bold magenta",
    }
)

console = Console(theme=custom_theme, file=get_console_stream(sys.stdout))

"""黑话词条的文本命中校验，学习与使用共用同一规则。"""

import re


def contains_jargon(text: str, term: str) -> bool:
    term = term.strip()
    if len(term) < 2 or term.isdecimal():
        return False
    pattern = re.escape(term)
    if term[0].isascii() and (term[0].isalnum() or term[0] == "_"):
        pattern = r"(?<![a-zA-Z0-9_])" + pattern
    if term[-1].isascii() and (term[-1].isalnum() or term[-1] == "_"):
        pattern += r"(?![a-zA-Z0-9_])"
    return re.search(pattern, text, re.IGNORECASE) is not None

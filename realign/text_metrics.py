"""Đếm ký tự chuẩn tiếng Việt + wrap 2 dòng. NFC rồi đếm grapheme (bỏ combining marks)."""
import unicodedata

_COMBINING = {"Mn", "Mc", "Me"}


def char_count(s: str) -> int:
    s = unicodedata.normalize("NFC", s).replace("\n", "")
    return sum(1 for c in s if unicodedata.category(c) not in _COMBINING)


def wrap_two_lines(text: str, cpl_max: int) -> str:
    if char_count(text) <= cpl_max:
        return text
    words = text.split(" ")
    best = None  # (key, k)
    for k in range(1, len(words)):
        c1 = char_count(" ".join(words[:k]))
        c2 = char_count(" ".join(words[k:]))
        valid = c1 <= cpl_max and c2 <= cpl_max
        comma = 0 if words[k - 1].endswith(",") else 1
        # ưu tiên: hợp lệ > cân bằng > bottom-heavy > ngắt sau dấu phẩy
        key = (0 if valid else 1, abs(c2 - c1), 0 if c2 >= c1 else 1, comma)
        if best is None or key < best[0]:
            best = (key, k)
    k = best[1]
    return " ".join(words[:k]) + "\n" + " ".join(words[k:])

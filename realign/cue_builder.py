"""Gom word-level -> cue theo luật production (Netflix VN / BBC / ESIST).
Ngắt: hết câu | lặng >= PAUSE_SPLIT -> segment; mỗi segment dài hơn 1 cue
chia cân bằng (số cue tối thiểu) ưu tiên ngắt ở dấu phẩy. Ép tốc độ đọc CPS. Gộp cue vụn."""
import realign.config as config
from realign.text_metrics import char_count, wrap_two_lines


def fill_word_times(words: list[dict]) -> list[dict]:
    words = [dict(w) for w in words]
    n = len(words)
    for i, w in enumerate(words):
        if w.get("start") is None or w.get("end") is None:
            prev_e = next((words[j]["end"] for j in range(i - 1, -1, -1)
                           if words[j].get("end") is not None), 0.0)
            nxt_s = next((words[j]["start"] for j in range(i + 1, n)
                          if words[j].get("start") is not None), prev_e + 0.3)
            w["start"] = prev_e
            w["end"] = max(prev_e + 0.05, nxt_s)
    return words


def _make_cue(ws: list[dict]) -> dict:
    return {"text": " ".join(w["w"] for w in ws),
            "start": ws[0]["start"], "end": ws[-1]["end"]}


def _fits_lines(text: str, cfg) -> bool:
    """True nếu text word-wrap được vào <= LINES_MAX dòng, mỗi dòng <= CPL_MAX
    (word-wrap tham lam, không cắt đôi từ). Thay cho mốc CHAR_MAX cũ."""
    lines, cur = 1, 0
    for word in text.split(" "):
        wlen = char_count(word)
        if wlen > cfg.CPL_MAX:
            return False                       # 1 từ đã dài hơn 1 dòng
        if cur == 0:
            cur = wlen
        elif cur + 1 + wlen <= cfg.CPL_MAX:
            cur += 1 + wlen
        else:
            lines += 1
            cur = wlen
            if lines > cfg.LINES_MAX:
                return False
    return lines <= cfg.LINES_MAX


def _join(words: list[dict]) -> str:
    return " ".join(w["w"] for w in words)


def _split_segments(words: list[dict], cfg) -> list[list[dict]]:
    """Tách words thành segment tại hard-break: hết câu hoặc khoảng ngừng >= PAUSE_SPLIT."""
    ends = tuple(cfg.SENTENCE_END)
    segs, cur = [], []
    for w in words:
        if cur:
            prev = cur[-1]
            if prev["w"].endswith(ends) or (w["start"] - prev["end"]) >= cfg.PAUSE_SPLIT:
                segs.append(cur)
                cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    return segs


def _fill_k(words: list[dict], k: int, cfg) -> list[list[dict]]:
    """Chia words thành tối đa k nhóm cân bằng (~total/k ký tự mỗi nhóm),
    ưu tiên ngắt ngay sau dấu phẩy/`;`/`:` khi đã gần target."""
    target = char_count(_join(words)) / k
    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        if (k - len(groups)) <= 1:                          # nhóm cuối: gom hết phần còn lại
            continue
        if (len(words) - i - 1) <= (k - len(groups) - 1):   # chừa đủ từ cho các nhóm sau
            groups.append(cur)
            cur = []
            continue
        acc = char_count(_join(cur))
        ends_soft = w["w"].endswith((",", ";", ":"))
        if (acc >= target * 0.6 and ends_soft) or acc >= target:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _segment_to_cues(words: list[dict], cfg) -> list[list[dict]]:
    """1 segment -> các nhóm từ, mỗi nhóm <= 2 dòng <= CPL. Segment dài hơn 1 cue
    -> chia cân bằng (số cue tối thiểu) + ưu tiên ngắt ở dấu phẩy (hết đuôi cụt)."""
    if _fits_lines(_join(words), cfg):
        return [words]
    for k in range(2, len(words) + 1):
        groups = _fill_k(words, k, cfg)
        if len(groups) == k and all(_fits_lines(_join(g), cfg) for g in groups):
            return groups
    return [[w] for w in words]                             # fallback hiếm (1 từ quá dài)


def _enforce_min_display(cues: list[dict], cfg) -> list[dict]:
    out, i = [], 0
    while i < len(cues):
        c = dict(cues[i])
        needed = max(cfg.DUR_MIN, char_count(c["text"]) / cfg.CPS_MAX)
        nxt_start = cues[i + 1]["start"] if i + 1 < len(cues) else None
        cap = c["start"] + cfg.DUR_MAX
        if nxt_start is not None:
            cap = min(cap, nxt_start - cfg.GAP_MIN)
        want_end = min(max(c["end"], c["start"] + needed), cap)
        if (want_end - c["start"]) >= cfg.DUR_MIN - 1e-6:
            c["end"] = want_end
            out.append(c)
            i += 1
            continue
        # không đủ chỗ đạt sàn -> gộp vào cue trước (ưu tiên) hoặc cue sau, nếu char vừa
        if out and _fits_lines(out[-1]["text"] + " " + c["text"], cfg):
            out[-1]["text"] += " " + c["text"]
            out[-1]["end"] = max(out[-1]["end"], c["end"])
            i += 1
            continue
        if i + 1 < len(cues) and _fits_lines(c["text"] + " " + cues[i + 1]["text"], cfg):
            nxt = dict(cues[i + 1])
            nxt["text"] = c["text"] + " " + nxt["text"]
            nxt["start"] = c["start"]
            cues[i + 1] = nxt
            i += 1
            continue
        # không gộp được -> chấp nhận, ít nhất đạt FLOOR
        c["end"] = max(c["end"], want_end, c["start"] + cfg.FLOOR)
        out.append(c)
        i += 1
    return out


def build_cues(words: list[dict], cfg=config) -> list[dict]:
    if not words:
        return []
    words = fill_word_times(words)
    raw = []
    for seg in _split_segments(words, cfg):
        for group in _segment_to_cues(seg, cfg):
            raw.append(_make_cue(group))
    cues = _enforce_min_display(raw, cfg)
    for c in cues:
        c["text"] = wrap_two_lines(c["text"], cfg.CPL_MAX)
    return cues

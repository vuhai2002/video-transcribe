"""Xuất cue -> text SRT."""


def fmt_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec * 1000) % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cues_to_srt(cues: list[dict]) -> str:
    blocks = [f"{n}\n{fmt_ts(c['start'])} --> {fmt_ts(c['end'])}\n{c['text']}"
              for n, c in enumerate(cues, 1)]
    return ("\n\n".join(blocks) + "\n") if blocks else ""

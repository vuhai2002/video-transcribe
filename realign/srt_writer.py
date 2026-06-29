"""Xuất cue -> text SRT."""


def fmt_ts(sec: float) -> str:
    total_ms = round(max(sec, 0.0) * 1000)   # round (không truncate) tránh lệch ms
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cues_to_srt(cues: list[dict]) -> str:
    blocks = [f"{n}\n{fmt_ts(c['start'])} --> {fmt_ts(c['end'])}\n{c['text']}"
              for n, c in enumerate(cues, 1)]
    return ("\n\n".join(blocks) + "\n") if blocks else ""

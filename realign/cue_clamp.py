"""Vệ sinh thời gian cue: monotonic, gap rules, DUR_MAX, floor, VAD trim bảo thủ."""
import realign.config as config


def clamp_cues(cues: list[dict], speech_region, cfg=config) -> list[dict]:
    if not cues:
        return []
    cues = [dict(c) for c in cues]
    cues.sort(key=lambda c: c["start"])

    for i, c in enumerate(cues):
        if i > 0 and c["start"] < cues[i - 1]["start"]:
            c["start"] = cues[i - 1]["start"]
        if c["end"] < c["start"] + cfg.FLOOR:
            c["end"] = c["start"] + cfg.FLOOR
        c["end"] = min(c["end"], c["start"] + cfg.DUR_MAX)

    if speech_region is not None:
        first, last = speech_region
        c0 = cues[0]
        if first > c0["start"]:
            c0["start"] = min(first, c0["end"] - cfg.DUR_MIN)
        cl = cues[-1]
        cl["end"] = max(min(cl["end"], last + cfg.VAD_PAD), cl["start"] + cfg.DUR_MIN)

    for i in range(len(cues) - 1):
        gap = cues[i + 1]["start"] - cues[i]["end"]
        if gap < cfg.PAUSE_GAP:  # overlap hoặc gap nhỏ -> để sát nhau, chống nhấp nháy
            cues[i]["end"] = max(
                cues[i]["start"] + cfg.FLOOR,
                min(cues[i + 1]["start"] - cfg.GAP_MIN, cues[i]["start"] + cfg.DUR_MAX),
            )
    return cues

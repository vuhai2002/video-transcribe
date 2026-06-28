"""SRT -> transcript txt: lay text cac cue (giu chu + dau cau), bo index/timestamp.
Line break trong txt khong quan trong (stage segment gom cue lai theo word time),
nhung GIU dau cau vi segment dung dau cau de ngat cau.
"""
import argparse
import re


def parse_srt_cues(srt_text: str) -> list[str]:
    srt_text = srt_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    cues: list[str] = []
    for block in re.split(r"\n\s*\n", srt_text):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        ts_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ts_idx is None:
            continue
        text = " ".join(ln.strip() for ln in lines[ts_idx + 1:]).strip()
        if text:
            cues.append(text)
    return cues


def srt_to_transcript(srt_text: str) -> str:
    return "\n".join(parse_srt_cues(srt_text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.srt, encoding="utf-8") as f:
        text = f.read()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(srt_to_transcript(text) + "\n")
    print(f"wrote transcript -> {a.out}")


if __name__ == "__main__":
    main()

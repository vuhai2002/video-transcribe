"""Stage 3 (re): words.json (+VAD) -> cue (luật production) -> clamp -> srt.
Chạy lại được tức thì cho cả 713 file không cần align lại."""
import argparse
import json

import realign.config as config
from realign.cue_builder import build_cues
from realign.cue_clamp import clamp_cues
from realign.srt_writer import cues_to_srt


def segment(words_json: str, out_srt: str, vad_region=None, cfg=config) -> list[dict]:
    with open(words_json, encoding="utf-8") as f:
        data = json.load(f)
    cues = build_cues(data["words"], cfg)
    cues = clamp_cues(cues, vad_region, cfg)
    with open(out_srt, "w", encoding="utf-8") as f:
        f.write(cues_to_srt(cues))
    return cues


def _load_vad(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return (d["first"], d["last"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words-json", required=True)
    ap.add_argument("--out-srt", required=True)
    ap.add_argument("--vad-json")
    a = ap.parse_args()
    cues = segment(a.words_json, a.out_srt, _load_vad(a.vad_json))
    print(f"wrote {len(cues)} cues -> {a.out_srt}")


if __name__ == "__main__":
    main()

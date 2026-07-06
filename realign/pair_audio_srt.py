"""Ghép .srt <-> .mp3 theo title chuẩn hoá. Chuẩn hoá key CHỈ dùng nội bộ để ghép
2 tập filename (không cần trùng khít logic BE)."""
import argparse
import os
import re
import unicodedata


def normalize_key(title: str) -> str:
    """Chuẩn hoá title: NFD, bỏ dấu, lowercase, đ->d, gộp space."""
    s = unicodedata.normalize("NFD", title)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().replace("đ", "d")
    return re.sub(r"\s+", " ", s).strip()


def index_audio(audio_dirs: list[str]) -> dict[str, str]:
    """Chỉ mục .mp3 từ danh sách thư mục: {normalized_key -> đường_dẫn_mp3}."""
    idx: dict[str, str] = {}
    for d in audio_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".mp3"):
                continue
            key = normalize_key(os.path.splitext(name)[0])
            idx.setdefault(key, os.path.join(d, name))  # giữ file đầu nếu trùng key
    return idx


def build_pairs(src_dir: str, audio_dirs: list[str], ext: str = ".srt") -> tuple[list[dict], list[dict], list[dict]]:
    """Ghép transcript (.srt hoặc .txt) <-> .mp3.

    Args:
        src_dir: thư mục chứa transcript (.srt hoặc .txt)
        audio_dirs: danh sách thư mục chứa .mp3
        ext: đuôi nguồn cần ghép (".srt" hoặc ".txt")

    Returns:
        (pairs, unpaired_srt, unpaired_audio) với:
        - pairs: [{"srt_path", "srt_name", "kind", "audio_path", "key"}, ...] (kind = "srt"|"txt")
        - unpaired_srt: [{"srt_name", "srt_path", "key"}, ...]
        - unpaired_audio: [{"audio_name", "audio_path", "key"}, ...]
    """
    if not os.path.isdir(src_dir):
        raise ValueError(f"src_dir not found or not a directory: {src_dir!r}")
    audio_idx = index_audio(audio_dirs)
    kind = ext.lstrip(".").lower()
    used: set[str] = set()
    pairs, unpaired_srt = [], []
    for name in sorted(os.listdir(src_dir)):
        if not name.lower().endswith(ext.lower()):
            continue
        key = normalize_key(os.path.splitext(name)[0])
        src_path = os.path.join(src_dir, name)
        if key in audio_idx:
            used.add(key)
            pairs.append({"srt_path": src_path, "srt_name": name, "kind": kind,
                          "audio_path": audio_idx[key], "key": key})
        else:
            unpaired_srt.append({"srt_name": name, "srt_path": src_path, "key": key})
    unpaired_audio = [{"audio_name": os.path.basename(p), "audio_path": p, "key": k}
                      for k, p in audio_idx.items() if k not in used]
    return pairs, unpaired_srt, unpaired_audio


def format_unpaired_report(unpaired_srt: list[dict], unpaired_audio: list[dict]) -> str:
    """Format báo cáo file lẻ (chưa ghép được)."""
    out = ["# File lẻ (chưa ghép được)", "",
           f"## SRT không có audio ({len(unpaired_srt)})"]
    out += [f"- {u['srt_name']}  (key: {u['key']})" for u in unpaired_srt] or ["- (không có)"]
    out += ["", f"## Audio không có srt ({len(unpaired_audio)})"]
    out += [f"- {u['audio_name']}  (key: {u['key']})" for u in unpaired_audio] or ["- (không có)"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt-dir", required=True)
    ap.add_argument("--audio-dir", action="append", required=True)
    ap.add_argument("--report")
    a = ap.parse_args()
    pairs, us, ua = build_pairs(a.srt_dir, a.audio_dir)
    print(f"pairs={len(pairs)} unpaired_srt={len(us)} unpaired_audio={len(ua)}")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(format_unpaired_report(us, ua))
        print(f"report -> {a.report}")


if __name__ == "__main__":
    main()

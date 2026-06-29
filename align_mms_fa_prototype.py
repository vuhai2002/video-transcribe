#!/usr/bin/env python3
"""
PROTOTYPE - forced-align an existing transcript to audio with torchaudio MMS_FA.

Input  : audio (wav/mp3 already 16k mono recommended) + txt (one cue per line,
         e.g. converted from an existing .srt).
Output : .srt that REUSES the original line segmentation but with timestamps
         anchored acoustically to the audio (no accumulating drift).

Why this fixes the drift: instead of guessing time by character-count ratio,
CTC forced alignment finds where each word actually occurs in the audio. Errors
stay local, they do not accumulate.

Usage:
  python align_mms_fa_prototype.py --audio in.wav --txt in.txt --out out.srt \
      [--device cuda|cpu] [--emit-chunk-sec 20] [--max-sec 0]
  --max-sec N : only align the first N seconds (0 = full) - use a small N to test fast.
"""
import argparse
import re
import sys
import time
import unicodedata

import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle

try:
    sys.stdout.reconfigure(encoding="utf-8")  # print Vietnamese without cp1252 crash
except Exception:
    pass

try:
    from uroman import Uroman
    _URO = Uroman()
except Exception:
    _URO = None


def log(*a):
    print(*a)
    sys.stdout.flush()


def normalize_word(w: str) -> str:
    """Romanize a (Vietnamese) word to the MMS latin token set [a-z']."""
    if _URO is not None:
        try:
            w = _URO.romanize_string(w)
        except Exception:
            pass
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    w = w.lower().replace("đ", "d").replace("’", "'")
    return re.sub(r"[^a-z']", "", w)


def fmt_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec * 1000) % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--txt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--emit-chunk-sec", type=float, default=20.0)
    ap.add_argument("--max-sec", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    log(f"device={device} torch={torch.__version__} torchaudio={torchaudio.__version__}")

    import soundfile as sf

    data, sr = sf.read(args.audio, dtype="float32")  # avoid torchaudio.load (needs torchcodec)
    wav = torch.from_numpy(data)
    if wav.ndim == 2:
        wav = wav.mean(dim=1)
    wav = wav.unsqueeze(0)  # [1, samples]
    if sr != bundle.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
        sr = bundle.sample_rate
    if args.max_sec > 0:
        wav = wav[:, : int(args.max_sec * sr)]
    total_sec = wav.size(1) / sr
    log(f"audio: {total_sec:.1f}s ({total_sec / 60:.1f} min) @ {sr}Hz")

    lines = [ln.strip() for ln in open(args.txt, encoding="utf-8") if ln.strip()]
    words, line_of_word, norm_words = [], [], []
    for li, line in enumerate(lines):
        for w in line.split():
            words.append(w)
            line_of_word.append(li)
            norm_words.append(normalize_word(w))
    align_idx = [i for i, nw in enumerate(norm_words) if nw]
    align_words = [norm_words[i] for i in align_idx]
    log(f"lines={len(lines)} words={len(words)} alignable={len(align_idx)}")

    model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    t0 = time.time()
    chunk = int(args.emit_chunk_sec * sr)
    ems = []
    with torch.inference_mode():
        for i in range(0, wav.size(1), chunk):
            seg = wav[:, i : i + chunk].to(device)
            emi, _ = model(seg)
            ems.append(emi.cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
    emission = torch.cat(ems, dim=1)
    log(f"emission frames={emission.size(1)} ({time.time() - t0:.1f}s)")

    t1 = time.time()
    with torch.inference_mode():
        token_spans = aligner(emission[0].to(device), tokenizer(align_words))
    log(f"aligned {len(token_spans)} words ({time.time() - t1:.1f}s)")

    ratio = wav.size(1) / emission.size(1)
    line_times = [[] for _ in lines]
    for k, spans in enumerate(token_spans):
        i = align_idx[k]
        start = spans[0].start * ratio / sr
        end = spans[-1].end * ratio / sr
        line_times[line_of_word[i]].append((start, end))

    cues = []
    for li, line in enumerate(lines):
        wt = line_times[li]
        if wt:
            cues.append([line, min(s for s, _ in wt), max(e for _, e in wt)])
        else:
            cues.append([line, None, None])
    # fill cues with no aligned words by interpolating from neighbours
    for idx in range(len(cues)):
        if cues[idx][1] is None:
            prev_e = next((cues[j][2] for j in range(idx - 1, -1, -1) if cues[j][2] is not None), 0.0)
            nxt_s = next((cues[j][1] for j in range(idx + 1, len(cues)) if cues[j][1] is not None), prev_e + 0.5)
            cues[idx][1], cues[idx][2] = prev_e, max(prev_e + 0.1, nxt_s)
    for idx in range(len(cues)):
        if cues[idx][2] <= cues[idx][1]:
            cues[idx][2] = cues[idx][1] + 0.1

    with open(args.out, "w", encoding="utf-8") as f:
        for n, (line, s, e) in enumerate(cues, 1):
            f.write(f"{n}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{line}\n\n")
    log(f"wrote {len(cues)} cues -> {args.out}")
    if cues:
        log(f"first cue @ {fmt_ts(cues[0][1])} : {cues[0][0][:50]}")
        log(f"last cue  @ {fmt_ts(cues[-1][2])} : {cues[-1][0][:50]}")


if __name__ == "__main__":
    main()

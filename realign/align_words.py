"""Stage 2 (dat): forced-align transcript vao audio bang torchaudio MMS_FA.
Xuat word-level [{w,start,end,score}]. Tai dung logic prototype da chay (chunked
emissions tranh OOM GPU 4GB).

Import torch/torchaudio chi khi goi ham can GPU (load_audio, get_models, align).
Cac ham thuan (normalize_word, assemble_words) dung duoc khong can torch.
"""
import argparse
import json
import re
import sys
import time
import unicodedata

import realign.config as config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# uroman: optional, khong can GPU. Neu chua cai dat thi normalize_word van chay
# (chi mat kha nang romanize chu Viet qua uroman, NFD-strip van hoat dong).
try:
    from uroman import Uroman
    _URO = Uroman()
except Exception:
    _URO = None


def normalize_word(w: str) -> str:
    """Romanize (Vietnamese) word sang tap token MMS latin [a-z']. Dung uroman truoc,
    roi strip dau NFD, lowercase, thay the 'o' go -> 'd', loai ky tu ngoai [a-z']."""
    if _URO is not None:
        try:
            w = _URO.romanize_string(w)
        except Exception:
            pass
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    w = w.lower().replace("đ", "d").replace("’", "'")
    return re.sub(r"[^a-z']", "", w)


def read_words(txt_path: str) -> list[str]:
    """Doc toan bo token tu file txt (moi dong la mot cue), giu nguyen thu tu."""
    words: list[str] = []
    for ln in open(txt_path, encoding="utf-8"):
        words += ln.split()
    return words


def load_audio(path: str, sample_rate: int, max_sec: float = 0.0):
    """Doc audio bang soundfile (tranh torchaudio.load can torchcodec), resample neu can,
    cat doan dau neu max_sec > 0. Tra ve (Tensor[1,N], sr)."""
    import torch
    import torchaudio
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim == 2:
        wav = wav.mean(dim=1)
    wav = wav.unsqueeze(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate
    if max_sec > 0:
        wav = wav[:, : int(max_sec * sr)]
    return wav, sr


def get_models(device):
    """Load MMS_FA model + tokenizer + aligner mot lan, tai su dung cho nhieu file."""
    from torchaudio.pipelines import MMS_FA as bundle
    return bundle.get_model().to(device), bundle.get_tokenizer(), bundle.get_aligner()


def assemble_words(words_raw, align_idx, token_spans, ratio, sr) -> list[dict]:
    """Anh xa token_spans (danh sach spans torchaudio) tro lai vi tri goc trong words_raw.

    - align_idx[k] la chi so trong words_raw tuong ung voi token_spans[k].
    - Tu nao khong co trong align_idx (khong normalize duoc) -> start/end/score = None.
    - Thoi gian tinh: frame_index * ratio / sr (don vi giay).
    - score = trung binh score cua cac token trong tu.
    """
    times: dict[int, tuple] = {}
    for k, spans in enumerate(token_spans):
        i = align_idx[k]
        start = spans[0].start * ratio / sr
        end = spans[-1].end * ratio / sr
        score = sum(s.score for s in spans) / len(spans)
        times[i] = (start, end, score)
    out = []
    for i, w in enumerate(words_raw):
        if i in times:
            s, e, sc = times[i]
            out.append({"w": w, "start": float(s), "end": float(e), "score": float(sc)})
        else:
            out.append({"w": w, "start": None, "end": None, "score": None})
    return out


def align(
    audio_path: str,
    words_raw: list[str],
    device: str = "cpu",
    emit_chunk_sec: float = config.EMIT_CHUNK_SEC,
    max_sec: float = 0.0,
    models=None,
) -> list[dict]:
    """Forced-align words_raw vao audio_path dung MMS_FA.

    Chay chunked emission (emit_chunk_sec) de tranh OOM GPU 4GB.
    models=(model,tokenizer,aligner) co the truyen vao de tai su dung giua nhieu file.
    Tra ve list[Word] = [{w, start, end, score}] (start=None neu khong align duoc).
    """
    import torch
    dev = torch.device(device)
    wav, sr = load_audio(audio_path, config.SAMPLE_RATE, max_sec)
    norm = [normalize_word(w) for w in words_raw]
    align_idx = [i for i, nw in enumerate(norm) if nw]
    align_words_list = [norm[i] for i in align_idx]
    model, tokenizer, aligner = models if models is not None else get_models(dev)
    chunk = int(emit_chunk_sec * sr)
    ems = []
    with torch.inference_mode():
        for i in range(0, wav.size(1), chunk):
            emi, _ = model(wav[:, i: i + chunk].to(dev))
            ems.append(emi.cpu())
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    emission = torch.cat(ems, dim=1)
    with torch.inference_mode():
        token_spans = aligner(emission[0].to(dev), tokenizer(align_words_list))
    ratio = wav.size(1) / emission.size(1)
    return assemble_words(words_raw, align_idx, token_spans, ratio, sr)


def main():
    import torch
    ap = argparse.ArgumentParser(description="MMS forced-align -> word-level json")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--txt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--emit-chunk-sec", type=float, default=config.EMIT_CHUNK_SEC)
    ap.add_argument("--max-sec", type=float, default=0.0)
    a = ap.parse_args()
    words_raw = read_words(a.txt)
    t0 = time.time()
    words = align(a.audio, words_raw, a.device, a.emit_chunk_sec, a.max_sec)
    import os
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(
        {"key": a.key or a.audio, "audio": a.audio, "words": words},
        open(a.out_json, "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    n_al = sum(1 for w in words if w["start"] is not None)
    print(f"aligned {n_al}/{len(words)} words in {time.time() - t0:.1f}s -> {a.out_json}")


if __name__ == "__main__":
    main()

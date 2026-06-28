"""Tham số pipeline re-align - chuẩn production.
Xem plans .../reports/subtitle-timing-standards-research.md để biết nguồn từng số.
"""
# cue text
CPL_MAX = 42                      # ky tu / dong (dem sau NFC, theo grapheme)
LINES_MAX = 2
CHAR_MAX = CPL_MAX * LINES_MAX    # 84

# toc do doc
CPS_MAX = 15.0                    # tran cung
CPS_TARGET = 13.0                 # muc tieu thiet ke

# duration (giay)
DUR_MIN = 1.5
DUR_MAX = 7.0
FLOOR = 0.3                       # san tuyet doi

# segmentation
PAUSE_SPLIT = 0.6                 # lang giua 2 tu >= -> ngat cue
SENTENCE_END = ".!?:;"            # ky tu cuoi tu -> ngat cau

# khoang cach 2 cue
GAP_MIN = 0.084                   # 2 frame @24fps (chong nhap nhay)
PAUSE_GAP = 0.5                   # < -> de sat nhau; >= -> giu nguyen

# VAD
VAD_PAD = 0.3
VAD_TRUST_GAP = 2.0               # VAD lech alignment > nguong -> tin alignment

# audio / model
SAMPLE_RATE = 16000
EMIT_CHUNK_SEC = 20.0

# paths (override duoc qua CLI run_batch --audio-dirs)
AUDIO_DIRS = [r"Y:\run-script-1", r"Y:\run-script-2", r"Y:\run-script-3"]

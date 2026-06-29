"""Tham số pipeline re-align - chuẩn production.
Xem plans .../reports/subtitle-timing-standards-research.md để biết nguồn từng số.
"""
# cue text
CPL_MAX = 42                      # ký tự / dòng (đếm sau NFC, theo grapheme)
LINES_MAX = 2
CHAR_MAX = CPL_MAX * LINES_MAX    # 84 - chỉ tham chiếu; luật ngắt cue thực tế dùng _fits_lines theo CPL (xem cue_builder)

# tốc độ đọc
CPS_MAX = 15.0                    # trần cứng
CPS_TARGET = 13.0                 # mục tiêu thiết kế

# duration (giây)
DUR_MIN = 1.5
DUR_MAX = 7.0
FLOOR = 0.3                       # sàn tuyệt đối

# segmentation
PAUSE_SPLIT = 0.6                 # lặng giữa 2 từ >= -> ngắt cue
SENTENCE_END = ".!?:;"            # ký tự cuối từ -> ngắt câu

# khoảng cách 2 cue
GAP_MIN = 0.084                   # 2 frame @24fps (chống nhấp nháy)
PAUSE_GAP = 0.5                   # < -> để sát nhau; >= -> giữ nguyên

# VAD
VAD_PAD = 0.3
VAD_TRUST_GAP = 2.0               # VAD lệch alignment > ngưỡng -> tin alignment

# audio / model
SAMPLE_RATE = 16000
EMIT_CHUNK_SEC = 20.0

# paths (override được qua CLI run_batch --audio-dirs)
AUDIO_DIRS = [r"Y:\run-script-1", r"Y:\run-script-2", r"Y:\run-script-3"]

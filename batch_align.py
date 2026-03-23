#!/usr/bin/env python3
"""
Batch Align Script - Xử lý hàng loạt file mp3 + txt → srt
Load model 1 lần duy nhất, xử lý tuần tự tất cả file.
Hỗ trợ resume: skip file đã có .srt

Usage:
    python batch_align.py --mp3-dir /path/to/mp3 --txt-dir /path/to/txt --srt-dir /path/to/srt
"""

import whisperx
import torch
import gc
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta

# ============================================================
# CẤU HÌNH
# ============================================================
DEFAULT_MP3_DIR = "/home/vuhai/data/mp3"
DEFAULT_TXT_DIR = "/home/vuhai/data/txt"
DEFAULT_SRT_DIR = "/home/vuhai/data/srt"
MODEL_NAME = "large-v2"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16

# ============================================================
# LOGGING SETUP
# ============================================================
log_filename = f"batch_align_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)

# Tạo handlers dùng chung
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Cấu hình root logger để bắt TẤT CẢ log từ mọi thư viện
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

# Đảm bảo whisperx logger cũng ghi vào file
# Chỉ set cho parent logger, child loggers tự propagate
wx_logger = logging.getLogger('whisperx')
wx_logger.setLevel(logging.DEBUG)
wx_logger.addHandler(file_handler)
wx_logger.addHandler(stream_handler)
wx_logger.propagate = False  # Tránh duplicate qua root logger

logger = logging.getLogger(__name__)

# Giảm noise từ HTTP libraries (chỉ ghi WARNING trở lên)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ============================================================
# HELPER FUNCTIONS (từ align_whisperx.py)
# ============================================================

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds * 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def load_user_transcript(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def interpolate_timestamps_heuristic(user_lines, whisper_segments, audio_duration):
    if not whisper_segments:
        return []

    char_timeline = []
    cumulative_chars = 0
    char_timeline.append((0, whisper_segments[0]["start"]))

    for seg in whisper_segments:
        text_len = len(seg["text"])
        cumulative_chars += text_len
        char_timeline.append((cumulative_chars, seg["end"]))

    total_whisper_chars = cumulative_chars
    total_user_chars = sum(len(line) for line in user_lines)

    if total_user_chars == 0:
        return []
    scale = total_whisper_chars / total_user_chars

    def get_time(char_idx):
        target_char = char_idx * scale
        for i in range(len(char_timeline) - 1):
            c1, t1 = char_timeline[i]
            c2, t2 = char_timeline[i + 1]
            if c1 <= target_char <= c2:
                if c2 == c1:
                    return t1
                ratio = (target_char - c1) / (c2 - c1)
                return t1 + ratio * (t2 - t1)
        return char_timeline[-1][1]

    aligned_segments = []
    current_char_idx = 0

    for line in user_lines:
        line_len = len(line)
        start_time = get_time(current_char_idx)
        end_time = get_time(current_char_idx + line_len)

        if start_time >= end_time:
            end_time = start_time + 0.1

        aligned_segments.append({
            "text": line,
            "start": start_time,
            "end": end_time
        })
        current_char_idx += line_len

    return aligned_segments


# ============================================================
# BATCH PROCESSING
# ============================================================

def find_matching_pairs(mp3_dir, txt_dir):
    """Tìm các cặp mp3 + txt có cùng tên."""
    mp3_files = {os.path.splitext(f)[0]: f for f in os.listdir(mp3_dir) if f.endswith('.mp3')}
    txt_files = {os.path.splitext(f)[0]: f for f in os.listdir(txt_dir) if f.endswith('.txt')}

    pairs = []
    missing_txt = []

    for name in sorted(mp3_files.keys()):
        if name in txt_files:
            pairs.append({
                "name": name,
                "mp3": os.path.join(mp3_dir, mp3_files[name]),
                "txt": os.path.join(txt_dir, txt_files[name]),
            })
        else:
            missing_txt.append(name)

    return pairs, missing_txt


def process_single_file(mp3_path, txt_path, srt_path, whisper_model, align_model, align_metadata):
    """Xử lý 1 cặp file mp3 + txt → srt. Model đã được load sẵn."""

    # 1. Load audio
    audio = whisperx.load_audio(mp3_path)

    # 2. Transcribe để lấy anchor timestamps (force tiếng Việt)
    result_transcribe = whisper_model.transcribe(audio, batch_size=BATCH_SIZE, language="vi")
    whisper_segments = result_transcribe["segments"]
    logger.info(f"   Transcribe: {len(whisper_segments)} segments")

    # 3. Đọc text chuẩn
    user_lines = load_user_transcript(txt_path)
    logger.info(f"   Text chuẩn: {len(user_lines)} dòng")

    # 4. Interpolate timestamps
    audio_duration = whisper_segments[-1]['end'] if whisper_segments else 0
    rough_segments = interpolate_timestamps_heuristic(user_lines, whisper_segments, audio_duration)

    if not rough_segments:
        logger.error("   Không tạo được mapping!")
        return False

    # 5. Forced Alignment
    try:
        result_align = whisperx.align(
            rough_segments,
            align_model,
            align_metadata,
            audio,
            DEVICE,
            return_char_alignments=False
        )
        final_segments = result_align["segments"]
        logger.info(f"   Alignment: {len(final_segments)} segments aligned")
    except Exception as e:
        logger.warning(f"   ⚠️ Alignment lỗi, fallback: {e}")
        final_segments = rough_segments

    # 6. Xuất SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(final_segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

    return True


def run_batch(mp3_dir, txt_dir, srt_dir, force=False):
    """Chạy batch processing toàn bộ."""

    # Tạo thư mục output nếu chưa có
    os.makedirs(srt_dir, exist_ok=True)

    # Tìm cặp file
    pairs, missing_txt = find_matching_pairs(mp3_dir, txt_dir)

    logger.info("=" * 60)
    logger.info("🚀 BATCH ALIGN - BẮT ĐẦU")
    logger.info(f"   MP3 dir: {mp3_dir}")
    logger.info(f"   TXT dir: {txt_dir}")
    logger.info(f"   SRT dir: {srt_dir}")
    logger.info(f"   Tổng cặp file: {len(pairs)}")
    logger.info("=" * 60)

    if missing_txt:
        logger.warning(f"⚠️ {len(missing_txt)} file MP3 không có TXT tương ứng:")
        for name in missing_txt:
            logger.warning(f"   - {name}")

    # Kiểm tra file đã xử lý
    if not force:
        pending_pairs = []
        skipped = 0
        for pair in pairs:
            srt_path = os.path.join(srt_dir, pair["name"] + ".srt")
            if os.path.exists(srt_path):
                skipped += 1
            else:
                pending_pairs.append(pair)
        if skipped > 0:
            logger.info(f"⏭️  Bỏ qua {skipped} file đã có SRT (dùng --force để chạy lại)")
        pairs = pending_pairs

    if not pairs:
        logger.info("✅ Không có file nào cần xử lý!")
        return

    logger.info(f"📋 Cần xử lý: {len(pairs)} file")
    logger.info("")

    # ============================================================
    # LOAD MODEL 1 LẦN DUY NHẤT
    # ============================================================
    logger.info("🔄 Đang load Whisper model (1 lần duy nhất)...")
    t_start_model = time.time()
    whisper_model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
    logger.info(f"   ✅ Whisper model loaded ({time.time() - t_start_model:.1f}s)")

    logger.info("🔄 Đang load Alignment model...")
    t_start_align = time.time()
    align_model, align_metadata = whisperx.load_align_model(language_code="vi", device=DEVICE)
    logger.info(f"   ✅ Alignment model loaded ({time.time() - t_start_align:.1f}s)")
    logger.info("")

    # ============================================================
    # XỬ LÝ TỪNG FILE
    # ============================================================
    total = len(pairs)
    success_count = 0
    fail_count = 0
    failed_files = []
    total_start_time = time.time()

    for idx, pair in enumerate(pairs, 1):
        name = pair["name"]
        mp3_path = pair["mp3"]
        txt_path = pair["txt"]
        srt_path = os.path.join(srt_dir, name + ".srt")

        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[{idx}/{total}] 🎵 {name}")
        logger.info(f"   MP3: {mp3_path}")
        logger.info(f"   TXT: {txt_path}")

        file_start_time = time.time()

        try:
            result = process_single_file(
                mp3_path, txt_path, srt_path,
                whisper_model, align_model, align_metadata
            )
            elapsed = time.time() - file_start_time

            if result:
                success_count += 1
                logger.info(f"   ✅ Thành công! ({elapsed:.1f}s) → {srt_path}")
            else:
                fail_count += 1
                failed_files.append(name)
                logger.error(f"   ❌ Thất bại! ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - file_start_time
            fail_count += 1
            failed_files.append(name)
            logger.error(f"   ❌ Lỗi: {e} ({elapsed:.1f}s)")

        # Progress
        done = idx
        remaining = total - done
        avg_time = (time.time() - total_start_time) / done
        eta_seconds = remaining * avg_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        logger.info(f"   📊 Tiến trình: {done}/{total} | Thành công: {success_count} | Lỗi: {fail_count} | ETA: {eta_str}")

        # Clear CUDA cache sau mỗi file
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # TỔNG KẾT
    # ============================================================
    total_elapsed = time.time() - total_start_time
    total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))

    logger.info("")
    logger.info("=" * 60)
    logger.info("🏁 BATCH ALIGN - KẾT THÚC")
    logger.info(f"   ⏱️  Tổng thời gian: {total_elapsed_str}")
    logger.info(f"   ✅ Thành công: {success_count}/{total}")
    logger.info(f"   ❌ Thất bại: {fail_count}/{total}")

    if failed_files:
        logger.info(f"   📋 Danh sách file lỗi:")
        for name in failed_files:
            logger.info(f"      - {name}")

    logger.info(f"   📁 Log file: {log_filepath}")
    logger.info("=" * 60)

    # Cleanup models
    del whisper_model, align_model
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Align - Xử lý hàng loạt mp3 + txt → srt")
    parser.add_argument("--mp3-dir", default=DEFAULT_MP3_DIR, help="Thư mục chứa file MP3")
    parser.add_argument("--txt-dir", default=DEFAULT_TXT_DIR, help="Thư mục chứa file TXT")
    parser.add_argument("--srt-dir", default=DEFAULT_SRT_DIR, help="Thư mục output SRT")
    parser.add_argument("--force", action="store_true", help="Chạy lại tất cả, kể cả file đã có SRT")

    args = parser.parse_args()
    run_batch(args.mp3_dir, args.txt_dir, args.srt_dir, force=args.force)

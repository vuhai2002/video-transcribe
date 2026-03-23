#!/usr/bin/env python3
"""
Batch Transcribe Script - Xử lý hàng loạt file MP3 → TXT bằng Gemini API
Chạy SONG SONG nhiều luồng để tăng tốc.
Hỗ trợ resume: skip file đã có .txt

Usage:
    python3 batch_transcribe.py
    python3 batch_transcribe.py --workers 5
    python3 batch_transcribe.py --mp3-dir /path/to/mp3 --txt-dir /path/to/txt
    python3 batch_transcribe.py --force
"""

import os
import sys
import re
import time
import gc
import argparse
import logging
import traceback
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from google import genai
from google.genai import types

# ============================================================
# CẤU HÌNH
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

DEFAULT_MP3_DIR = "/home/vuhai/data/mp3"
DEFAULT_TXT_DIR = "/home/vuhai/data/txt"
DEFAULT_WORKERS = 5

# ============================================================
# LOGGING SETUP - Chi tiết để phân tích sau
# ============================================================
log_filename = f"batch_transcribe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)

# File handler - ghi TẤT CẢ (DEBUG trở lên)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'))

# Console handler - chỉ INFO trở lên
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'))

# Root logger - bắt tất cả log từ mọi thư viện (google, urllib3, ...)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

# Logger riêng cho script
logger = logging.getLogger("batch-transcribe")

# Giảm noise từ urllib3 và httpx (chỉ ghi WARNING trở lên)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

# ============================================================
# PROMPT (giống transcribe-gemini-txt.py)
# ============================================================

TRANSCRIBE_PROMPT = """\
Nghe file âm thanh đính kèm và chép lại toàn bộ nội dung thành văn bản (verbatim transcription).

**Yêu cầu bắt buộc:**
1. Chép chính xác từng từ, không tóm tắt, không bỏ sót từ nào.
2. Ngắt dòng tự nhiên theo ngữ điệu, mỗi câu hoặc mệnh đề ngắn trên một dòng riêng.
3. Mỗi dòng tối đa khoảng 60 ký tự để dễ đọc.
4. Không đánh số dòng, không thêm timestamp, không thêm tiêu đề hay ghi chú.
5. Không dùng markdown, không dùng code block. Chỉ trả về plain text thuần túy.
6. Dấu câu: sử dụng dấu chấm, dấu phẩy, dấu hỏi, dấu chấm than... tự nhiên theo ngữ cảnh.
7. Giữ nguyên ngôn ngữ gốc (tiếng Việt).

Chỉ trả về nội dung văn bản, không có bất kỳ lời dẫn hay giải thích nào.
"""

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

# Thread-safe counter
class ProgressTracker:
    """Thread-safe progress tracker."""
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.success = 0
        self.fail = 0
        self.failed_files = []
        self.start_time = time.time()
        self._lock = threading.Lock()

    def record_success(self, name):
        with self._lock:
            self.done += 1
            self.success += 1
            self._log_progress()

    def record_failure(self, name):
        with self._lock:
            self.done += 1
            self.fail += 1
            self.failed_files.append(name)
            self._log_progress()

    def _log_progress(self):
        elapsed = time.time() - self.start_time
        remaining = self.total - self.done
        avg_time = elapsed / self.done if self.done > 0 else 0
        eta_seconds = remaining * avg_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        logger.info(
            f"   📊 Tiến trình: {self.done}/{self.total} | "
            f"✅ {self.success} | ❌ {self.fail} | "
            f"⏱️ {elapsed_str} | ETA: {eta_str} | "
            f"Avg: {avg_time:.1f}s/file"
        )


# ============================================================
# CORE FUNCTION - Transcribe 1 file
# ============================================================

def transcribe_audio(client: genai.Client, audio_path: str, model_name: str, max_retries: int = 2) -> str | None:
    """
    Upload audio lên Gemini File API, gọi model để transcribe,
    trả về text thuần túy (mỗi câu một dòng).
    """
    file_name = os.path.basename(audio_path)
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    remote_file = None
    temp_link = None
    # Dùng thread id để tránh xung đột symlink giữa các thread
    thread_id = threading.current_thread().ident

    try:
        # 1. Upload file lên Gemini
        # Workaround: google-genai SDK không hỗ trợ tên file Unicode
        # Tạo symlink tạm với tên ASCII để upload
        logger.info(f"   ☁️  Đang upload: {file_name} ({file_size_mb:.1f} MB)")
        t_upload_start = time.time()
        try:
            file_name.encode('ascii')
            upload_path = audio_path
        except UnicodeEncodeError:
            ext = Path(audio_path).suffix
            temp_link = os.path.join(tempfile.gettempdir(), f"gemini_upload_{thread_id}{ext}")
            if os.path.exists(temp_link):
                os.remove(temp_link)
            os.symlink(os.path.abspath(audio_path), temp_link)
            upload_path = temp_link
            logger.debug(f"   Tạo symlink tạm: {temp_link}")
        remote_file = client.files.upload(file=upload_path)
        upload_time = time.time() - t_upload_start
        logger.info(f"   ☁️  Upload xong ({upload_time:.1f}s)")

        # 2. Chờ xử lý xong
        logger.debug(f"   Chờ Gemini xử lý file...")
        t_process_start = time.time()
        wait_count = 0
        while remote_file.state.name == "PROCESSING":
            time.sleep(2)
            wait_count += 1
            remote_file = client.files.get(name=remote_file.name)
            if wait_count % 15 == 0:  # Log mỗi 30 giây
                logger.debug(f"   Vẫn đang xử lý... ({wait_count * 2}s)")

        process_time = time.time() - t_process_start

        if remote_file.state.name == "FAILED":
            logger.error(f"   ❌ Gemini xử lý file thất bại. State: {remote_file.state.name}")
            return None

        logger.info(f"   ✨ File sẵn sàng ({process_time:.1f}s). Đang transcribe...")

        # 3. Gọi Gemini
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        ]

        total_attempts = max_retries + 1
        for attempt in range(total_attempts):
            try:
                logger.debug(f"   Attempt {attempt+1}/{total_attempts} - Gọi generate_content...")
                t_gen_start = time.time()

                response = client.models.generate_content(
                    model=model_name,
                    contents=[remote_file, TRANSCRIBE_PROMPT],
                    config=types.GenerateContentConfig(
                        max_output_tokens=65536,
                        safety_settings=safety_settings,
                    ),
                )

                gen_time = time.time() - t_gen_start
                logger.debug(f"   generate_content hoàn tất ({gen_time:.1f}s)")

                # Kiểm tra finish reason
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                if finish_reason != "STOP":
                    if finish_reason == "RECITATION":
                        logger.error(f"   ⛔ Bị chặn bản quyền (Copyright): {file_name}")
                        return None

                    logger.warning(
                        f"   ⚠️  [{attempt+1}/{total_attempts}] Finish Reason: {finish_reason}"
                    )
                    if attempt < total_attempts - 1:
                        wait_time = 5 * (attempt + 1)
                        logger.info(f"   ⏳ Chờ {wait_time}s trước khi retry...")
                        time.sleep(wait_time)
                        continue
                    return None

                # Lấy text và clean
                text_result = response.text.strip()

                # Xóa markdown code block nếu model vô tình thêm
                text_result = re.sub(r"^```\w*\s*", "", text_result)
                text_result = re.sub(r"\s*```\s*$", "", text_result)
                text_result = text_result.strip()

                line_count = text_result.count("\n") + 1
                char_count = len(text_result)
                logger.info(f"   📝 Transcribe thành công: {line_count} dòng, {char_count} ký tự ({gen_time:.1f}s)")

                # Log usage metadata nếu có
                try:
                    usage = response.usage_metadata
                    logger.debug(f"   Token usage - prompt: {usage.prompt_token_count}, "
                                f"candidates: {usage.candidates_token_count}, "
                                f"total: {usage.total_token_count}")
                except Exception:
                    pass

                return text_result

            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__

                if "429" in error_str or "ResourceExhausted" in error_type:
                    logger.warning(f"   🔥 [{attempt+1}/{total_attempts}] Rate limit (429): {e}")
                    wait_time = 60 * (attempt + 1)
                    logger.info(f"   ⏳ Chờ {wait_time}s trước khi retry (rate limit)...")
                    time.sleep(wait_time)
                elif "500" in error_str or "InternalServerError" in error_type:
                    logger.warning(f"   🔥 [{attempt+1}/{total_attempts}] Server Error 500: {e}")
                    wait_time = 10 * (attempt + 1)
                    logger.info(f"   ⏳ Chờ {wait_time}s trước khi retry...")
                    time.sleep(wait_time)
                elif "503" in error_str or "ServiceUnavailable" in error_type:
                    logger.warning(f"   🔥 [{attempt+1}/{total_attempts}] Service Unavailable 503: {e}")
                    wait_time = 10 * (attempt + 1)
                    logger.info(f"   ⏳ Chờ {wait_time}s trước khi retry...")
                    time.sleep(wait_time)
                elif "DeadlineExceeded" in error_type or "timeout" in error_str.lower():
                    logger.warning(f"   🔥 [{attempt+1}/{total_attempts}] Timeout: {e}")
                    wait_time = 15 * (attempt + 1)
                    logger.info(f"   ⏳ Chờ {wait_time}s trước khi retry...")
                    time.sleep(wait_time)
                elif "ValueError" in error_type:
                    logger.error(f"   ❌ ValueError: {e} (có thể bị chặn bởi safety filter)")
                    logger.debug(f"   Traceback: {traceback.format_exc()}")
                    return None
                else:
                    logger.error(f"   ❌ Lỗi không xác định khi generate: {error_type}: {e}")
                    logger.debug(f"   Traceback: {traceback.format_exc()}")
                    return None

        logger.error(f"   ❌ Hết {total_attempts} lần thử. Bỏ qua file này.")
        return None

    except Exception as e:
        logger.error(f"   ❌ Lỗi upload/setup: {type(e).__name__}: {e}")
        logger.debug(f"   Traceback: {traceback.format_exc()}")
        return None

    finally:
        # Dọn dẹp symlink tạm
        if temp_link and os.path.exists(temp_link):
            try:
                os.remove(temp_link)
            except Exception:
                pass
        # Dọn dẹp file trên Gemini cloud
        if remote_file:
            try:
                client.files.delete(name=remote_file.name)
                logger.debug(f"   🗑️  Đã xóa file trên cloud: {remote_file.name}")
            except Exception as e:
                logger.debug(f"   ⚠️ Không xóa được file cloud: {e}")


# ============================================================
# WORKER FUNCTION - Xử lý 1 file (chạy trong thread)
# ============================================================

def process_one_file(client, f, txt_dir, model_name, max_retries, tracker, idx, total):
    """Worker function xử lý 1 file. Chạy trong thread pool."""
    name = f["name"]
    mp3_path = f["path"]
    txt_path = os.path.join(txt_dir, name + ".txt")

    logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"[{idx}/{total}] 🎵 {name}")
    logger.info(f"   MP3 : {mp3_path} ({f['size_mb']:.1f} MB)")
    logger.info(f"   → TXT: {txt_path}")

    file_start_time = time.time()

    try:
        text_result = transcribe_audio(client, mp3_path, model_name, max_retries=max_retries)
        elapsed = time.time() - file_start_time

        if text_result:
            # Lưu file
            with open(txt_path, "w", encoding="utf-8") as fout:
                fout.write(text_result + "\n")

            line_count = text_result.count("\n") + 1
            char_count = len(text_result)
            logger.info(f"   ✅ [{name}] Thành công! {line_count} dòng, {char_count} ký tự ({elapsed:.1f}s)")
            tracker.record_success(name)
            return True
        else:
            logger.error(f"   ❌ [{name}] Thất bại! ({elapsed:.1f}s) - Không nhận được kết quả")
            tracker.record_failure(name)
            return False

    except Exception as e:
        elapsed = time.time() - file_start_time
        logger.error(f"   ❌ [{name}] Lỗi ngoại lệ: {type(e).__name__}: {e} ({elapsed:.1f}s)")
        logger.debug(f"   Traceback: {traceback.format_exc()}")
        tracker.record_failure(name)
        return False


# ============================================================
# BATCH PROCESSING
# ============================================================

def find_mp3_files(mp3_dir):
    """Tìm tất cả file MP3 trong thư mục."""
    files = []
    for f in sorted(os.listdir(mp3_dir)):
        ext = Path(f).suffix.lower()
        if ext in AUDIO_EXTENSIONS:
            files.append({
                "name": os.path.splitext(f)[0],
                "filename": f,
                "path": os.path.join(mp3_dir, f),
                "size_mb": os.path.getsize(os.path.join(mp3_dir, f)) / (1024 * 1024),
            })
    return files


def run_batch(client: genai.Client, mp3_dir, txt_dir, model_name, max_retries=2, force=False, workers=5):
    """Chạy batch transcribe song song."""

    # Tạo thư mục output nếu chưa có
    os.makedirs(txt_dir, exist_ok=True)

    # Tìm file audio
    all_files = find_mp3_files(mp3_dir)

    logger.info("=" * 60)
    logger.info("🚀 BATCH TRANSCRIBE - BẮT ĐẦU (SONG SONG)")
    logger.info(f"   MP3 dir  : {mp3_dir}")
    logger.info(f"   TXT dir  : {txt_dir}")
    logger.info(f"   Model    : {model_name}")
    logger.info(f"   Workers  : {workers} luồng song song")
    logger.info(f"   Retry    : {max_retries}")
    logger.info(f"   Tổng file: {len(all_files)}")
    total_size = sum(f["size_mb"] for f in all_files)
    logger.info(f"   Tổng dung lượng: {total_size:.1f} MB")
    logger.info(f"   Log file : {log_filepath}")
    logger.info(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Kiểm tra file đã xử lý (resume)
    if not force:
        pending_files = []
        skipped = 0
        skipped_names = []
        for f in all_files:
            txt_path = os.path.join(txt_dir, f["name"] + ".txt")
            if os.path.exists(txt_path):
                # Kiểm tra file txt có rỗng không
                if os.path.getsize(txt_path) > 0:
                    skipped += 1
                    skipped_names.append(f["name"])
                else:
                    logger.warning(f"   ⚠️ File TXT rỗng, sẽ chạy lại: {f['name']}")
                    pending_files.append(f)
            else:
                pending_files.append(f)
        if skipped > 0:
            logger.info(f"⏭️  Bỏ qua {skipped} file đã có TXT (dùng --force để chạy lại)")
            logger.debug(f"   Danh sách đã skip: {skipped_names}")
        all_files = pending_files

    if not all_files:
        logger.info("✅ Không có file nào cần xử lý!")
        return

    total = len(all_files)
    logger.info(f"📋 Cần xử lý: {total} file")
    pending_size = sum(f["size_mb"] for f in all_files)
    logger.info(f"   Dung lượng cần xử lý: {pending_size:.1f} MB")
    logger.info("")

    # Liệt kê tất cả file cần xử lý vào log
    logger.debug("📋 DANH SÁCH FILE CẦN XỬ LÝ:")
    for i, f in enumerate(all_files, 1):
        logger.debug(f"   {i:3d}. {f['filename']} ({f['size_mb']:.1f} MB)")
    logger.debug("")

    # ============================================================
    # XỬ LÝ SONG SONG VỚI THREADPOOLEXECUTOR
    # ============================================================
    tracker = ProgressTracker(total)
    total_start_time = time.time()

    logger.info(f"🔀 Khởi chạy {workers} worker threads...")
    logger.info("")

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="Worker") as executor:
        futures = {}
        for idx, f in enumerate(all_files, 1):
            future = executor.submit(
                process_one_file,
                client, f, txt_dir, model_name, max_retries, tracker, idx, total
            )
            futures[future] = f["name"]

        # Chờ tất cả hoàn thành
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"   ❌ [{name}] Worker exception: {type(e).__name__}: {e}")
                logger.debug(f"   Traceback: {traceback.format_exc()}")

    # ============================================================
    # TỔNG KẾT
    # ============================================================
    total_elapsed = time.time() - total_start_time
    total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))
    avg_per_file = total_elapsed / total if total > 0 else 0
    effective_speed = total / (total_elapsed / 60) if total_elapsed > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("🏁 BATCH TRANSCRIBE - KẾT THÚC")
    logger.info(f"   🕐 Bắt đầu  : {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   🕐 Kết thúc  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   ⏱️  Tổng thời gian: {total_elapsed_str}")
    logger.info(f"   ⏱️  Trung bình: {avg_per_file:.1f}s / file (wall clock)")
    logger.info(f"   🔀 Workers  : {workers} luồng song song")
    logger.info(f"   🚀 Tốc độ   : {effective_speed:.1f} file/phút")
    logger.info(f"   ✅ Thành công: {tracker.success}/{total}")
    logger.info(f"   ❌ Thất bại : {tracker.fail}/{total}")

    if tracker.failed_files:
        logger.info(f"   📋 Danh sách file LỖI ({len(tracker.failed_files)}):")
        for name in tracker.failed_files:
            logger.info(f"      - {name}")

    logger.info(f"   📁 Log file : {log_filepath}")
    logger.info("=" * 60)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Transcribe - Xử lý hàng loạt MP3 → TXT bằng Gemini (song song)")
    parser.add_argument("--mp3-dir", default=DEFAULT_MP3_DIR, help="Thư mục chứa file MP3")
    parser.add_argument("--txt-dir", default=DEFAULT_TXT_DIR, help="Thư mục output TXT")
    parser.add_argument("--model", default=None, help="Model Gemini (mặc định: đọc từ .env)")
    parser.add_argument("--retry", type=int, default=2, help="Số lần retry khi lỗi (mặc định: 2)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Số luồng song song (mặc định: 5)")
    parser.add_argument("--force", action="store_true", help="Chạy lại tất cả, kể cả file đã có TXT")

    args = parser.parse_args()

    # Model
    model_name = args.model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        logger.error("❌ Chưa cấu hình GOOGLE_API_KEY trong file .env")
        logger.error(f"   → Mở file: {SCRIPT_DIR / '.env'} và điền API key.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    logger.info(f"🔑 API Key configured. Model: {model_name}")

    run_batch(
        client, args.mp3_dir, args.txt_dir, model_name,
        max_retries=args.retry, force=args.force, workers=args.workers
    )

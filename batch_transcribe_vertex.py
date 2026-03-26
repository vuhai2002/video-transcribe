#!/usr/bin/env python3
"""
Batch Transcribe (Vertex AI) - Xử lý hàng loạt file MP3 → TXT bằng Gemini trên Vertex AI
Sử dụng Google Cloud $300 Free Trial Credit thay vì API Key từ AI Studio.

Thay đổi so với batch_transcribe.py (AI Studio):
  - Xác thực bằng Service Account (JSON key) thay vì API Key
  - Upload audio lên Google Cloud Storage (GCS) thay vì Gemini File API
  - Gọi Gemini qua Vertex AI endpoint

Usage:
    python3 batch_transcribe_vertex.py
    python3 batch_transcribe_vertex.py --workers 5
    python3 batch_transcribe_vertex.py --mp3-dir /path/to/mp3 --txt-dir /path/to/txt
    python3 batch_transcribe_vertex.py --force
"""

import os
import sys
import re
import time
import argparse
import logging
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.cloud import storage
from google.oauth2 import service_account

# ============================================================
# CẤU HÌNH
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

DEFAULT_MP3_DIR = "/home/vuhai/data/mp3"
DEFAULT_TXT_DIR = "/home/vuhai/data/txt"
DEFAULT_WORKERS = 5

# ============================================================
# LOGGING SETUP
# ============================================================
log_filename = f"batch_transcribe_vertex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)

file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

logger = logging.getLogger("batch-transcribe-vertex")

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("google.cloud.storage").setLevel(logging.WARNING)

# ============================================================
# PROMPT
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

MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
}

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
]


# ============================================================
# PROGRESS TRACKER
# ============================================================
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
# AUTHENTICATION - Service Account
# ============================================================
def load_credentials():
    """Load service account credentials từ JSON key file."""
    sa_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not sa_key_path:
        logger.error("❌ Chưa cấu hình GOOGLE_APPLICATION_CREDENTIALS trong .env")
        logger.error("   → Đặt đường dẫn đến file JSON key của Service Account.")
        sys.exit(1)

    if not os.path.exists(sa_key_path):
        logger.error(f"❌ Không tìm thấy file key: {sa_key_path}")
        sys.exit(1)

    credentials = service_account.Credentials.from_service_account_file(
        sa_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    logger.info(f"🔑 Service Account: {credentials.service_account_email}")
    return credentials


# ============================================================
# CORE FUNCTION - Transcribe 1 file qua Vertex AI
# ============================================================
def transcribe_audio(
    client: genai.Client,
    storage_client: storage.Client,
    bucket_name: str,
    audio_path: str,
    model_name: str,
    max_retries: int = 2,
) -> str | None:
    """
    Upload audio lên GCS, gọi Gemini qua Vertex AI để transcribe,
    trả về text thuần túy.
    """
    file_name = os.path.basename(audio_path)
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    ext = Path(audio_path).suffix.lower()
    mime_type = MIME_TYPES.get(ext, "audio/mpeg")

    # Thread-safe GCS blob path
    thread_id = threading.current_thread().ident
    blob_name = f"transcribe-temp/{thread_id}/{file_name}"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    try:
        # 1. Upload lên Google Cloud Storage
        logger.info(f"   ☁️  Đang upload lên GCS: {file_name} ({file_size_mb:.1f} MB)")
        t_upload_start = time.time()
        blob.upload_from_filename(audio_path, content_type=mime_type)
        upload_time = time.time() - t_upload_start
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"   ☁️  Upload GCS xong ({upload_time:.1f}s)")

        # 2. Gọi Gemini qua Vertex AI (dùng GCS URI)
        total_attempts = max_retries + 1
        for attempt in range(total_attempts):
            try:
                logger.debug(f"   Attempt {attempt+1}/{total_attempts} - Gọi generate_content...")
                t_gen_start = time.time()

                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type),
                        TRANSCRIBE_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        max_output_tokens=65536,
                        safety_settings=SAFETY_SETTINGS,
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
                text_result = re.sub(r"^```\w*\s*", "", text_result)
                text_result = re.sub(r"\s*```\s*$", "", text_result)
                text_result = text_result.strip()

                line_count = text_result.count("\n") + 1
                char_count = len(text_result)
                logger.info(
                    f"   📝 Transcribe thành công: {line_count} dòng, {char_count} ký tự ({gen_time:.1f}s)"
                )

                # Log usage metadata
                try:
                    usage = response.usage_metadata
                    logger.debug(
                        f"   Token usage - prompt: {usage.prompt_token_count}, "
                        f"candidates: {usage.candidates_token_count}, "
                        f"total: {usage.total_token_count}"
                    )
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
                    logger.error(f"   ❌ Lỗi không xác định: {error_type}: {e}")
                    logger.debug(f"   Traceback: {traceback.format_exc()}")
                    if attempt >= total_attempts - 1:
                        return None

        logger.error(f"   ❌ Hết {total_attempts} lần thử. Bỏ qua file này.")
        return None

    except Exception as e:
        logger.error(f"   ❌ Lỗi upload/setup: {type(e).__name__}: {e}")
        logger.debug(f"   Traceback: {traceback.format_exc()}")
        return None

    finally:
        # Dọn dẹp file trên GCS
        try:
            blob.delete()
            logger.debug(f"   🗑️  Đã xóa file GCS: {blob_name}")
        except Exception as e:
            logger.debug(f"   ⚠️ Không xóa được file GCS: {e}")


# ============================================================
# WORKER FUNCTION
# ============================================================
def process_one_file(client, storage_client, bucket_name, f, txt_dir, model_name, max_retries, tracker, idx, total):
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
        text_result = transcribe_audio(
            client, storage_client, bucket_name,
            mp3_path, model_name, max_retries=max_retries,
        )
        elapsed = time.time() - file_start_time

        if text_result:
            with open(txt_path, "w", encoding="utf-8") as fout:
                fout.write(text_result + "\n")

            line_count = text_result.count("\n") + 1
            char_count = len(text_result)
            logger.info(f"   ✅ [{name}] Thành công! {line_count} dòng, {char_count} ký tự ({elapsed:.1f}s)")
            tracker.record_success(name)
            return True
        else:
            logger.error(f"   ❌ [{name}] Thất bại! ({elapsed:.1f}s)")
            tracker.record_failure(name)
            return False

    except Exception as e:
        elapsed = time.time() - file_start_time
        logger.error(f"   ❌ [{name}] Lỗi: {type(e).__name__}: {e} ({elapsed:.1f}s)")
        logger.debug(f"   Traceback: {traceback.format_exc()}")
        tracker.record_failure(name)
        return False


# ============================================================
# BATCH PROCESSING
# ============================================================
def find_mp3_files(mp3_dir):
    """Tìm tất cả file audio trong thư mục."""
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


def run_batch(client, storage_client, bucket_name, mp3_dir, txt_dir, model_name,
              max_retries=2, force=False, workers=5):
    """Chạy batch transcribe song song qua Vertex AI."""
    os.makedirs(txt_dir, exist_ok=True)

    all_files = find_mp3_files(mp3_dir)

    logger.info("=" * 60)
    logger.info("🚀 BATCH TRANSCRIBE (VERTEX AI) - BẮT ĐẦU")
    logger.info(f"   MP3 dir  : {mp3_dir}")
    logger.info(f"   TXT dir  : {txt_dir}")
    logger.info(f"   Model    : {model_name}")
    logger.info(f"   Workers  : {workers} luồng song song")
    logger.info(f"   GCS Bucket: {bucket_name}")
    logger.info(f"   Retry    : {max_retries}")
    logger.info(f"   Tổng file: {len(all_files)}")
    total_size = sum(f["size_mb"] for f in all_files)
    logger.info(f"   Tổng dung lượng: {total_size:.1f} MB")
    logger.info(f"   Log file : {log_filepath}")
    logger.info(f"   Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Resume: skip file đã có TXT
    if not force:
        pending_files = []
        skipped = 0
        for f in all_files:
            txt_path = os.path.join(txt_dir, f["name"] + ".txt")
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                skipped += 1
            else:
                if os.path.exists(txt_path) and os.path.getsize(txt_path) == 0:
                    logger.warning(f"   ⚠️ File TXT rỗng, sẽ chạy lại: {f['name']}")
                pending_files.append(f)
        if skipped > 0:
            logger.info(f"⏭️  Bỏ qua {skipped} file đã có TXT (dùng --force để chạy lại)")
        all_files = pending_files

    if not all_files:
        logger.info("✅ Không có file nào cần xử lý!")
        return

    total = len(all_files)
    pending_size = sum(f["size_mb"] for f in all_files)
    logger.info(f"📋 Cần xử lý: {total} file ({pending_size:.1f} MB)")
    logger.info("")

    # Log danh sách file
    logger.debug("📋 DANH SÁCH FILE CẦN XỬ LÝ:")
    for i, f in enumerate(all_files, 1):
        logger.debug(f"   {i:3d}. {f['filename']} ({f['size_mb']:.1f} MB)")

    # Chạy song song
    tracker = ProgressTracker(total)
    total_start_time = time.time()

    logger.info(f"🔀 Khởi chạy {workers} worker threads...")
    logger.info("")

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="Worker") as executor:
        futures = {}
        for idx, f in enumerate(all_files, 1):
            future = executor.submit(
                process_one_file,
                client, storage_client, bucket_name,
                f, txt_dir, model_name, max_retries, tracker, idx, total,
            )
            futures[future] = f["name"]

        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"   ❌ [{name}] Worker exception: {type(e).__name__}: {e}")

    # Tổng kết
    total_elapsed = time.time() - total_start_time
    total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))
    avg_per_file = total_elapsed / total if total > 0 else 0
    effective_speed = total / (total_elapsed / 60) if total_elapsed > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("🏁 BATCH TRANSCRIBE (VERTEX AI) - KẾT THÚC")
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
    parser = argparse.ArgumentParser(
        description="Batch Transcribe (Vertex AI) - MP3 → TXT bằng Gemini trên Google Cloud"
    )
    parser.add_argument("--mp3-dir", default=DEFAULT_MP3_DIR, help="Thư mục chứa file MP3")
    parser.add_argument("--txt-dir", default=DEFAULT_TXT_DIR, help="Thư mục output TXT")
    parser.add_argument("--model", default=None, help="Model Gemini (mặc định: đọc từ .env)")
    parser.add_argument("--retry", type=int, default=2, help="Số lần retry khi lỗi")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Số luồng song song")
    parser.add_argument("--force", action="store_true", help="Chạy lại tất cả, kể cả file đã có TXT")

    args = parser.parse_args()

    # --- Validate cấu hình ---
    # 1. Project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("❌ Chưa cấu hình GOOGLE_CLOUD_PROJECT trong .env")
        logger.error("   → Thêm: GOOGLE_CLOUD_PROJECT=transcribe-491102")
        sys.exit(1)

    # 2. Location
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    # 3. GCS Bucket
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        logger.error("❌ Chưa cấu hình GCS_BUCKET_NAME trong .env")
        logger.error("   → Tạo bucket trên Google Cloud Storage rồi thêm tên vào .env")
        sys.exit(1)

    # 4. Model
    model_name = args.model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # 5. Service Account credentials
    credentials = load_credentials()

    # --- Khởi tạo clients ---
    # Gemini client qua Vertex AI
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials,
    )
    logger.info(f"🔗 Vertex AI: project={project_id}, location={location}")
    logger.info(f"🤖 Model: {model_name}")

    # Google Cloud Storage client
    storage_client = storage.Client(
        credentials=credentials,
        project=project_id,
    )
    logger.info(f"🪣 GCS Bucket: {bucket_name}")

    # --- Chạy batch ---
    run_batch(
        client, storage_client, bucket_name,
        args.mp3_dir, args.txt_dir, model_name,
        max_retries=args.retry, force=args.force, workers=args.workers,
    )

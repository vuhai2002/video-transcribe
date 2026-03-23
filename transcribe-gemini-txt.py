#!/usr/bin/env python3
"""
Gemini Audio → Text Transcription (Simple)
==========================================
Gửi file MP3 lên Google Gemini API và nhận lại file TXT
(mỗi câu một dòng, ngắt tự nhiên theo ngữ điệu).

File TXT đầu ra dùng để nạp cho align_whisperx.py.

Cấu hình:
    - GOOGLE_API_KEY và GEMINI_MODEL lưu trong file .env cùng thư mục.

Usage:
    python3 transcribe-gemini-txt.py --audio "path/to/file.mp3"
    python3 transcribe-gemini-txt.py --audio "path/to/file.mp3" --output "path/to/output.txt"
    python3 transcribe-gemini-txt.py --audio "path/to/file.mp3" --model gemini-2.5-pro
"""

import os
import sys
import re
import time
import argparse
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

# ============================================================
# SETUP
# ============================================================

# Load .env từ cùng thư mục với script
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transcribe-gemini")

# Giảm noise từ HTTP libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Định dạng audio hỗ trợ
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

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

# ============================================================
# CORE FUNCTION
# ============================================================

def transcribe_audio(client: genai.Client, audio_path: str, model_name: str, max_retries: int = 3) -> str | None:
    """
    Upload audio lên Gemini File API, gọi model để transcribe,
    trả về text thuần túy (mỗi câu một dòng).
    """
    file_name = os.path.basename(audio_path)
    remote_file = None
    temp_link = None

    try:
        # 1. Upload file lên Gemini
        # Workaround: google-genai SDK không hỗ trợ tên file Unicode
        # Tạo symlink tạm với tên ASCII để upload
        log.info(f"☁️  Đang upload: {file_name}")
        try:
            file_name.encode('ascii')
            upload_path = audio_path
        except UnicodeEncodeError:
            ext = Path(audio_path).suffix
            temp_link = os.path.join(tempfile.gettempdir(), f"gemini_upload_{os.getpid()}{ext}")
            if os.path.exists(temp_link):
                os.remove(temp_link)
            os.symlink(os.path.abspath(audio_path), temp_link)
            upload_path = temp_link
            log.debug(f"   Tạo symlink tạm: {temp_link}")
        remote_file = client.files.upload(file=upload_path)

        # 2. Chờ xử lý xong
        while remote_file.state.name == "PROCESSING":
            time.sleep(2)
            remote_file = client.files.get(name=remote_file.name)

        if remote_file.state.name == "FAILED":
            log.error("❌ Gemini xử lý file thất bại.")
            return None

        log.info(f"✨ File sẵn sàng. Đang transcribe: {file_name}")

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
                response = client.models.generate_content(
                    model=model_name,
                    contents=[remote_file, TRANSCRIBE_PROMPT],
                    config=types.GenerateContentConfig(
                        max_output_tokens=65536,
                        safety_settings=safety_settings,
                    ),
                )

                # Kiểm tra finish reason
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                if finish_reason != "STOP":
                    if finish_reason == "RECITATION":
                        log.error(f"⛔ Bị chặn bản quyền (Copyright): {file_name}")
                        return None

                    log.warning(
                        f"⚠️  [{attempt+1}/{total_attempts}] Finish Reason: {finish_reason}"
                    )
                    if attempt < total_attempts - 1:
                        time.sleep(5)
                        continue
                    return None

                # Lấy text và clean
                text_result = response.text.strip()

                # Xóa markdown code block nếu model vô tình thêm
                text_result = re.sub(r"^```\w*\s*", "", text_result)
                text_result = re.sub(r"\s*```\s*$", "", text_result)

                return text_result.strip()

            except Exception as e:
                error_type = type(e).__name__
                if "500" in str(e) or "InternalServerError" in error_type:
                    log.warning(f"🔥 [{attempt+1}/{total_attempts}] Server Error 500. Đang thử lại...")
                    time.sleep(10)
                elif "503" in str(e) or "ServiceUnavailable" in error_type:
                    log.warning(f"🔥 [{attempt+1}/{total_attempts}] Service Unavailable 503. Đang thử lại...")
                    time.sleep(10)
                elif "429" in str(e) or "ResourceExhausted" in error_type:
                    log.warning(f"🔥 [{attempt+1}/{total_attempts}] Rate limit 429. Đang thử lại...")
                    time.sleep(60)
                elif "ValueError" in error_type:
                    log.error(f"❌ ValueError: {e} (có thể bị chặn)")
                    return None
                else:
                    log.error(f"❌ Lỗi không xác định: {e}")
                    return None

        return None

    except Exception as e:
        log.error(f"❌ Lỗi upload/setup: {e}")
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
                log.debug(f"🗑️  Đã xóa file trên cloud: {remote_file.name}")
            except Exception:
                pass


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio thành text bằng Google Gemini"
    )
    parser.add_argument(
        "--audio", required=True,
        help="Đường dẫn file audio (.mp3, .wav, .m4a, ...)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Đường dẫn file TXT đầu ra (mặc định: cùng tên, cùng thư mục với file audio)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Tên model Gemini (mặc định: đọc từ .env GEMINI_MODEL)"
    )
    parser.add_argument(
        "--retry", type=int, default=3,
        help="Số lần retry khi lỗi server (mặc định: 3)"
    )

    args = parser.parse_args()

    # --- Validate audio ---
    audio_path = os.path.abspath(args.audio)
    if not os.path.isfile(audio_path):
        log.error(f"❌ Không tìm thấy file: {audio_path}")
        sys.exit(1)

    ext = Path(audio_path).suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        log.error(f"❌ Định dạng không hỗ trợ: {ext}. Hỗ trợ: {AUDIO_EXTENSIONS}")
        sys.exit(1)

    # --- Output path ---
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = os.path.splitext(audio_path)[0] + ".txt"

    # --- Model ---
    model_name = args.model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # --- API Key ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        log.error("❌ Chưa cấu hình GOOGLE_API_KEY trong file .env")
        log.error(f"   → Mở file: {SCRIPT_DIR / '.env'} và điền API key.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # --- Start ---
    log.info("=" * 50)
    log.info("🚀 GEMINI AUDIO → TEXT TRANSCRIPTION")
    log.info(f"   Audio : {audio_path}")
    log.info(f"   Output: {output_path}")
    log.info(f"   Model : {model_name}")
    log.info("=" * 50)

    text_result = transcribe_audio(client, audio_path, model_name, max_retries=args.retry)

    if not text_result:
        log.error("❌ Transcribe thất bại. Không có kết quả.")
        sys.exit(1)

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_result + "\n")

    # --- Stats ---
    line_count = text_result.count("\n") + 1
    log.info(f"✅ Hoàn tất! Đã lưu {line_count} dòng → {output_path}")


if __name__ == "__main__":
    main()

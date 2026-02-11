#!/usr/bin/env python3
"""
Gemini Audio Transcription Pipeline
===================================
Quy trình:
1. Quét file MP3 trên Google Drive (qua Rclone).
2. Worker (đa luồng) tải file về VM.
3. Upload lên Gemini File API -> Chờ active.
4. Gọi Gemini Flash/Pro để tạo SRT.
5. Upload SRT lên Google Drive.
6. Xóa file tạm (trên VM và trên Gemini).

Usage:
    export GOOGLE_API_KEY="AIzaSy..."
    python3 transcribe-gemini.py --drive-input "Output/Audio" --drive-output "Output" --workers 3
"""

import os
import sys
import json
import time
import argparse
import subprocess
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thư viện Google
import google.generativeai as genai
from google.api_core import retry

# --- CẤU HÌNH MẶC ĐỊNH ---
# Nên dùng gemini-3-pro-preview
DEFAULT_MODEL = "gemini-3-pro-preview" 
RCLONE_REMOTE = "gdrive"
LOCAL_WORK_DIR = os.path.expanduser("~/gemini-work")

# Định dạng audio hỗ trợ
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
VN_TZ = timezone(timedelta(hours=7))

# ============================================================
# LOGGING SETUP
# ============================================================
def setup_logging():
    log_dir = os.path.join(LOCAL_WORK_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"gemini_{datetime.now(VN_TZ).strftime('%Y%m%d_%H%M%S')}.log")
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger("gemini-transcribe")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# ============================================================
# RCLONE UTILS (Giữ nguyên logic ổn định từ file cũ)
# ============================================================
def rclone_run(cmd_args, logger):
    cmd = ["rclone"] + cmd_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Rclone error: {result.stderr.strip()}")
        return False
    return True

def get_remote_files(remote_path, logger):
    """Lấy danh sách file audio trên Drive"""
    logger.info(f"🔍 Scanning: {RCLONE_REMOTE}:{remote_path}")
    cmd = ["lsjson", f"{RCLONE_REMOTE}:{remote_path}", "--recursive", "--files-only"]
    result = subprocess.run(["rclone"] + cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error("Failed to list files.")
        return []
        
    try:
        files = json.loads(result.stdout)
        audio_files = [f for f in files if Path(f["Path"]).suffix.lower() in AUDIO_EXTENSIONS]
        # Sort theo tên
        audio_files.sort(key=lambda x: x["Path"])
        return audio_files
    except Exception as e:
        logger.error(f"JSON Parse error: {e}")
        return []

def check_exists_on_drive(remote_srt_path, logger):
    """Kiểm tra xem file SRT đã tồn tại trên Drive chưa"""
    cmd = ["lsjson", f"{RCLONE_REMOTE}:{remote_srt_path}"]
    result = subprocess.run(["rclone"] + cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip() != "":
        try:
            items = json.loads(result.stdout)
            return len(items) > 0
        except:
            return False
    return False

# ============================================================
# GEMINI CORE LOGIC
# ============================================================
def call_gemini_api(local_audio_path, model_name, logger):
    """
    Core logic: Upload -> Wait -> Generate -> Delete Remote
    """
    file_name = os.path.basename(local_audio_path)
    remote_file = None
    
    try:
        # 1. Upload File
        logger.info(f"   ☁️  [Gemini] Uploading: {file_name}")
        remote_file = genai.upload_file(path=local_audio_path)
        
        # 2. Wait for Processing
        while remote_file.state.name == "PROCESSING":
            time.sleep(2)
            remote_file = genai.get_file(remote_file.name)
            
        if remote_file.state.name == "FAILED":
            raise ValueError("Gemini File Processing FAILED.")
            
        logger.info(f"   ✨ [Gemini] Ready. Generating SRT...")

        # 3. Generate Content
        model = genai.GenerativeModel(model_name)
        
        # Prompt kỹ thuật (giữ nguyên logic test.py)
        prompt = """
        Nghe file âm thanh này và tạo phụ đề chính xác từng từ (verbatim).
        Output bắt buộc phải ở định dạng chuẩn SRT (SubRip).
        Không bao gồm bất kỳ lời dẫn hay markdown code block nào (như ```srt), chỉ trả về plain text của nội dung SRT.
        Kiểm tra kỹ timeline, đảm bảo không bị chồng chéo.
        """
        
        # Safety settings: Tắt filter để tránh chặn nội dung tôn giáo/triết học
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Retry policy cho API call
        response = model.generate_content(
            [remote_file, prompt],
            request_options={"timeout": 1200}, # Tăng timeout cho file dài
            safety_settings=safety_settings
        )
        
        text_result = response.text
        
        # 4. Clean formatting (FIXED REGEX HERE)
        # Xóa ```srt ở đầu
        text_result = re.sub(r'^```srt\s*', '', text_result, flags=re.IGNORECASE)
        # Xóa ``` ở đầu
        text_result = re.sub(r'^```\s*', '', text_result)
        # Xóa ``` ở cuối
        text_result = re.sub(r'```\s*$', '', text_result)
        
        return text_result.strip()

    except Exception as e:
        logger.error(f"   ❌ [Gemini Error] {file_name}: {e}")
        return None
    finally:
        # 5. Dọn dẹp file trên cloud (quan trọng để không đầy quota)
        if remote_file:
            try:
                genai.delete_file(remote_file.name)
                # logger.debug(f"   🗑️  [Gemini] Deleted cloud file: {remote_file.name}")
            except:
                pass

# ============================================================
# WORKER PIPELINE
# ============================================================
def process_single_file(file_info, drive_input, drive_output, model_name, logger):
    """
    Quy trình xử lý 1 file: Tải -> Gemini -> Lưu SRT -> Upload -> Xóa
    """
    rel_path = file_info["Path"]
    file_name = os.path.basename(rel_path)
    base_name = os.path.splitext(file_name)[0]
    
    # Đường dẫn cục bộ
    local_audio = os.path.join(LOCAL_WORK_DIR, "downloads", file_name)
    local_srt = os.path.join(LOCAL_WORK_DIR, "output", f"{base_name}.srt")
    
    # Đường dẫn remote output
    remote_srt_rel = os.path.join("SRT", os.path.dirname(rel_path), f"{base_name}.srt")
    remote_srt_full = f"{drive_output}/{remote_srt_rel}".replace("\\", "/") # Rclone dùng forward slash

    try:
        # Check if output exists
        if check_exists_on_drive(remote_srt_full, logger):
            logger.info(f"⏭️  Skipping (Exists): {base_name}.srt")
            return "SKIPPED"

        logger.info(f"🎬 Start: {file_name}")
        
        # 1. Download
        os.makedirs(os.path.dirname(local_audio), exist_ok=True)
        if not rclone_run(["copyto", f"{RCLONE_REMOTE}:{drive_input}/{rel_path}", local_audio], logger):
            return "DOWNLOAD_FAILED"
            
        # 2. Transcribe via Gemini
        srt_content = call_gemini_api(local_audio, model_name, logger)
        
        if not srt_content:
            return "GEMINI_FAILED"
            
        # 3. Save Local SRT
        os.makedirs(os.path.dirname(local_srt), exist_ok=True)
        with open(local_srt, "w", encoding="utf-8") as f:
            f.write(srt_content)
            
        # 4. Upload SRT
        if not rclone_run(["copyto", local_srt, f"{RCLONE_REMOTE}:{remote_srt_full}"], logger):
            return "UPLOAD_FAILED"
            
        logger.info(f"✅ DONE: {base_name}.srt")
        return "SUCCESS"

    except Exception as e:
        logger.error(f"❌ Exception processing {file_name}: {e}")
        return "ERROR"
        
    finally:
        # 5. Cleanup Local Files (Giải phóng ổ cứng VM)
        if os.path.exists(local_audio): os.remove(local_audio)
        if os.path.exists(local_srt): os.remove(local_srt)

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Gemini Transcription Worker")
    parser.add_argument("--drive-input", required=True, help="Folder Audio trên Drive (e.g. Output/Audio)")
    parser.add_argument("--drive-output", required=True, help="Folder Output gốc trên Drive (e.g. Output)")
    parser.add_argument("--api-key", help="Gemini API Key (hoặc set env GOOGLE_API_KEY)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=3, help="Số luồng chạy song song (Default: 3)")
    
    args = parser.parse_args()
    
    # Config API Key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: Missing API Key. Pass --api-key or set GOOGLE_API_KEY env var.")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    
    logger = setup_logging()
    logger.info("="*50)
    logger.info(f"🚀 GEMINI TRANSCRIPTION STARTED")
    logger.info(f"   Model:   {args.model}")
    logger.info(f"   Workers: {args.workers}")
    logger.info(f"   Input:   {args.drive_input}")
    logger.info("="*50)

    # 1. List Files
    all_files = get_remote_files(args.drive_input, logger)
    logger.info(f"📁 Found {len(all_files)} audio files.")
    
    if not all_files:
        return

    # 2. Run Parallel Processing
    success = 0
    failed = 0
    skipped = 0
    
    # ThreadPoolExecutor giúp chạy song song
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks
        future_to_file = {
            executor.submit(
                process_single_file, 
                f, 
                args.drive_input, 
                args.drive_output, 
                args.model, 
                logger
            ): f for f in all_files
        }
        
        # Process results as they finish
        for future in as_completed(future_to_file):
            status = future.result()
            if status == "SUCCESS": success += 1
            elif status == "SKIPPED": skipped += 1
            else: failed += 1
            
            # Tính toán tiến độ
            total_done = success + failed + skipped
            print(f"📊 Progress: {total_done}/{len(all_files)} | ✅ {success} | ⏭️ {skipped} | ❌ {failed}", end="\r")

    logger.info("\n" + "="*50)
    logger.info(f"🏁 COMPLETED. Success: {success}, Failed: {failed}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
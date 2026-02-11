#!/usr/bin/env python3
"""
Audio Transcription Pipeline (Spot Instance Safe)
===================================================
Tải MP3 từ Google Drive, transcribe bằng faster-whisper (với Silero VAD),
upload kết quả (JSON, SRT) lên Drive, và dọn dẹp.

Quy trình: [transcribe.py]    MP3 → Transcribe → JSON/SRT → Drive

Tính năng:
- Input: Folder MP3 trên Google Drive (đã được convert từ convert_audio.py)
- faster-whisper + Silero VAD: Transcription chính xác, chống hallucination
- Anti-hallucination: condition_on_previous_text=False, strict thresholds, post-processing filter
- Post-processing: Tự động phát hiện & loại bỏ segments lặp/hallucinate
- Pipeline gối đầu: Tải MP3 B trong khi Transcribe MP3 A
- Checkpoint trên Google Drive: An toàn khi Spot Instance bị thu hồi
- Auto-resume: Tự động tiếp tục từ file cuối cùng khi VM khởi động lại
- Log chi tiết: Ghi lại mọi bước xử lý
- Xóa sạch file trên VM sau khi xử lý xong

Usage:
    python3 transcribe.py --drive-input "Output/Audio" --drive-output "Output"
    python3 transcribe.py --drive-input "Output/Audio" --drive-output "Output" --test
    python3 transcribe.py --drive-input "Output/Audio" --drive-output "Output" --model medium
"""

import os
import sys
import json
import re
import time
import argparse
import subprocess
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone
from queue import Queue

# Vietnam timezone (UTC+7)
VN_TZ = timezone(timedelta(hours=7))

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_WHISPER_MODEL = "large-v3"
RCLONE_REMOTE = "gdrive"
LOCAL_WORK_DIR = os.path.expanduser("~/transcribe-work")
CHECKPOINT_FILENAME = "checkpoint.json"

# Supported audio formats (input)
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

# ============================================================
# LOGGING SETUP
# ============================================================
def setup_logging():
    """Setup logging to both file and console with detailed format."""
    log_dir = os.path.join(LOCAL_WORK_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"transcribe_{datetime.now(VN_TZ).strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-7s] [%(threadName)-10s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (all logs)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger("transcribe")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file: {log_file}")
    return logger

# ============================================================
# RCLONE HELPERS
# ============================================================
def rclone_run(args, desc="rclone", logger=None):
    """Run an rclone command and return (success, stdout)."""
    cmd = ["rclone"] + args
    if logger:
        logger.debug(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        if logger:
            logger.error(f"{desc} failed: {result.stderr.strip()}")
        return False, result.stderr.strip()
    
    return True, result.stdout.strip()


def rclone_list_files(remote_path, logger):
    """
    List all audio files in a remote path using rclone lsjson.
    Returns list of dicts: [{"Path": "audio.mp3", "Size": 123456, "Name": "audio.mp3"}, ...]
    """
    logger.info(f"Scanning remote path: {RCLONE_REMOTE}:{remote_path}")
    
    ok, output = rclone_run(
        ["lsjson", f"{RCLONE_REMOTE}:{remote_path}", "--recursive", "--files-only"],
        desc="List files",
        logger=logger
    )
    
    if not ok:
        logger.error(f"Failed to list files from {remote_path}")
        return []
    
    try:
        all_files = json.loads(output)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse rclone output: {e}")
        return []
    
    # Filter audio files only
    audio_files = [
        f for f in all_files
        if Path(f["Path"]).suffix.lower() in AUDIO_EXTENSIONS
    ]
    
    # Sort A-Z by path
    audio_files.sort(key=lambda f: f["Path"])
    
    logger.info(f"Found {len(audio_files)} audio files (sorted A-Z)")
    for i, f in enumerate(audio_files[:5]):
        size_mb = f.get("Size", 0) / (1024 * 1024)
        logger.debug(f"  [{i+1}] {f['Path']} ({size_mb:.1f} MB)")
    if len(audio_files) > 5:
        logger.debug(f"  ... and {len(audio_files) - 5} more")
    
    return audio_files


def rclone_download(remote_path, local_path, logger):
    """Download a single file from remote to local."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    logger.info(f"⬇️  Downloading: {remote_path}")
    start = time.time()
    
    ok, output = rclone_run(
        ["copyto", f"{RCLONE_REMOTE}:{remote_path}", local_path, "--progress"],
        desc=f"Download {Path(remote_path).name}",
        logger=logger
    )
    
    elapsed = time.time() - start
    if ok:
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"⬇️  Downloaded: {Path(remote_path).name} ({size_mb:.1f} MB in {elapsed:.1f}s)")
    else:
        logger.error(f"⬇️  Download FAILED: {remote_path} ({elapsed:.1f}s)")
    
    return ok


def rclone_upload(local_path, remote_path, logger):
    """Upload a single file from local to remote."""
    logger.info(f"⬆️  Uploading: {Path(local_path).name} → {remote_path}")
    start = time.time()
    
    ok, output = rclone_run(
        ["copyto", local_path, f"{RCLONE_REMOTE}:{remote_path}"],
        desc=f"Upload {Path(local_path).name}",
        logger=logger
    )
    
    elapsed = time.time() - start
    if ok:
        logger.info(f"⬆️  Uploaded: {Path(local_path).name} ({elapsed:.1f}s)")
    else:
        logger.error(f"⬆️  Upload FAILED: {Path(local_path).name} ({elapsed:.1f}s)")
    
    return ok

# ============================================================
# CHECKPOINT MANAGEMENT (Saved on Google Drive)
# ============================================================
class CheckpointManager:
    """
    Manages checkpoint on Google Drive for Spot Instance safety.
    
    Checkpoint format:
    {
        "audios": {
            "path/to/audio.mp3": {
                "status": "done" | "processing" | "failed",
                "started_at": "2026-02-10T10:00:00",
                "completed_at": "2026-02-10T10:15:00",
                "processing_time_seconds": 900.0,
                "output_json": "Output/JSON/audio.json",
                "output_srt": "Output/SRT/audio.srt"
            }
        },
        "stats": {
            "total_processed": 100,
            "total_failed": 2,
            "last_updated": "2026-02-10T10:15:00"
        }
    }
    """
    
    def __init__(self, drive_output_path, logger):
        self.logger = logger
        self.drive_checkpoint_path = f"{drive_output_path}/{CHECKPOINT_FILENAME}"
        self.local_checkpoint_path = os.path.join(LOCAL_WORK_DIR, CHECKPOINT_FILENAME)
        self.data = {"audios": {}, "stats": {"total_processed": 0, "total_failed": 0}}
        self._lock = threading.Lock()
    
    def load(self):
        """Load checkpoint from Google Drive."""
        self.logger.info(f"Loading checkpoint from Drive: {self.drive_checkpoint_path}")
        
        ok = rclone_download(
            self.drive_checkpoint_path,
            self.local_checkpoint_path,
            self.logger
        )
        
        if ok and os.path.exists(self.local_checkpoint_path):
            try:
                with open(self.local_checkpoint_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                # Support both old "videos" key and new "audios" key
                if "videos" in self.data and "audios" not in self.data:
                    self.data["audios"] = self.data.pop("videos")
                tracked = len(self.data.get('audios', {}))
                self.logger.info(f"Checkpoint loaded: {tracked} files tracked")
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Checkpoint corrupt, starting fresh: {e}")
                self.data = {"audios": {}, "stats": {"total_processed": 0, "total_failed": 0}}
        else:
            self.logger.info("No existing checkpoint found, starting fresh")
    
    def save(self):
        """Save checkpoint to both local and Google Drive."""
        with self._lock:
            self.data["stats"]["last_updated"] = datetime.now(VN_TZ).isoformat()
            
            # Save locally first
            with open(self.local_checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            # Upload to Drive
            ok = rclone_upload(
                self.local_checkpoint_path,
                self.drive_checkpoint_path,
                self.logger
            )
            
            if ok:
                self.logger.debug("Checkpoint saved to Drive")
            else:
                self.logger.warning("Failed to save checkpoint to Drive (will retry next time)")
    
    def mark_processing(self, audio_path):
        """Mark an audio file as currently being processed."""
        with self._lock:
            self.data["audios"][audio_path] = {
                "status": "processing",
                "started_at": datetime.now(VN_TZ).isoformat(),
                "completed_at": None
            }
        self.save()
        self.logger.info(f"📝 Checkpoint: PROCESSING → {audio_path}")
    
    def mark_done(self, audio_path, processing_time, output_json, output_srt):
        """Mark an audio file as successfully processed."""
        with self._lock:
            self.data["audios"][audio_path] = {
                "status": "done",
                "started_at": self.data["audios"].get(audio_path, {}).get("started_at"),
                "completed_at": datetime.now(VN_TZ).isoformat(),
                "processing_time_seconds": round(processing_time, 1),
                "output_json": output_json,
                "output_srt": output_srt
            }
            self.data["stats"]["total_processed"] = sum(
                1 for v in self.data["audios"].values() if v["status"] == "done"
            )
        self.save()
        self.logger.info(f"✅ Checkpoint: DONE → {audio_path} ({processing_time:.1f}s)")
    
    def mark_failed(self, audio_path, error_message):
        """Mark an audio file as failed."""
        with self._lock:
            self.data["audios"][audio_path] = {
                "status": "failed",
                "started_at": self.data["audios"].get(audio_path, {}).get("started_at"),
                "completed_at": datetime.now(VN_TZ).isoformat(),
                "error": str(error_message)
            }
            self.data["stats"]["total_failed"] = sum(
                1 for v in self.data["audios"].values() if v["status"] == "failed"
            )
        self.save()
        self.logger.error(f"❌ Checkpoint: FAILED → {audio_path}: {error_message}")
    
    def get_status(self, audio_path):
        """Get status of an audio file: 'done', 'processing', 'failed', or None."""
        return self.data.get("audios", {}).get(audio_path, {}).get("status")
    
    def get_pending_audios(self, all_audios):
        """
        Filter audio files that need processing.
        - Skip 'done' audios
        - Re-process 'processing' audios (crashed mid-way)
        - Re-process 'failed' audios (might succeed this time)
        """
        pending = []
        skipped = 0
        retry = 0
        
        for audio in all_audios:
            path = audio["Path"]
            status = self.get_status(path)
            
            if status == "done":
                skipped += 1
            elif status == "processing":
                self.logger.warning(f"🔄 Re-processing (crashed mid-way): {path}")
                retry += 1
                pending.append(audio)
            elif status == "failed":
                self.logger.warning(f"🔄 Retrying (previously failed): {path}")
                retry += 1
                pending.append(audio)
            else:
                pending.append(audio)
        
        self.logger.info(f"Audios: {len(all_audios)} total | {skipped} done | {retry} retry | {len(pending)} pending")
        return pending

# ============================================================
# SRT GENERATION
# ============================================================
def format_timestamp_srt(seconds):
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments, srt_path, logger):
    """Generate SRT subtitle file from Whisper segments."""
    logger.info(f"📝 Generating SRT: {Path(srt_path).name}")
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp_srt(seg["start"])
            end_ts = format_timestamp_srt(seg["end"])
            text = seg["text"].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n\n")
    
    logger.info(f"📝 SRT generated: {Path(srt_path).name} ({len(segments)} segments)")
    return True

# ============================================================
# HALLUCINATION DETECTION & POST-PROCESSING
# ============================================================
def filter_hallucinated_segments(segments, duration, logger):
    """
    Post-processing filter:
    1. Removes specific hallucinated phrases (YouTuber intros, subtitles credits).
    2. Removes repetitive loops.
    3. Removes abnormal duration segments.
    """
    if not segments:
        return segments
    
    original_count = len(segments)
    
    # === 0. Blacklist Filter (Các từ khóa ảo giác thường gặp của Whisper) ===
    # Whisper thường bị hallucinate ra các câu này khi gặp im lặng
    BLACKLIST_PHRASES = [
        "hãy subscribe", "kênh ghiền mì gõ", "đăng ký kênh", 
        "subtitles by", "amara.org", "vietsub bởi", 
        "chúc các bạn nghe nhạc", "bản quyền thuộc về",
        "click vào nút đăng ký", "đừng quên like"
    ]
    
    filtered_0 = []
    for seg in segments:
        text_lower = seg["text"].lower()
        # Nếu segment chứa từ cấm
        if any(bad in text_lower for bad in BLACKLIST_PHRASES):
            logger.warning(f"  [Filter] Removed Blacklist Phrase: '{seg['text'][:50]}...'")
            continue
        # Nếu segment chỉ toàn dấu chấm, phẩy hoặc ký tự lạ
        if not re.search(r'[a-zA-Zăâđêôơưàảãạáằẳẵặắầẩẫậấèẻẽẹéềểễệếìỉĩịíòỏõọóồổỗộốờởỡợớùủũụúừửữựứỳỷỹỵý]', text_lower):
             logger.debug(f"  [Filter] Removed empty/symbol segment: '{seg['text']}'")
             continue
        filtered_0.append(seg)

    # === 1. Consecutive Filter (Lọc câu lặp liên tiếp) ===
    filtered_1 = []
    if filtered_0:
        filtered_1.append(filtered_0[0])
        for i in range(1, len(filtered_0)):
            current_text = filtered_0[i]["text"].strip()
            prev_text = filtered_0[i-1]["text"].strip()
            
            # Tính tỷ lệ giống nhau (Levenshtein đơn giản hoặc check string)
            # Nếu giống nhau > 90% thì bỏ
            if current_text == prev_text:
                logger.debug(f"  [Filter] Removed exact repeat: '{current_text[:30]}...'")
                continue
                
            # Check lặp nội bộ: "A A A A A"
            # Nếu 1 từ xuất hiện quá nhiều lần trong 1 câu ngắn
            words = current_text.split()
            if len(words) > 10:
                unique_words = set(words)
                if len(unique_words) < len(words) * 0.3: # Quá ít từ vựng đa dạng -> Lặp
                     logger.debug(f"  [Filter] Removed internal loop: '{current_text[:30]}...'")
                     continue

            filtered_1.append(filtered_0[i])

    # === 2. Duration/Text Ratio (Lọc segment dài nhưng ít chữ) ===
    filtered_2 = []
    for seg in filtered_1:
        seg_duration = seg["end"] - seg["start"]
        text_len = len(seg["text"].strip())
        
        # Segment dài > 15s mà dưới 10 ký tự -> Rác
        if seg_duration > 15 and text_len < 10:
             continue
        
        # Segment cực dài (>30s) mà text quá ngắn (<30 chars) -> Nhạc nền bị nhận nhầm
        if seg_duration > 30 and text_len < 30:
            logger.debug(f"  [Filter] Removed long silence hallucination: {seg_duration}s / '{seg['text'][:20]}'")
            continue
            
        filtered_2.append(seg)
    
    total_removed = original_count - len(filtered_2)
    if total_removed > 0:
        logger.info(f"🧹 Post-processing: {original_count} → {len(filtered_2)} segments (removed {total_removed})")
    
    return filtered_2


# ============================================================
# WHISPER TRANSCRIPTION (faster-whisper + Silero VAD)
# ============================================================
def transcribe_audio(audio_path, model, logger):
    """
    Transcribe audio file using faster-whisper with Optimized VAD for Vietnamese.
    """
    logger.info(f"🧠 Transcribing: {Path(audio_path).name}")
    start = time.time()
    
    # Prompt kỹ thuật: Hướng dẫn model không lặp, dùng tiếng Việt chuẩn
    initial_prompt = "Đây là bài giảng Phật pháp, ngôn ngữ tiếng Việt rõ ràng, mạch lạc. Không lặp lại câu."

    segments_iter, info = model.transcribe(
        audio_path,
        language="vi",
        task="transcribe",
        beam_size=5,
        best_of=5,
        
        # ===== ANTI-HALLUCINATION & SETTINGS =====
        condition_on_previous_text=False,      # TUYỆT ĐỐI FALSE để tránh vòng lặp vô tận
        temperature=[0.0, 0.2],                # Chỉ cho phép nhiệt độ thấp, nếu không chắc chắn thì bỏ qua luôn (tránh bịa ra text ở nhiệt độ cao)
        compression_ratio_threshold=2.0,       # Chặt hơn (gốc 2.4). Nếu nén text lại mà ratio cao nghĩa là text bị lặp -> Bỏ.
        log_prob_threshold=-1.0,               # Tăng độ tự tin cần thiết (gốc -1.0, có thể giữ nguyên hoặc giảm chút)
        no_speech_threshold=0.6,               # Tăng lên 0.6: Phải xác suất là tiếng người > 60% mới lấy.
        repetition_penalty=1.2,                # Phạt nặng việc lặp từ (gốc 1.0)
        
        initial_prompt=initial_prompt,
        
        # ===== SILERO VAD - QUAN TRỌNG NHẤT =====
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.6,                     # Tăng lên 0.6: Chỉ lấy giọng nói thật rõ, bỏ nhạc nền/tiếng ồn.
            min_speech_duration_ms=250,
            max_speech_duration_s=20.0,        # QUAN TRỌNG: Cắt cứng mỗi 20s. Không để vô tận (inf) gây lỗi lặp 60s.
            min_silence_duration_ms=1000,      # Giảm xuống để VAD cắt segment nhanh hơn khi ngắt nghỉ.
            speech_pad_ms=400,
        ),
    )
    
    segments_list = []
    # Lưu ý: faster-whisper là generator, lỗi sẽ bắn ra khi loop
    try:
        for seg in segments_iter:
            # Quick check: Nếu segment sinh ra text giống hệt segment ngay trước đó -> Bỏ qua luôn tại nguồn
            if segments_list and seg.text.strip() == segments_list[-1]["text"].strip():
                continue
                
            segments_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })
    except Exception as e:
        logger.error(f"Error during transcription loop: {e}")
        # Vẫn trả về những gì đã làm được
        pass

    elapsed_transcribe = time.time() - start
    logger.info(f"🧠 Transcription done raw: {len(segments_list)} segments")
    logger.info(f"🧠 Detected info: duration={info.duration:.1f}s")
    
    # ===== POST-PROCESSING =====
    segments_list = filter_hallucinated_segments(segments_list, info.duration, logger)
    
    elapsed = time.time() - start
    
    full_text = " ".join(seg["text"].strip() for seg in segments_list)
    result = {
        "text": full_text,
        "language": info.language,
        "duration": info.duration,
        "segments": segments_list,
    }
    
    return result, elapsed


def normalize_audio_name(filename):
    """
    Normalize audio filename to clean title case, preserving Vietnamese diacritics.
    
    Rules:
    - Trim whitespace
    - Replace underscores with spaces (except between digits: 20_05_2024)
    - Collapse multiple spaces
    - Normalize hyphens as separators: " - "
    - Preserve parentheses, commas, dots
    - Apply title case
    
    Examples:
        'ai la chu nhan.mp3'                        → 'Ai Là Chủ Nhân.mp3'
        'kinh nikaya 15 - kinh sa mon qua.mp3'      → 'Kinh Nikaya 15 - Kinh Sa Môn Quả.mp3'
    """
    stem = Path(filename).stem
    ext = Path(filename).suffix.lower()
    
    # Replace underscores with spaces, EXCEPT between digits (dates: 20_05_2024)
    name = re.sub(r'(?<!\d)_|_(?!\d)', ' ', stem)
    
    # Collapse multiple spaces and trim
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Normalize spaces around hyphens: "abc  -  def" → "abc - def"
    name = re.sub(r'\s*-\s*', ' - ', name)
    
    # Apply title case
    name = name.title()
    
    return f"{name}{ext}"


def format_result_json(audio_name, drive_path, whisper_result, processing_time):
    """Format Whisper result into structured JSON for database import."""
    segments = []
    
    for i, seg in enumerate(whisper_result.get("segments", [])):
        segments.append({
            "segment_id": i,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })
    
    duration = whisper_result.get("duration", segments[-1]["end"] if segments else 0)
    
    # Derive video name from audio name (replace .mp3 with original reference)
    video_name = normalize_audio_name(audio_name)
    
    return {
        "video_name": video_name,
        "video_name_original": audio_name,
        "drive_path": drive_path,
        "full_text": whisper_result.get("text", "").strip(),
        "duration": round(duration, 2),
        "language": whisper_result.get("language", "vi"),
        "transcribed_at": datetime.now(VN_TZ).isoformat(),
        "model_used": DEFAULT_WHISPER_MODEL,
        "total_segments": len(segments),
        "processing_time_seconds": round(processing_time, 1),
        "segments": segments
    }

# ============================================================
# PREFETCH (Download MP3 in Background)
# ============================================================
class Prefetcher:
    """
    Background thread that downloads the NEXT MP3 file
    while the main thread is busy transcribing the CURRENT file.
    
    Flow:
        Main Thread:   Transcribe A → Upload A → Transcribe B → Upload B → ...
        Prefetcher:    Download B → Download C → ...
    """
    
    def __init__(self, drive_input_path, logger):
        self.drive_input_path = drive_input_path
        self.logger = logger
        self.queue = Queue(maxsize=1)  # Only prefetch 1 audio ahead
        self._thread = None
        self._stop = threading.Event()
    
    def prefetch(self, audio_info):
        """
        Download MP3 from Google Drive in background.
        Puts (audio_path, audio_info) into queue when ready.
        Puts (None, audio_info) if failed.
        """
        audio_name = audio_info["Path"]
        remote_path = f"{self.drive_input_path}/{audio_name}"
        local_audio = os.path.join(LOCAL_WORK_DIR, "audio", audio_name)
        
        try:
            # Download MP3
            self.logger.info(f"[Prefetch] Downloading: {audio_name}")
            ok = rclone_download(remote_path, local_audio, self.logger)
            if not ok:
                self.queue.put((None, audio_info))
                return
            
            self.queue.put((local_audio, audio_info))
            
        except Exception as e:
            self.logger.error(f"[Prefetch] Error: {e}")
            self.queue.put((None, audio_info))
    
    def start_prefetch(self, audio_info):
        """Start prefetching an audio file in background thread."""
        self._thread = threading.Thread(
            target=self.prefetch,
            args=(audio_info,),
            name="Prefetcher",
            daemon=True
        )
        self._thread.start()
    
    def get_result(self, timeout=None):
        """Wait for prefetch result."""
        return self.queue.get(timeout=timeout)
    
    def wait(self):
        """Wait for current prefetch to complete."""
        if self._thread and self._thread.is_alive():
            self._thread.join()

# ============================================================
# MAIN PIPELINE
# ============================================================
def process_audio(audio_path, audio_info, model, drive_output_path, checkpoint, logger):
    """
    Process a single audio file (already downloaded from Drive):
    1. Transcribe audio → JSON + SRT
    2. Upload JSON, SRT to Drive
    3. Update checkpoint
    4. Cleanup local files (audio + output)
    """
    audio_rel_path = audio_info["Path"]
    audio_name = Path(audio_rel_path).stem
    
    # Local output paths
    local_json = os.path.join(LOCAL_WORK_DIR, "output", f"{audio_name}.json")
    local_srt = os.path.join(LOCAL_WORK_DIR, "output", f"{audio_name}.srt")
    os.makedirs(os.path.dirname(local_json), exist_ok=True)
    
    # Remote output paths
    remote_json = f"{drive_output_path}/JSON/{audio_name}.json"
    remote_srt = f"{drive_output_path}/SRT/{audio_name}.srt"
    
    total_start = time.time()
    
    try:
        # ===== STEP 1: Transcribe =====
        logger.info(f"{'='*60}")
        logger.info(f"🎬 Processing: {audio_rel_path}")
        logger.info(f"{'='*60}")
        
        checkpoint.mark_processing(audio_rel_path)
        
        whisper_result, transcribe_time = transcribe_audio(audio_path, model, logger)
        
        # ===== STEP 2: Generate JSON =====
        formatted = format_result_json(
            audio_name=Path(audio_rel_path).name,
            drive_path=audio_rel_path,
            whisper_result=whisper_result,
            processing_time=transcribe_time
        )
        
        with open(local_json, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 JSON saved locally: {local_json}")
        
        # ===== STEP 3: Generate SRT =====
        generate_srt(formatted["segments"], local_srt, logger)
        
        # ===== STEP 4: Upload to Drive =====
        logger.info(f"⬆️  Uploading outputs to Drive...")
        
        upload_ok = True
        if not rclone_upload(local_json, remote_json, logger):
            upload_ok = False
        if not rclone_upload(local_srt, remote_srt, logger):
            upload_ok = False
        
        if not upload_ok:
            logger.warning("⚠️  Some uploads failed, but continuing...")
        
        # ===== STEP 5: Update Checkpoint =====
        total_time = time.time() - total_start
        checkpoint.mark_done(
            audio_path=audio_rel_path,
            processing_time=total_time,
            output_json=remote_json,
            output_srt=remote_srt
        )
        
        # ===== STEP 6: Cleanup ALL local files =====
        for f in [local_json, local_srt, audio_path]:
            if os.path.exists(f):
                os.remove(f)
                logger.debug(f"🗑️  Deleted: {f}")
        
        logger.info(f"✅ DONE: {audio_rel_path} | "
                     f"Transcribe: {transcribe_time:.1f}s | "
                     f"Total: {total_time:.1f}s | "
                     f"Segments: {formatted['total_segments']}")
        
        return True
        
    except Exception as e:
        total_time = time.time() - total_start
        logger.error(f"❌ FAILED: {audio_rel_path} ({total_time:.1f}s): {e}", exc_info=True)
        checkpoint.mark_failed(audio_rel_path, str(e))
        
        # Cleanup on failure
        for f in [local_json, local_srt, audio_path]:
            if os.path.exists(f):
                os.remove(f)
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Audio Transcription Pipeline (Spot Instance Safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all MP3 files in Drive folder
  python3 transcribe.py --drive-input "Output/Audio" --drive-output "Output"
  
  # Test with first audio file only
  python3 transcribe.py --drive-input "Output/Audio" --drive-output "Output" --test
  
  # Use medium model (faster, less accurate)
  python3 transcribe.py --drive-input "Output/Audio" --drive-output "Output" --model medium
  
  # Process only 10 files
  python3 transcribe.py --drive-input "Output/Audio" --drive-output "Output" --batch-size 10
        """
    )
    parser.add_argument("--drive-input", type=str, required=True,
                        help="Google Drive folder path containing MP3 files (e.g. 'Output/Audio')")
    parser.add_argument("--drive-output", type=str, required=True,
                        help="Google Drive folder path for output (e.g. 'Output')")
    parser.add_argument("--model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help=f"Whisper model (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only first audio file")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Process N files then stop (0 = all)")
    
    args = parser.parse_args()
    
    # ===== SETUP =====
    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("🚀 AUDIO TRANSCRIPTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Drive input:  {RCLONE_REMOTE}:{args.drive_input}")
    logger.info(f"Drive output: {RCLONE_REMOTE}:{args.drive_output}")
    logger.info(f"Whisper model: {args.model}")
    logger.info(f"Work directory: {LOCAL_WORK_DIR}")
    logger.info(f"Test mode: {args.test}")
    logger.info(f"Batch size: {args.batch_size or 'ALL'}")
    
    # ===== STEP 1: Load Checkpoint =====
    checkpoint = CheckpointManager(args.drive_output, logger)
    checkpoint.load()
    
    # ===== STEP 2: List Audio Files =====
    all_audios = rclone_list_files(args.drive_input, logger)
    if not all_audios:
        logger.error("No audio files found. Check --drive-input path.")
        sys.exit(1)
    
    # ===== STEP 3: Filter Pending Audio Files =====
    pending_audios = checkpoint.get_pending_audios(all_audios)
    if not pending_audios:
        logger.info("🎉 All audio files already processed! Nothing to do.")
        sys.exit(0)
    
    # Apply batch size / test mode
    if args.test:
        pending_audios = pending_audios[:1]
        logger.info("🧪 TEST MODE: Processing only 1 audio file")
    elif args.batch_size > 0:
        pending_audios = pending_audios[:args.batch_size]
        logger.info(f"📦 BATCH MODE: Processing {len(pending_audios)} audio files")
    
    # ===== STEP 4: Load Whisper Model (faster-whisper) =====
    logger.info(f"🧠 Loading faster-whisper model: {args.model}")
    from faster_whisper import WhisperModel
    
    # Use float16 on GPU for speed, int8 on CPU for memory efficiency
    model = WhisperModel(
        args.model,
        device="cuda",          # Use GPU (change to "cpu" if no GPU)
        compute_type="float16",  # float16 for GPU, int8 for CPU
    )
    logger.info(f"🧠 faster-whisper model loaded successfully! (device=cuda, compute_type=float16)")
    
    # ===== STEP 5: Pipeline Processing =====
    prefetcher = Prefetcher(args.drive_input, logger)
    total = len(pending_audios)
    success_count = 0
    fail_count = 0
    pipeline_start = time.time()
    
    # Start prefetching first audio
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 Starting pipeline: {total} audio files to process")
    logger.info(f"{'='*60}\n")
    
    # Prefetch first audio
    prefetcher.start_prefetch(pending_audios[0])
    
    for idx, audio_info in enumerate(pending_audios):
        # Start prefetching NEXT audio (if available)
        if idx + 1 < total:
            next_audio = pending_audios[idx + 1]
        else:
            next_audio = None
        
        # Wait for current audio's prefetch to complete (max 10 min for MP3)
        logger.info(f"\n[{idx+1}/{total}] Waiting for MP3 download...")
        try:
            audio_path, fetched_info = prefetcher.get_result(timeout=600)
        except Exception:
            logger.error(f"[{idx+1}/{total}] TIMEOUT waiting for download: {audio_info['Path']}")
            checkpoint.mark_failed(audio_info["Path"], "Download timeout (>10min)")
            fail_count += 1
            # Kill stuck prefetcher and start fresh for next audio
            if next_audio:
                prefetcher = Prefetcher(args.drive_input, logger)
                prefetcher.start_prefetch(next_audio)
            continue
        
        # Start prefetching next audio immediately (pipeline!)
        if next_audio:
            prefetcher.start_prefetch(next_audio)
            logger.debug(f"[Pipeline] Prefetching next: {next_audio['Path']}")
        
        # Process current audio
        if audio_path is None:
            logger.error(f"[{idx+1}/{total}] Skipping (download failed): {audio_info['Path']}")
            checkpoint.mark_failed(audio_info["Path"], "Download failed")
            fail_count += 1
            continue
        
        success = process_audio(
            audio_path=audio_path,
            audio_info=audio_info,
            model=model,
            drive_output_path=args.drive_output,
            checkpoint=checkpoint,
            logger=logger
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # Progress report with ETA
        elapsed = time.time() - pipeline_start
        avg_per_audio = elapsed / (idx + 1)
        remaining = (total - idx - 1) * avg_per_audio
        remaining_str = str(timedelta(seconds=int(remaining)))
        
        logger.info(f"📊 Progress: [{idx+1}/{total}] | ✅ {success_count} | ❌ {fail_count} | "
                     f"Avg: {avg_per_audio:.0f}s/file | ETA: {remaining_str}")
    
    # ===== FINAL SUMMARY =====
    total_elapsed = time.time() - pipeline_start
    total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🏁 PIPELINE COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info(f"Total files:  {total}")
    logger.info(f"Success:      {success_count}")
    logger.info(f"Failed:       {fail_count}")
    logger.info(f"Total time:   {total_elapsed_str}")
    logger.info(f"Output:       {RCLONE_REMOTE}:{args.drive_output}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

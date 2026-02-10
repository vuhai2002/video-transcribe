#!/usr/bin/env python3
"""
Video Transcription Pipeline (Spot Instance Safe)
===================================================
Tự động tải video từ Google Drive, tách audio, transcribe bằng Whisper,
upload kết quả (JSON, SRT, MP3) lên Drive, và dọn dẹp.

Tính năng:
- Pipeline gối đầu: Tải Video B trong khi Transcribe Video A
- Checkpoint trên Google Drive: An toàn khi Spot Instance bị thu hồi
- Auto-resume: Tự động tiếp tục từ video cuối cùng khi VM khởi động lại
- Log chi tiết: Ghi lại mọi bước xử lý

Usage:
    python3 transcribe.py --drive-input "Folder/Video" --drive-output "Output"
    python3 transcribe.py --drive-input "Folder/Video" --drive-output "Output" --test
    python3 transcribe.py --drive-input "Folder/Video" --drive-output "Output" --model medium
"""

import os
import sys
import json
import time
import argparse
import subprocess
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from queue import Queue

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_WHISPER_MODEL = "large-v3"
RCLONE_REMOTE = "gdrive"
LOCAL_WORK_DIR = os.path.expanduser("~/transcribe-work")
CHECKPOINT_FILENAME = "checkpoint.json"

# Supported video formats
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}

# ============================================================
# LOGGING SETUP
# ============================================================
def setup_logging():
    """Setup logging to both file and console with detailed format."""
    log_dir = os.path.join(LOCAL_WORK_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"transcribe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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
    List all video files in a remote path using rclone lsjson.
    Returns list of dicts: [{"Path": "sub/video.mp4", "Size": 123456, "Name": "video.mp4"}, ...]
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
    
    # Filter video files only
    video_files = [
        f for f in all_files
        if Path(f["Path"]).suffix.lower() in VIDEO_EXTENSIONS
    ]
    
    # Sort A-Z by path
    video_files.sort(key=lambda f: f["Path"])
    
    logger.info(f"Found {len(video_files)} video files (sorted A-Z)")
    for i, f in enumerate(video_files[:5]):
        size_mb = f.get("Size", 0) / (1024 * 1024)
        logger.debug(f"  [{i+1}] {f['Path']} ({size_mb:.1f} MB)")
    if len(video_files) > 5:
        logger.debug(f"  ... and {len(video_files) - 5} more")
    
    return video_files


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
        "videos": {
            "path/to/video.mp4": {
                "status": "done" | "processing",
                "started_at": "2026-02-10T10:00:00",
                "completed_at": "2026-02-10T10:15:00",
                "processing_time_seconds": 900.0,
                "output_json": "Output/JSON/video.json",
                "output_srt": "Output/SRT/video.srt",
                "output_mp3": "Output/Audio/video.mp3"
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
        self.data = {"videos": {}, "stats": {"total_processed": 0, "total_failed": 0}}
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
                self.logger.info(f"Checkpoint loaded: {len(self.data.get('videos', {}))} videos tracked")
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Checkpoint corrupt, starting fresh: {e}")
                self.data = {"videos": {}, "stats": {"total_processed": 0, "total_failed": 0}}
        else:
            self.logger.info("No existing checkpoint found, starting fresh")
    
    def save(self):
        """Save checkpoint to both local and Google Drive."""
        with self._lock:
            self.data["stats"]["last_updated"] = datetime.now().isoformat()
            
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
    
    def mark_processing(self, video_path):
        """Mark a video as currently being processed."""
        with self._lock:
            self.data["videos"][video_path] = {
                "status": "processing",
                "started_at": datetime.now().isoformat(),
                "completed_at": None
            }
        self.save()
        self.logger.info(f"📝 Checkpoint: PROCESSING → {video_path}")
    
    def mark_done(self, video_path, processing_time, output_json, output_srt, output_mp3):
        """Mark a video as successfully processed."""
        with self._lock:
            self.data["videos"][video_path] = {
                "status": "done",
                "started_at": self.data["videos"].get(video_path, {}).get("started_at"),
                "completed_at": datetime.now().isoformat(),
                "processing_time_seconds": round(processing_time, 1),
                "output_json": output_json,
                "output_srt": output_srt,
                "output_mp3": output_mp3
            }
            self.data["stats"]["total_processed"] = sum(
                1 for v in self.data["videos"].values() if v["status"] == "done"
            )
        self.save()
        self.logger.info(f"✅ Checkpoint: DONE → {video_path} ({processing_time:.1f}s)")
    
    def mark_failed(self, video_path, error_message):
        """Mark a video as failed."""
        with self._lock:
            self.data["videos"][video_path] = {
                "status": "failed",
                "started_at": self.data["videos"].get(video_path, {}).get("started_at"),
                "completed_at": datetime.now().isoformat(),
                "error": str(error_message)
            }
            self.data["stats"]["total_failed"] = sum(
                1 for v in self.data["videos"].values() if v["status"] == "failed"
            )
        self.save()
        self.logger.error(f"❌ Checkpoint: FAILED → {video_path}: {error_message}")
    
    def get_status(self, video_path):
        """Get status of a video: 'done', 'processing', 'failed', or None."""
        return self.data.get("videos", {}).get(video_path, {}).get("status")
    
    def get_pending_videos(self, all_videos):
        """
        Filter videos that need processing.
        - Skip 'done' videos
        - Re-process 'processing' videos (crashed mid-way)
        - Re-process 'failed' videos (might succeed this time)
        """
        pending = []
        skipped = 0
        retry = 0
        
        for video in all_videos:
            path = video["Path"]
            status = self.get_status(path)
            
            if status == "done":
                skipped += 1
            elif status == "processing":
                self.logger.warning(f"🔄 Re-processing (crashed mid-way): {path}")
                retry += 1
                pending.append(video)
            elif status == "failed":
                self.logger.warning(f"🔄 Retrying (previously failed): {path}")
                retry += 1
                pending.append(video)
            else:
                pending.append(video)
        
        self.logger.info(f"Videos: {len(all_videos)} total | {skipped} done | {retry} retry | {len(pending)} pending")
        return pending

# ============================================================
# AUDIO EXTRACTION (MP4 → MP3)
# ============================================================
def extract_audio(video_path, audio_path, logger):
    """Extract audio from video file to MP3 format."""
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    
    logger.info(f"🎵 Extracting audio: {Path(video_path).name} → {Path(audio_path).name}")
    start = time.time()
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                    # No video
        "-acodec", "libmp3lame",  # MP3 codec
        "-ab", "128k",            # 128kbps (good for speech)
        "-ar", "16000",           # 16kHz (Whisper optimal)
        "-ac", "1",               # Mono
        "-y",                     # Overwrite
        audio_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"🎵 Audio extracted: {Path(audio_path).name} ({size_mb:.1f} MB in {elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        logger.error(f"🎵 FFmpeg FAILED for {Path(video_path).name} ({elapsed:.1f}s): {e.stderr[:500]}")
        return False

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
# WHISPER TRANSCRIPTION
# ============================================================
def transcribe_audio(audio_path, model, logger):
    """Transcribe audio file using Whisper. Returns structured result."""
    logger.info(f"🧠 Transcribing: {Path(audio_path).name}")
    start = time.time()
    
    result = model.transcribe(
        audio_path,
        language="vi",
        task="transcribe",
        verbose=False,
        word_timestamps=False,
        fp16=True,
    )
    
    elapsed = time.time() - start
    num_segments = len(result.get("segments", []))
    logger.info(f"🧠 Transcription done: {num_segments} segments in {elapsed:.1f}s")
    
    return result, elapsed


import re

def normalize_video_name(filename):
    """
    Normalize video filename to clean title case, preserving Vietnamese diacritics.
    
    Rules:
    - Trim whitespace
    - Replace underscores with spaces (except between digits: 20_05_2024)
    - Collapse multiple spaces
    - Normalize hyphens as separators: " - "
    - Preserve parentheses, commas, dots
    - Apply title case
    
    Examples:
        'ai là chủ nhân.mp4'                       → 'Ai Là Chủ Nhân.mp4'
        'kinh nikaya 15 - kinh sa môn quả (trường bộ).mp4'
                                                    → 'Kinh Nikaya 15 - Kinh Sa Môn Quả (Trường Bộ).mp4'
        'su phu noi chuyen - dai le phat dan - 20_05_2024.mp4'
                                                    → 'Sư Phụ Nói Chuyện - Đại Lễ Phật Đản - 20_05_2024.mp4'
        'luan ve nhan qua 06 - giong doc huong duong.mp4'
                                                    → 'Luận Về Nhân Quả 06 - Giọng Đọc Hướng Dương.mp4'
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


def format_result_json(video_name, drive_path, whisper_result, processing_time):
    """Format Whisper result into structured JSON for database import."""
    segments = []
    
    for i, seg in enumerate(whisper_result.get("segments", [])):
        segments.append({
            "segment_id": i,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })
    
    duration = segments[-1]["end"] if segments else 0
    
    return {
        "video_name": normalize_video_name(video_name),
        "video_name_original": video_name,
        "drive_path": drive_path,
        "full_text": whisper_result.get("text", "").strip(),
        "duration": round(duration, 2),
        "language": whisper_result.get("language", "vi"),
        "transcribed_at": datetime.now().isoformat(),
        "model_used": DEFAULT_WHISPER_MODEL,
        "total_segments": len(segments),
        "processing_time_seconds": round(processing_time, 1),
        "segments": segments
    }

# ============================================================
# PREFETCH (Download + Extract Audio in Background)
# ============================================================
class Prefetcher:
    """
    Background thread that downloads the NEXT video and extracts audio
    while the main thread is busy transcribing the CURRENT video.
    
    Flow:
        Main Thread:   Transcribe A → Upload A → Transcribe B → Upload B → ...
        Prefetcher:    Download B + Extract B → Download C + Extract C → ...
    """
    
    def __init__(self, drive_input_path, logger):
        self.drive_input_path = drive_input_path
        self.logger = logger
        self.queue = Queue(maxsize=1)  # Only prefetch 1 video ahead
        self._thread = None
        self._stop = threading.Event()
    
    def prefetch(self, video_info):
        """
        Download video and extract audio in background.
        Puts (video_path, audio_path, video_info) into queue when ready.
        Puts (None, None, video_info) if failed.
        """
        video_name = Path(video_info["Path"]).stem
        remote_path = f"{self.drive_input_path}/{video_info['Path']}"
        local_video = os.path.join(LOCAL_WORK_DIR, "download", video_info["Path"])
        local_audio = os.path.join(LOCAL_WORK_DIR, "audio", f"{video_name}.mp3")
        
        try:
            # Step 1: Download video
            self.logger.info(f"[Prefetch] Downloading: {video_info['Path']}")
            ok = rclone_download(remote_path, local_video, self.logger)
            if not ok:
                self.queue.put((None, None, video_info))
                return
            
            # Step 2: Extract audio
            self.logger.info(f"[Prefetch] Extracting audio: {video_info['Path']}")
            ok = extract_audio(local_video, local_audio, self.logger)
            if not ok:
                # Cleanup failed video
                if os.path.exists(local_video):
                    os.remove(local_video)
                self.queue.put((None, None, video_info))
                return
            
            # Step 3: Delete video file (we only need audio for transcription)
            if os.path.exists(local_video):
                os.remove(local_video)
                self.logger.debug(f"[Prefetch] Deleted video (keeping audio): {Path(local_video).name}")
            
            self.queue.put((local_video, local_audio, video_info))
            
        except Exception as e:
            self.logger.error(f"[Prefetch] Error: {e}")
            self.queue.put((None, None, video_info))
    
    def start_prefetch(self, video_info):
        """Start prefetching a video in background thread."""
        self._thread = threading.Thread(
            target=self.prefetch,
            args=(video_info,),
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
def process_video(audio_path, video_info, model, drive_output_path, checkpoint, logger):
    """
    Process a single video (audio already extracted by prefetcher):
    1. Transcribe audio → JSON + SRT
    2. Upload JSON, SRT, MP3 to Drive
    3. Update checkpoint
    4. Cleanup local files
    """
    video_path_on_drive = video_info["Path"]
    video_name = Path(video_path_on_drive).stem
    
    # Local output paths
    local_json = os.path.join(LOCAL_WORK_DIR, "output", f"{video_name}.json")
    local_srt = os.path.join(LOCAL_WORK_DIR, "output", f"{video_name}.srt")
    os.makedirs(os.path.dirname(local_json), exist_ok=True)
    
    # Remote output paths
    remote_json = f"{drive_output_path}/JSON/{video_name}.json"
    remote_srt = f"{drive_output_path}/SRT/{video_name}.srt"
    remote_mp3 = f"{drive_output_path}/Audio/{video_name}.mp3"
    
    total_start = time.time()
    
    try:
        # ===== STEP 1: Transcribe =====
        logger.info(f"{'='*60}")
        logger.info(f"🎬 Processing: {video_path_on_drive}")
        logger.info(f"{'='*60}")
        
        checkpoint.mark_processing(video_path_on_drive)
        
        whisper_result, transcribe_time = transcribe_audio(audio_path, model, logger)
        
        # ===== STEP 2: Generate JSON =====
        formatted = format_result_json(
            video_name=Path(video_path_on_drive).name,
            drive_path=video_path_on_drive,
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
        if not rclone_upload(audio_path, remote_mp3, logger):
            upload_ok = False
        
        if not upload_ok:
            logger.warning("⚠️  Some uploads failed, but continuing...")
        
        # ===== STEP 5: Update Checkpoint =====
        total_time = time.time() - total_start
        checkpoint.mark_done(
            video_path=video_path_on_drive,
            processing_time=total_time,
            output_json=remote_json,
            output_srt=remote_srt,
            output_mp3=remote_mp3
        )
        
        # ===== STEP 6: Cleanup =====
        for f in [local_json, local_srt, audio_path]:
            if os.path.exists(f):
                os.remove(f)
                logger.debug(f"🗑️  Deleted: {f}")
        
        logger.info(f"✅ DONE: {video_path_on_drive} | "
                     f"Transcribe: {transcribe_time:.1f}s | "
                     f"Total: {total_time:.1f}s | "
                     f"Segments: {formatted['total_segments']}")
        
        return True
        
    except Exception as e:
        total_time = time.time() - total_start
        logger.error(f"❌ FAILED: {video_path_on_drive} ({total_time:.1f}s): {e}", exc_info=True)
        checkpoint.mark_failed(video_path_on_drive, str(e))
        
        # Cleanup on failure
        for f in [local_json, local_srt, audio_path]:
            if os.path.exists(f):
                os.remove(f)
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Video Transcription Pipeline (Spot Instance Safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in Drive folder
  python3 transcribe.py --drive-input "BaiGiang" --drive-output "Output"
  
  # Test with first video only
  python3 transcribe.py --drive-input "BaiGiang" --drive-output "Output" --test
  
  # Use medium model (faster, less accurate)
  python3 transcribe.py --drive-input "BaiGiang" --drive-output "Output" --model medium
  
  # Process only 10 videos
  python3 transcribe.py --drive-input "BaiGiang" --drive-output "Output" --batch-size 10
        """
    )
    parser.add_argument("--drive-input", type=str, required=True,
                        help="Google Drive folder path containing videos (e.g. 'BaiGiang/LichSu')")
    parser.add_argument("--drive-output", type=str, required=True,
                        help="Google Drive folder path for output (e.g. 'Output')")
    parser.add_argument("--model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help=f"Whisper model (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only first video")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Process N videos then stop (0 = all)")
    
    args = parser.parse_args()
    
    # ===== SETUP =====
    os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("🚀 VIDEO TRANSCRIPTION PIPELINE")
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
    
    # ===== STEP 2: List Videos =====
    all_videos = rclone_list_files(args.drive_input, logger)
    if not all_videos:
        logger.error("No video files found. Check --drive-input path.")
        sys.exit(1)
    
    # ===== STEP 3: Filter Pending Videos =====
    pending_videos = checkpoint.get_pending_videos(all_videos)
    if not pending_videos:
        logger.info("🎉 All videos already processed! Nothing to do.")
        sys.exit(0)
    
    # Apply batch size / test mode
    if args.test:
        pending_videos = pending_videos[:1]
        logger.info("🧪 TEST MODE: Processing only 1 video")
    elif args.batch_size > 0:
        pending_videos = pending_videos[:args.batch_size]
        logger.info(f"📦 BATCH MODE: Processing {len(pending_videos)} videos")
    
    # ===== STEP 4: Load Whisper Model =====
    logger.info(f"🧠 Loading Whisper model: {args.model}")
    import whisper
    model = whisper.load_model(args.model)
    logger.info(f"🧠 Model loaded successfully!")
    
    # ===== STEP 5: Pipeline Processing =====
    prefetcher = Prefetcher(args.drive_input, logger)
    total = len(pending_videos)
    success_count = 0
    fail_count = 0
    
    # Start prefetching first video
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 Starting pipeline: {total} videos to process")
    logger.info(f"{'='*60}\n")
    
    # Prefetch first video
    prefetcher.start_prefetch(pending_videos[0])
    
    for idx, video_info in enumerate(pending_videos):
        # Start prefetching NEXT video (if available)
        if idx + 1 < total:
            next_video = pending_videos[idx + 1]
        else:
            next_video = None
        
        # Wait for current video's prefetch to complete
        logger.info(f"\n[{idx+1}/{total}] Waiting for download + audio extraction...")
        _, audio_path, fetched_info = prefetcher.get_result()
        
        # Start prefetching next video immediately (pipeline!)
        if next_video:
            prefetcher.start_prefetch(next_video)
            logger.debug(f"[Pipeline] Prefetching next: {next_video['Path']}")
        
        # Process current video
        if audio_path is None:
            logger.error(f"[{idx+1}/{total}] Skipping (download/extract failed): {video_info['Path']}")
            checkpoint.mark_failed(video_info["Path"], "Download or audio extraction failed")
            fail_count += 1
            continue
        
        success = process_video(
            audio_path=audio_path,
            video_info=video_info,
            model=model,
            drive_output_path=args.drive_output,
            checkpoint=checkpoint,
            logger=logger
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # Progress report
        elapsed_videos = idx + 1
        eta_per_video = (time.time() - time.time()) if idx == 0 else None  # Placeholder
        logger.info(f"📊 Progress: [{elapsed_videos}/{total}] | ✅ {success_count} | ❌ {fail_count}")
    
    # ===== FINAL SUMMARY =====
    logger.info(f"\n{'='*60}")
    logger.info(f"🏁 PIPELINE COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {total}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Output: {RCLONE_REMOTE}:{args.drive_output}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

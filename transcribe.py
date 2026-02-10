#!/usr/bin/env python3
"""
Video Transcription Pipeline
=============================
Downloads videos from Google Drive, extracts audio (MP3),
runs Whisper transcription, and outputs structured JSON
with timestamps for database import.

Usage:
    python3 transcribe.py --drive-folder <FOLDER_ID> [--model large-v3] [--output-dir ./output]
    python3 transcribe.py --local-dir /path/to/videos [--model large-v3] [--output-dir ./output]
    python3 transcribe.py --resume  # Resume from last checkpoint
"""

import os
import sys
import json
import time
import argparse
import subprocess
import hashlib
import logging
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_WHISPER_MODEL = "large-v3"  # Best accuracy for Vietnamese
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_AUDIO_DIR = "./audio"
DEFAULT_VIDEO_DIR = "./videos"
CHECKPOINT_FILE = "./checkpoint.json"
LOG_FILE = "./transcribe.log"

# Supported video formats
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}

# ============================================================
# LOGGING SETUP
# ============================================================
def setup_logging():
    """Setup logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================
# CHECKPOINT MANAGEMENT (Resume support)
# ============================================================
def load_checkpoint():
    """Load checkpoint to resume from last processed video."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed": [], "failed": [], "last_updated": None}

def save_checkpoint(checkpoint):
    """Save checkpoint after processing each video."""
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

# ============================================================
# GOOGLE DRIVE INTEGRATION
# ============================================================
def install_gdown():
    """Install gdown for Google Drive downloads."""
    try:
        import gdown
    except ImportError:
        logger.info("Installing gdown for Google Drive access...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown
    return gdown

def list_drive_folder(folder_id):
    """List all video files in a Google Drive folder."""
    gdown = install_gdown()
    import gdown.download_folder as df
    
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info(f"Listing files in Google Drive folder: {folder_id}")
    
    # Use gdown to list files
    try:
        files = gdown.download_folder(
            url=url,
            quiet=True,
            use_cookies=False,
            output=DEFAULT_VIDEO_DIR,
            remaining_ok=True
        )
        return files
    except Exception as e:
        logger.error(f"Error listing Drive folder: {e}")
        logger.info("Alternative: Use rclone for better Google Drive integration")
        return []

def download_from_drive_rclone(folder_id, output_dir):
    """
    Download files from Google Drive using rclone (recommended).
    Requires rclone to be configured with a remote named 'gdrive'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "rclone", "copy",
        f"gdrive:{{path_in_drive}}",  # Replace with actual path
        output_dir,
        "--include", "*.mp4",
        "--progress",
        "--transfers", "3",
        "--checkers", "3",
        "-v"
    ]
    
    logger.info(f"Downloading from Google Drive to {output_dir}")
    subprocess.run(cmd, check=True)

# ============================================================
# AUDIO EXTRACTION (MP4 → MP3)
# ============================================================
def extract_audio(video_path, audio_dir):
    """
    Extract audio from video file to MP3 format.
    Returns path to the audio file.
    """
    os.makedirs(audio_dir, exist_ok=True)
    
    video_name = Path(video_path).stem
    audio_path = os.path.join(audio_dir, f"{video_name}.mp3")
    
    # Skip if already extracted
    if os.path.exists(audio_path):
        logger.info(f"Audio already exists: {audio_path}")
        return audio_path
    
    logger.info(f"Extracting audio: {video_path} → {audio_path}")
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                    # No video
        "-acodec", "libmp3lame",  # MP3 codec
        "-ab", "128k",            # 128kbps bitrate (good enough for speech)
        "-ar", "16000",           # 16kHz sample rate (Whisper optimal)
        "-ac", "1",               # Mono (speech doesn't need stereo)
        "-y",                     # Overwrite if exists
        audio_path
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        logger.info(f"Audio extracted successfully: {audio_path}")
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error for {video_path}: {e.stderr}")
        return None

# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================
def transcribe_audio(audio_path, model_name=DEFAULT_WHISPER_MODEL):
    """
    Transcribe audio file using Whisper.
    Returns structured result with segments and timestamps.
    """
    import whisper
    
    logger.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    logger.info(f"Transcribing: {audio_path}")
    start_time = time.time()
    
    result = model.transcribe(
        audio_path,
        language="vi",           # Vietnamese
        task="transcribe",       # Transcription (not translation)
        verbose=False,
        word_timestamps=False,   # Not needed - segments are enough
        fp16=True,               # Use FP16 for GPU acceleration
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Transcription completed in {elapsed:.1f}s")
    
    return result

def format_result_for_db(video_name, video_path, whisper_result):
    """
    Format Whisper result into structured JSON for database import.
    
    Output format:
    {
        "video_name": "bai_giang_01.mp4",
        "video_path": "/path/to/video",
        "full_text": "Toàn bộ nội dung...",
        "duration": 3600.5,
        "language": "vi",
        "transcribed_at": "2026-02-10T04:00:00",
        "segments": [
            {
                "segment_id": 0,
                "start": 0.0,
                "end": 5.2,
                "text": "Xin chào các em...",
                "words": [
                    {"word": "Xin", "start": 0.0, "end": 0.3},
                    {"word": "chào", "start": 0.3, "end": 0.6},
                    ...
                ]
            },
            ...
        ]
    }
    """
    segments = []
    
    for i, seg in enumerate(whisper_result.get("segments", [])):
        segment_data = {
            "segment_id": i,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        
        segments.append(segment_data)
    
    # Calculate duration from last segment
    duration = 0
    if segments:
        duration = segments[-1]["end"]
    
    return {
        "video_name": video_name,
        "video_path": video_path,
        "full_text": whisper_result.get("text", "").strip(),
        "duration": round(duration, 2),
        "language": whisper_result.get("language", "vi"),
        "transcribed_at": datetime.now().isoformat(),
        "model_used": DEFAULT_WHISPER_MODEL,
        "total_segments": len(segments),
        "segments": segments
    }

# ============================================================
# MAIN PIPELINE
# ============================================================
def get_video_files(video_dir):
    """Get all video files from directory recursively."""
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, f))
    return video_files

def process_single_video(video_path, model, output_dir, audio_dir):
    """Process a single video: extract audio → transcribe → save JSON."""
    video_name = Path(video_path).name
    output_file = os.path.join(output_dir, f"{Path(video_path).stem}.json")
    
    # Skip if already processed
    if os.path.exists(output_file):
        logger.info(f"Already processed, skipping: {video_name}")
        return output_file, True
    
    try:
        # Step 1: Extract audio
        audio_path = extract_audio(video_path, audio_dir)
        if not audio_path:
            return None, False
        
        # Step 2: Transcribe
        logger.info(f"Transcribing: {video_name}")
        start_time = time.time()
        
        result = model.transcribe(
            audio_path,
            language="vi",
            task="transcribe",
            verbose=False,
            word_timestamps=False,
            fp16=True,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Transcription done in {elapsed:.1f}s: {video_name}")
        
        # Step 3: Format and save
        formatted = format_result_for_db(video_name, video_path, result)
        formatted["processing_time_seconds"] = round(elapsed, 1)
        
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved: {output_file}")
        
        # Step 4: Clean up audio to save disk space
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"Cleaned up audio: {audio_path}")
        
        return output_file, True
        
    except Exception as e:
        logger.error(f"Error processing {video_name}: {e}")
        return None, False

def main():
    parser = argparse.ArgumentParser(description="Video Transcription Pipeline")
    parser.add_argument("--local-dir", type=str, default=DEFAULT_VIDEO_DIR,
                        help="Directory containing video files")
    parser.add_argument("--drive-folder", type=str, default=None,
                        help="Google Drive folder ID to download from")
    parser.add_argument("--model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help="Whisper model to use (tiny, base, small, medium, large-v3)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save JSON transcriptions")
    parser.add_argument("--audio-dir", type=str, default=DEFAULT_AUDIO_DIR,
                        help="Directory for temporary audio files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only first video")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Number of videos to process (0 = all)")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Download from Google Drive if specified
    if args.drive_folder:
        logger.info("Downloading videos from Google Drive...")
        logger.info("TIP: For large folders, use rclone instead:")
        logger.info("  rclone copy gdrive:'folder/path' ./videos --include '*.mp4' -P")
        download_from_drive_rclone(args.drive_folder, args.local_dir)
    
    # Get video files
    video_files = get_video_files(args.local_dir)
    logger.info(f"Found {len(video_files)} video files")
    
    if not video_files:
        logger.error(f"No video files found in {args.local_dir}")
        logger.info("Please download videos first or specify correct path")
        sys.exit(1)
    
    # Load checkpoint for resume
    checkpoint = load_checkpoint()
    processed_set = set(checkpoint["processed"])
    
    # Filter out already processed videos
    if args.resume:
        video_files = [v for v in video_files if v not in processed_set]
        logger.info(f"Resuming: {len(processed_set)} already done, {len(video_files)} remaining")
    
    # Test mode
    if args.test:
        video_files = video_files[:1]
        logger.info("TEST MODE: Processing only first video")
    
    # Batch size
    if args.batch_size > 0:
        video_files = video_files[:args.batch_size]
        logger.info(f"Batch mode: Processing {args.batch_size} videos")
    
    # Load Whisper model once
    logger.info(f"Loading Whisper model: {args.model}")
    import whisper
    model = whisper.load_model(args.model)
    logger.info("Model loaded successfully!")
    
    # Process videos
    total = len(video_files)
    success_count = 0
    fail_count = 0
    
    for idx, video_path in enumerate(video_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing [{idx}/{total}]: {Path(video_path).name}")
        logger.info(f"{'='*60}")
        
        output_file, success = process_single_video(
            video_path, model, args.output_dir, args.audio_dir
        )
        
        if success:
            success_count += 1
            checkpoint["processed"].append(video_path)
        else:
            fail_count += 1
            checkpoint["failed"].append(video_path)
        
        # Save checkpoint after each video
        save_checkpoint(checkpoint)
        
        logger.info(f"Progress: {idx}/{total} | ✅ {success_count} | ❌ {fail_count}")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETED!")
    logger.info(f"Total: {total} | Success: {success_count} | Failed: {fail_count}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()

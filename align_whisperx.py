import whisperx
import torch
import gc
import argparse
import os
import sys
import json
import logging
from rapidfuzz import process, fuzz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds * 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def load_user_transcript(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        # Lọc dòng trống và strip whitespace
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def interpolate_timestamps_heuristic(user_lines, whisper_segments, audio_duration):
    """
    Map User Lines to Whisper Segments using character-length heuristic.
    Algorithm:
    1. Calculate total length of User Text.
    2. Calculate total length of Whisper Text within valid segments.
    3. Iterate through User Lines and assign start/end proportional to their length 
       relative to the Whisper text flow.
    """
    if not whisper_segments:
        return []

    # Flatten whisper segments to character timeline
    char_timeline = [] # List containing (cumulative_char_count, timestamp)
    cumulative_chars = 0
    
    # Add initial start point
    char_timeline.append((0, whisper_segments[0]["start"]))
    
    for seg in whisper_segments:
        text_len = len(seg["text"])
        start = seg["start"]
        end = seg["end"]
        
        # We assume linear distribution of characters within a segment
        # But simply using segment boundaries is robust enough for interpolation
        cumulative_chars += text_len
        char_timeline.append((cumulative_chars, end))
        
    total_whisper_chars = cumulative_chars
    
    # Calculate total user chars
    total_user_chars = sum(len(line) for line in user_lines)
    
    # Scale factor
    if total_user_chars == 0: return []
    scale = total_whisper_chars / total_user_chars
    
    # Helper to get time at char index
    def get_time(char_idx):
        target_char = char_idx * scale
        
        # Find in timeline
        for i in range(len(char_timeline) - 1):
            c1, t1 = char_timeline[i]
            c2, t2 = char_timeline[i+1]
            if c1 <= target_char <= c2:
                if c2 == c1: return t1
                ratio = (target_char - c1) / (c2 - c1)
                return t1 + ratio * (t2 - t1)
        return char_timeline[-1][1]

    aligned_segments = []
    current_char_idx = 0
    
    for line in user_lines:
        line_len = len(line)
        start_time = get_time(current_char_idx)
        end_time = get_time(current_char_idx + line_len)
        
        # Safety check: Ensure start < end
        if start_time >= end_time:
            end_time = start_time + 0.1
            
        aligned_segments.append({
            "text": line,
            "start": start_time,
            "end": end_time
        })
        current_char_idx += line_len
        
    return aligned_segments

def align_transcript_with_whisperx(audio_file, text_file, model_name="large-v2", device="cuda", compute_type="float16"):
    print(f"--- 🚀 Bắt đầu xử lý: {audio_file} ---")

    # 1. Load Audio
    if not os.path.exists(audio_file):
        logging.error(f"Không tìm thấy file audio: {audio_file}")
        return

    # 2. Transcribe để lấy 'Anchor Timestamps'
    logging.info("1️⃣  Đang chạy Whisper Transcribe để lấy khung thời gian (Anchor)...")
    try:
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_file)
        result_transcribe = model.transcribe(audio, batch_size=16)
        
        # Free memory model transcribe
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        whisper_segments = result_transcribe["segments"]
        logging.info(f"   ✅ Đã Transcribe xong: {len(whisper_segments)} segments.")
    except Exception as e:
        logging.error(f"Lỗi khi transcribe: {e}")
        return

    # 3. Đọc Text chuẩn từ file user
    logging.info("2️⃣  Đọc file Text chuẩn và mapping vào Audio...")
    user_lines = load_user_transcript(text_file)
    logging.info(f"   📖 Text chuẩn có {len(user_lines)} dòng.")

    # 4. Interpolate Timestamps (Heuristic Mapping)
    # Đây là bước quan trọng: gán timestamp giả định cho User Text dựa trên Whisper Text
    logging.info("3️⃣  Mapping User Text vào Whisper Timestamps (Heuristic)...")
    
    # Lấy duration thực tế từ audio nếu cần, nhưng whisper_segments[-1]['end'] là đủ
    audio_duration = whisper_segments[-1]['end'] if whisper_segments else 0
    
    rough_segments = interpolate_timestamps_heuristic(user_lines, whisper_segments, audio_duration)
    
    if not rough_segments:
        logging.error("Lỗi: Không tạo được mapping (có thể file text rỗng hoặc whisper không ra text)")
        return

    # 5. Chạy WhisperX Alignment (Phoneme-level)
    logging.info("4️⃣  Chạy Forced Alignment (căn chỉnh từ ngữ chính xác - Phoneme Level)...")
    try:
        # Load alignment model
        model_a, metadata = whisperx.load_align_model(language_code="vi", device=device)
        
        # Align
        result_align = whisperx.align(
            rough_segments, 
            model_a, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
        
        # Free memory align model
        del model_a
        gc.collect()
        torch.cuda.empty_cache()
        
        final_segments = result_align["segments"]
        logging.info(f"   ✅ Alignment thành công: {len(final_segments)} lines align tốt.")
        
    except Exception as e:
        logging.error(f"❌ Lỗi khi align với WhisperX: {e}")
        logging.warning("   ⚠️ Đang fallback về timestamps nội suy (Mappings sơ bộ)...")
        final_segments = rough_segments

    # 6. Xuất SRT
    output_srt = os.path.splitext(audio_file)[0] + ".srt"
    logging.info(f"5️⃣  Xuất file SRT: {output_srt}")
    
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(final_segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
            
    logging.info("✅ Hoàn tất!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool align text vào audio dùng WhisperX")
    parser.add_argument("--audio", required=True, help="Đường dẫn file audio input (.mp3, .wav)")
    parser.add_argument("--text", required=True, help="Đường dẫn file text input (mỗi câu 1 dòng)")
    parser.add_argument("--model", default="large-v2", help="Whisper model size (tiny, base, small, medium, large-v2)")
    
    args = parser.parse_args()
    
    align_transcript_with_whisperx(args.audio, args.text, model_name=args.model)

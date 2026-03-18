#!/usr/bin/env python3
"""
Hybrid Transcription: Whisper Timestamps + Gemini Text Correction
=================================================================
Bước 1: Chạy faster-whisper → lấy SRT với timestamp chuẩn (text có thể sai)
Bước 2: Gửi từng batch segments qua Gemini → sửa text, giữ nguyên timestamp
Bước 3: Xuất SRT cuối cùng (timestamp Whisper + text Gemini)

Usage:
    python hybrid_transcribe.py --audio "path/to/audio.mp3"
    python hybrid_transcribe.py --audio "path/to/audio.mp3" --model large-v3 --batch-size 30
"""

import os
import re
import json
import time
import argparse
from datetime import timedelta

import google.generativeai as genai
from google.api_core import exceptions

# --- CẤU HÌNH GEMINI ---
genai.configure(api_key="AIzaSyBkwEJFrbePfjfUhxPw9xaDIVA7S8ZaQ00")
GEMINI_MODEL = "gemini-3-pro-preview"

# --- CẤU HÌNH WHISPER ---
DEFAULT_WHISPER_MODEL = "large-v3"


# ============================================================
# BƯỚC 1: WHISPER - Lấy timestamp chuẩn
# ============================================================
def whisper_transcribe(audio_path, model_name=DEFAULT_WHISPER_MODEL):
    """Chạy faster-whisper để lấy segments với timestamp chính xác."""
    from faster_whisper import WhisperModel
    
    print(f"🔊 [Whisper] Đang tải model '{model_name}'...")
    model = WhisperModel(model_name, device="auto", compute_type="auto")
    
    initial_prompt = "Đây là bài giảng Phật pháp tiếng Việt, ngôn ngữ rõ ràng mạch lạc."
    
    print(f"🔊 [Whisper] Đang transcribe: {os.path.basename(audio_path)}")
    start = time.time()
    
    segments_iter, info = model.transcribe(
        audio_path,
        language="vi",
        task="transcribe",
        beam_size=5,
        best_of=5,
        condition_on_previous_text=False,
        temperature=[0.0, 0.2],
        compression_ratio_threshold=2.0,
        no_speech_threshold=0.6,
        repetition_penalty=1.2,
        initial_prompt=initial_prompt,
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.6,
            min_speech_duration_ms=250,
            max_speech_duration_s=20.0,
            min_silence_duration_ms=1000,
            speech_pad_ms=400,
        ),
    )
    
    segments = []
    for seg in segments_iter:
        # Bỏ qua segment lặp
        if segments and seg.text.strip() == segments[-1]["text"].strip():
            continue
        segments.append({
            "id": len(segments) + 1,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })
    
    elapsed = time.time() - start
    print(f"🔊 [Whisper] Xong! {len(segments)} segments trong {elapsed:.1f}s (audio: {info.duration:.0f}s)")
    
    # Giải phóng model khỏi VRAM
    del model
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass
    
    return segments


# ============================================================
# BƯỚC 2: GEMINI - Sửa text cho từng batch
# ============================================================
def gemini_correct_batch(segments_batch, batch_num, total_batches, max_retries=3):
    """
    Gửi 1 batch segments (có text Whisper) qua Gemini để sửa lại text.
    Gemini chỉ sửa text, KHÔNG động vào timestamp.
    Trả về list các text đã sửa (theo đúng thứ tự).
    """
    model = genai.GenerativeModel(model_name=GEMINI_MODEL)
    
    # Tạo danh sách text cần sửa, đánh số rõ ràng
    lines = []
    for seg in segments_batch:
        lines.append(f'[{seg["id"]}] {seg["text"]}')
    text_block = "\n".join(lines)
    
    prompt = f"""Bạn là chuyên gia chỉnh sửa phụ đề tiếng Việt. Dưới đây là các dòng phụ đề được tạo tự động từ một bài giảng Phật giáo.
Các dòng có thể bị sai chính tả, sai dấu, sai từ, hoặc thiếu dấu câu.

**Yêu cầu:**
1. Sửa chính tả và từ ngữ cho đúng nghĩa tiếng Việt.
2. Đặc biệt chú ý các thuật ngữ Phật học: chánh niệm, Bồ Tát, Niết Bàn, luân hồi, nhân quả, nghiệp báo, tái sinh, giác ngộ, v.v.
3. Giữ nguyên ý nghĩa gốc, KHÔNG tóm tắt, KHÔNG thêm bớt ý.
4. Giữ nguyên số thứ tự [ID] cho mỗi dòng.
5. Nếu dòng nào đã đúng rồi thì giữ nguyên.
6. Trả về ĐÚNG FORMAT: mỗi dòng bắt đầu bằng [ID] rồi text đã sửa.
7. KHÔNG thêm bất kỳ giải thích hay comment nào khác.

--- CÁC DÒNG CẦN SỬA ---
{text_block}
--- HẾT ---

Trả về các dòng đã sửa theo đúng format [ID] text:"""

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    for attempt in range(max_retries):
        try:
            print(f"   ✨ [Gemini] Batch {batch_num}/{total_batches} ({len(segments_batch)} segments) - Lần {attempt+1}...")
            
            response = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 8192, "temperature": 0.1},
                request_options={"timeout": 120},
                safety_settings=safety_settings,
            )
            
            candidate = response.candidates[0]
            if candidate.finish_reason not in (1, None):
                print(f"   ⚠️  Finish reason: {candidate.finish_reason}. Thử lại...")
                time.sleep(3)
                continue
            
            result_text = response.text.strip()
            
            # Parse kết quả: mỗi dòng có dạng [ID] text
            corrected = {}
            for line in result_text.split("\n"):
                line = line.strip()
                match = re.match(r'\[(\d+)\]\s*(.*)', line)
                if match:
                    seg_id = int(match.group(1))
                    corrected_text = match.group(2).strip()
                    corrected[seg_id] = corrected_text
            
            # Áp dụng text đã sửa vào segments
            corrected_count = 0
            for seg in segments_batch:
                if seg["id"] in corrected and corrected[seg["id"]]:
                    if seg["text"] != corrected[seg["id"]]:
                        corrected_count += 1
                    seg["text"] = corrected[seg["id"]]
            
            print(f"   ✅ [Gemini] Batch {batch_num}: Đã sửa {corrected_count}/{len(segments_batch)} segments")
            return segments_batch
            
        except exceptions.ResourceExhausted:
            print(f"   ⏳ Rate limited. Đợi 30s...")
            time.sleep(30)
        except exceptions.InternalServerError:
            print(f"   🔥 Server Error 500. Đợi 10s...")
            time.sleep(10)
        except exceptions.ServiceUnavailable:
            print(f"   🔥 Service Unavailable 503. Đợi 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            time.sleep(5)
    
    # Nếu tất cả retry đều thất bại → giữ nguyên text Whisper
    print(f"   ⚠️  Batch {batch_num} thất bại. Giữ nguyên text Whisper.")
    return segments_batch


def gemini_correct_all(segments, batch_size=30):
    """Chia segments thành batches và gửi qua Gemini sửa text."""
    print(f"\n📝 [Gemini] Bắt đầu sửa text cho {len(segments)} segments (batch_size={batch_size})...")
    
    total_batches = (len(segments) + batch_size - 1) // batch_size
    corrected_segments = []
    
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        
        corrected_batch = gemini_correct_batch(batch, batch_num, total_batches)
        corrected_segments.extend(corrected_batch)
        
        # Nghỉ giữa các batch để tránh rate limit
        if batch_num < total_batches:
            time.sleep(2)
    
    print(f"📝 [Gemini] Hoàn tất! {len(corrected_segments)} segments đã được xử lý.\n")
    return corrected_segments


# ============================================================
# BƯỚC 3: XUẤT SRT
# ============================================================
def format_timestamp_srt(seconds):
    """Convert giây → HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments, output_path):
    """Xuất file SRT chuẩn từ segments."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp_srt(seg["start"])
            end_ts = format_timestamp_srt(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n")
    print(f"💾 SRT đã lưu: {output_path} ({len(segments)} segments)")


def write_comparison(segments_whisper, segments_gemini, output_path):
    """Xuất file so sánh text Whisper vs Gemini (để kiểm tra chất lượng)."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SO SÁNH: Whisper (gốc) vs Gemini (đã sửa)\n")
        f.write("=" * 80 + "\n\n")
        
        changed = 0
        for w, g in zip(segments_whisper, segments_gemini):
            if w["text"] != g["text"]:
                changed += 1
                f.write(f"[{w['id']}] {format_timestamp_srt(w['start'])} → {format_timestamp_srt(w['end'])}\n")
                f.write(f"  Whisper: {w['text']}\n")
                f.write(f"  Gemini:  {g['text']}\n\n")
        
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Tổng cộng: {changed}/{len(segments_whisper)} segments đã thay đổi\n")
    
    print(f"📊 Comparison đã lưu: {output_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Transcription: Whisper Timestamps + Gemini Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio", required=True, help="Đường dẫn file audio")
    parser.add_argument("--model", default=DEFAULT_WHISPER_MODEL, help=f"Whisper model (mặc định: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--batch-size", type=int, default=30, help="Số segments mỗi batch gửi Gemini (mặc định: 30)")
    parser.add_argument("--output-dir", default=".", help="Thư mục output (mặc định: thư mục hiện tại)")
    parser.add_argument("--whisper-only", action="store_true", help="Chỉ chạy Whisper, không sửa bằng Gemini")
    parser.add_argument("--gemini-only", type=str, default=None, 
                        help="Chỉ chạy Gemini sửa text từ file SRT Whisper có sẵn (truyền path file .srt)")
    
    args = parser.parse_args()
    
    audio_file = args.audio
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    srt_whisper_path = os.path.join(output_dir, f"{base_name}_whisper.srt")
    srt_final_path = os.path.join(output_dir, f"{base_name}.srt")
    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.txt")
    
    # ========== CHẾ ĐỘ: Chỉ Gemini (từ SRT có sẵn) ==========
    if args.gemini_only:
        print(f"📂 Chế độ: Gemini-only, đọc SRT từ {args.gemini_only}")
        segments = parse_srt_file(args.gemini_only)
        if not segments:
            print("❌ Không parse được SRT!")
            return
        
        import copy
        segments_whisper_copy = copy.deepcopy(segments)
        
        segments = gemini_correct_all(segments, batch_size=args.batch_size)
        write_srt(segments, srt_final_path)
        write_comparison(segments_whisper_copy, segments, comparison_path)
        
        print(f"\n🎉 HOÀN TẤT!")
        print(f"   📄 SRT cuối: {srt_final_path}")
        print(f"   📊 So sánh:  {comparison_path}")
        return
    
    # ========== CHẾ ĐỘ THƯỜNG: Whisper → Gemini ==========
    if not os.path.exists(audio_file):
        print(f"❌ Không tìm thấy file: {audio_file}")
        return
    
    # Bước 1: Whisper
    print("=" * 60)
    print("BƯỚC 1: Whisper - Lấy timestamp")
    print("=" * 60)
    segments = whisper_transcribe(audio_file, model_name=args.model)
    
    if not segments:
        print("❌ Whisper không tạo được segment nào!")
        return
    
    # Lưu SRT Whisper gốc
    write_srt(segments, srt_whisper_path)
    
    if args.whisper_only:
        # Nếu chỉ chạy Whisper
        print(f"\n🎉 HOÀN TẤT (Whisper only)!")
        print(f"   📄 SRT Whisper: {srt_whisper_path}")
        return
    
    # Bước 2: Gemini sửa text
    print("=" * 60)
    print("BƯỚC 2: Gemini - Sửa text")  
    print("=" * 60)
    
    import copy
    segments_whisper_copy = copy.deepcopy(segments)
    
    segments = gemini_correct_all(segments, batch_size=args.batch_size)
    
    # Bước 3: Xuất SRT cuối cùng
    print("=" * 60)
    print("BƯỚC 3: Xuất kết quả")
    print("=" * 60)
    
    write_srt(segments, srt_final_path)
    write_comparison(segments_whisper_copy, segments, comparison_path)
    
    print(f"\n🎉 HOÀN TẤT!")
    print(f"   📄 SRT Whisper (gốc): {srt_whisper_path}")
    print(f"   📄 SRT cuối (hybrid): {srt_final_path}")
    print(f"   📊 So sánh:           {comparison_path}")


def parse_srt_file(srt_path):
    """Parse file SRT thành list segments."""
    if not os.path.exists(srt_path):
        print(f"❌ Không tìm thấy file SRT: {srt_path}")
        return []
    
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    segments = []
    entries = re.split(r'\n\n+', content.strip())
    
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # Parse timestamp line: 00:00:01,000 --> 00:00:05,000
        ts_match = re.match(
            r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
            lines[1].strip()
        )
        if not ts_match:
            continue
        
        start_secs = srt_timestamp_to_seconds(ts_match.group(1))
        end_secs = srt_timestamp_to_seconds(ts_match.group(2))
        text = "\n".join(lines[2:]).strip()
        
        segments.append({
            "id": len(segments) + 1,
            "start": start_secs,
            "end": end_secs,
            "text": text,
        })
    
    print(f"📂 Đã parse {len(segments)} segments từ {srt_path}")
    return segments


def srt_timestamp_to_seconds(ts):
    """Convert HH:MM:SS,mmm → seconds."""
    h, m, s_ms = ts.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


if __name__ == "__main__":
    main()

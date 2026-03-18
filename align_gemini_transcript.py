import whisperx
import torch
import gc
import argparse
import os

def align_transcript(audio_file, text_file, output_dir="."):
    # Cấu hình
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 # Giảm xuống nếu ít VRAM
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    print(f"1. Đang tải audio: {audio_file} trên thiết bị {device}...")
    if not os.path.exists(audio_file):
        print(f"Lỗi: Không tìm thấy file audio {audio_file}")
        return

    audio = whisperx.load_audio(audio_file)

    # 2. Đọc file text từ Gemini và tạo cấu trúc giả lập để WhisperX hiểu
    print(f"2. Đang đọc transcript từ {text_file}...")
    if not os.path.exists(text_file):
        print(f"Lỗi: Không tìm thấy file text {text_file}")
        return

    with open(text_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Tạo các segment giả định (WhisperX cần input là list các dict hoặc dict có key 'segments')
    # WhisperX align mong đợi một list các segments
    segments = [{"text": line, "start": 0.0, "end": 0.0} for line in lines] 
    
    # Một số phiên bản WhisperX mong đợi input dạng {"segments": [...]}
    transcript_obj = {"segments": segments, "language": "vi"}

    # 3. Load Model Alignment (Wav2Vec2 cho tiếng Việt)
    print("3. Đang tải model alignment (có thể mất thời gian lần đầu)...")
    try:
        model_a, metadata = whisperx.load_align_model(language_code="vi", device=device)
    except Exception as e:
        print(f"Lỗi khi load model alignment: {e}")
        print("Đang thử load model mặc định 'VI'...")
        model_a, metadata = whisperx.load_align_model(language_code="vi", device=device)

    # 4. Thực hiện Alignment
    print("4. Bắt đầu căn chỉnh thời gian (Alignment)...")
    try:
        # Lưu ý: WhisperX trả về kết quả đã align
        result = whisperx.align(
            segments, # Chuyền list segments vào
            model_a, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
    except Exception as e:
        print(f"Lỗi khi align: {e}")
        return

    # 5. Xuất ra file SRT
    print("5. Đang xuất file SRT...")
    from whisperx.utils import get_writer
    
    # Tự động đặt tên file output cùng tên với audio file nhưng đuôi .srt
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_srt = os.path.join(output_dir, base_name + ".srt")
    
    # WhisperX writer - cần truyền đủ options mà nó yêu cầu
    writer = get_writer("srt", output_dir)
    writer_options = {
        "max_line_width": None,     # Không giới hạn độ dài dòng
        "max_line_count": None,     # Không giới hạn số dòng
        "highlight_words": False,   # Không highlight từ
    }
    writer(result, audio_file, writer_options)
    
    print(f"Hoàn tất! File SRT đã được lưu tại thư mục: {output_dir}")
    print(f"Tên file dự kiến: {base_name}.srt")

    # Dọn dẹp bộ nhớ
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công cụ align text vào audio dùng WhisperX")
    parser.add_argument("--audio", type=str, default="video_phat_giao.mp3", help="Đường dẫn file audio/video")
    parser.add_argument("--text", type=str, default="transcript_tu_gemini.txt", help="Đường dẫn file text (mỗi câu 1 dòng)")
    
    args = parser.parse_args()
    
    align_transcript(args.audio, args.text)

import google.generativeai as genai
import time
import os

# --- CẤU HÌNH ---
# Thay API Key MỚI của bạn vào đây (Đừng dùng key cũ đã lộ)
genai.configure(api_key="AIzaSyCLN0Gjoc-prWKDEWSQA0sQ6CUhSO2Fl7I") 

# Model
MODEL_NAME = "gemini-3-pro-preview" 

def process_audio(file_path):
    print(f"--> Đang xử lý: {os.path.basename(file_path)}")
    
    try:
        # 1. Upload file
        print("   Uploading...")
        audio_file = genai.upload_file(path=file_path)

        # Đợi file active
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = genai.get_file(audio_file.name)

        if audio_file.state.name == "FAILED":
            raise ValueError("Upload thất bại.")

        # 2. Gửi yêu cầu (Prompt kỹ thuật để ra đúng chuẩn SRT)
        print("   Generating subtitle...")
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        prompt = """
        Nghe file âm thanh đính kèm và tạo phụ đề chính xác từng từ (verbatim).
        **Yêu cầu kỹ thuật:**
        1. **Độ chính xác:** Timestamp phải khớp chính xác với âm thanh.
        3. **Trình bày:** Ngắt dòng tự nhiên theo ngữ điệu. Tối đa 40 ký tự trên một dòng để đảm bảo hiển thị tốt trên mọi màn hình.
        4. **Nội dung:** Tuyệt đối không tóm tắt, không bỏ sót từ nào.
        Output bắt buộc phải ở định dạng chuẩn SRT (SubRip).
        Không bao gồm bất kỳ lời dẫn hay markdown code block nào (như ```srt), chỉ trả về plain text của nội dung SRT.
        """
        
        response = model.generate_content(
            [audio_file, prompt],
            request_options={"timeout": 600}
        )

        # 3. Dọn dẹp trên cloud
        genai.delete_file(audio_file.name)
        
        return response.text

    except Exception as e:
        print(f"   LỖI: {e}")
        return None

# --- CHẠY THỬ NGHIỆM VỚI 1 FILE ---
if __name__ == "__main__":
    # Lưu ý: Thêm chữ r trước đường dẫn để tránh lỗi ký tự Windows
    target_file = r"F:\1-audio-bai-giang\Ai ăn nấy no, ai tu nấy chứng.mp3"
    
    if os.path.exists(target_file):
        srt_content = process_audio(target_file)
        
        if srt_content:
            # Tạo tên file output: thay đuôi .mp3 bằng .srt
            output_file = os.path.splitext(target_file)[0] + ".srt"
            
            # Lưu file với encoding utf-8 để hiển thị tiếng Việt đúng
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(srt_content)
            
            print(f"✅ Xong! Đã lưu tại: {output_file}")
    else:
        print("❌ Không tìm thấy file.")
# -*- coding: utf-8 -*-
import os
import argparse
import time
import re
from google.api_core.client_options import ClientOptions
from google.cloud import speech_v2
from google.cloud import storage

# --- CẤU HÌNH ---
LOCATION = "us-central1"
# Model Chirp 2 (Universal Speech Model mới nhất)
MODEL_ID = "chirp_2" 

def sanitize_filename(filename):
    """Chuyển đổi tên file sang dạng an toàn (không dấu, không ký tự lạ)"""
    # Bỏ dấu tiếng Việt
    s = filename
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ÙÚỤỦŨƯỪỨỰỬỮ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[đ]', 'd', s)
    s = re.sub(r'[Đ]', 'D', s)
    # Thay thế ký tự lạ bằng _
    s = re.sub(r'[^a-zA-Z0-9\._-]', '_', s)
    return s

def upload_to_gcs(local_path, bucket_name, dest_blob_name):
    """Upload file với cấu hình tối ưu cho mạng yếu/file lớn"""
    print(f"1. ☁️  Đang upload lên GCS: gs://{bucket_name}/{dest_blob_name}")
    print("      (Vui lòng đợi, không tắt cửa sổ này...)")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    
    # Chunk size 10MB để upload ổn định hơn
    blob.chunk_size = 10 * 1024 * 1024 

    with open(local_path, "rb") as f:
        # Timeout upload tối đa 20 phút
        blob.upload_from_file(f, timeout=1200)
        
    return f"gs://{bucket_name}/{dest_blob_name}"

def delete_blob(bucket_name, blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        print(f"   🗑️  Đã dọn dẹp file tạm trên GCS.")
    except:
        pass

def result_to_srt(transcript_results):
    """Chuyển JSON result thành SRT format"""
    if not transcript_results:
        return ""
    
    srt_output = []
    index = 1
    all_words = []

    # 1. Gom tất cả từ lại
    for result in transcript_results:
        if result.alternatives:
            top_alt = result.alternatives[0]
            for word_info in top_alt.words:
                start_sec = word_info.start_offset.total_seconds()
                end_sec = word_info.end_offset.total_seconds()
                word_text = word_info.word.strip()
                all_words.append({'word': word_text, 'start': start_sec, 'end': end_sec})

    if not all_words:
        return ""

    # 2. Chia dòng (tối đa 45 ký tự hoặc ngắt quãng > 0.8s)
    current_line = []
    for i, w in enumerate(all_words):
        current_line.append(w)
        line_len = sum(len(x['word']) + 1 for x in current_line)
        
        should_break = False
        if i == len(all_words) - 1:
            should_break = True
        else:
            next_w = all_words[i+1]
            gap = next_w['start'] - w['end']
            if gap > 0.8 or line_len > 45: 
                should_break = True
        
        if should_break:
            # Format time: HH:MM:SS,mmm
            def fmt(s):
                import datetime
                dt = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=s)
                return dt.strftime("%H:%M:%S,%f")[:-3]

            start_str = fmt(current_line[0]['start'])
            end_str = fmt(current_line[-1]['end'])
            text = " ".join([x['word'] for x in current_line])
            
            srt_output.append(str(index))
            srt_output.append(f"{start_str} --> {end_str}")
            srt_output.append(text)
            srt_output.append("")
            index += 1
            current_line = []

    return "\n".join(srt_output)

def transcribe_chirp(project_id, input_uri, output_path):
    # Cấu hình Client
    api_endpoint = f"{LOCATION}-speech.googleapis.com"
    client_options = ClientOptions(api_endpoint=api_endpoint)
    client = speech_v2.SpeechClient(client_options=client_options)
    
    recognizer_id = f"chirp-rec-{int(time.time())}"
    parent = f"projects/{project_id}/locations/{LOCATION}"
    
    created_recognizer = None

    try:
        print(f"2. ⚙️  Đang khởi tạo Recognizer ({MODEL_ID}) ...")
        
        # --- PHẦN SỬA LỖI QUAN TRỌNG ---
        # Thay vì gọi class AutoDecodingConfig(), ta truyền DICT {}
        # Thư viện sẽ tự hiểu đây là cấu hình mặc định (Auto Decode)
        recognition_config = speech_v2.RecognitionConfig(
            auto_decoding_config={},  # <--- FIX: Truyền dict rỗng thay vì class
            language_codes=["vi-VN"], 
            model=MODEL_ID,
            features=speech_v2.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True
            )
        )
        
        recognizer_request = speech_v2.CreateRecognizerRequest(
            parent=parent,
            recognizer_id=recognizer_id,
            recognizer=speech_v2.Recognizer(
                default_recognition_config=recognition_config
            )
        )
        # -------------------------------
        
        operation = client.create_recognizer(request=recognizer_request)
        created_recognizer = operation.result(timeout=120)
        print(f"   ✅ Đã tạo Recognizer thành công: {created_recognizer.name}")
        
        # 3. Gửi Batch Request
        print(f"3. 🚀 Đang gửi yêu cầu Batch Recognize...")
        
        batch_request = speech_v2.BatchRecognizeRequest(
            recognizer=created_recognizer.name,
            files=[speech_v2.BatchRecognizeFileMetadata(uri=input_uri)],
            recognition_output_config=speech_v2.RecognitionOutputConfig(
                inline_response_config=speech_v2.InlineOutputConfig()
            )
        )

        batch_op = client.batch_recognize(request=batch_request)
        print("   ⏳ Đang chờ Google xử lý (Chirp chạy Batch nên sẽ mất vài phút)...")
        
        # Chờ kết quả (tối đa 60 phút)
        result = batch_op.result(timeout=3600)

        # 4. Xử lý kết quả
        for uri, file_result in result.results.items():
            if file_result.error and file_result.error.code != 0:
                print(f"❌ Lỗi từ Google: {file_result.error.message}")
                continue

            if file_result.inline_result:
                print("   ✅ Đã nhận dữ liệu, đang tạo SRT...")
                srt_content = result_to_srt(file_result.inline_result.transcript.results)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                print(f"\n🎉 HOÀN TẤT! File SRT lưu tại:\n   {output_path}")
            else:
                print("⚠️ Không có kết quả trả về (Inline result empty).")

    except Exception as e:
        print(f"\n❌ CÓ LỖI XẢY RA: {e}")
        # In thêm chi tiết lỗi nếu là lỗi API
        if hasattr(e, 'message'):
            print(f"   Chi tiết: {e.message}")
    
    finally:
        if created_recognizer:
            try:
                print("   🧹 Đang xóa Recognizer tạm...")
                client.delete_recognizer(request=speech_v2.DeleteRecognizerRequest(name=created_recognizer.name))
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="File audio local")
    parser.add_argument("--bucket", required=True, help="Bucket GCS")
    parser.add_argument("--project", required=True, help="Project ID")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File không tồn tại: {args.file}")
        exit(1)

    # Tự động đổi tên file upload cho an toàn
    clean_bucket = args.bucket.replace("gs://", "").strip("/")
    base_name = os.path.basename(args.file)
    safe_name = sanitize_filename(base_name)
    gcs_path = f"temp_uploads/{safe_name}"
    
    try:
        gcs_uri = upload_to_gcs(args.file, clean_bucket, gcs_path)
        srt_file = os.path.splitext(args.file)[0] + ".srt"
        transcribe_chirp(args.project, gcs_uri, srt_file)
        delete_blob(clean_bucket, gcs_path)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
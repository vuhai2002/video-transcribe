# Video Transcription Pipeline - Hướng dẫn chi tiết

## 🎯 Mục tiêu
Trích xuất subtitle từ ~1500 video bài giảng tiếng Việt trên Google Drive,
lưu vào PostgreSQL để tìm kiếm theo từ khóa với timestamp chính xác.

---

## 📁 Cấu trúc project

```
~/transcribe-project/
├── transcribe.py          # Script chính: tải video → tách audio → Whisper
├── import_to_db.py        # Import JSON kết quả vào PostgreSQL
├── db_schema.sql          # Schema database
├── videos/                # Thư mục chứa video tải từ Google Drive
├── audio/                 # Thư mục chứa audio tạm (tự xóa sau khi xong)
├── output/                # Thư mục chứa kết quả JSON
├── checkpoint.json        # File theo dõi tiến trình (auto-resume)
└── transcribe.log         # Log file
```

---

## 🚀 HƯỚNG DẪN TỪNG BƯỚC

### Bước 1: Cài đặt rclone để kết nối Google Drive

```bash
# Cài rclone
curl https://rclone.org/install.sh | sudo bash

# Cấu hình Google Drive
rclone config
```

Khi chạy `rclone config`, chọn:
1. `n` (New remote)
2. Đặt tên: `gdrive`
3. Chọn `drive` (Google Drive)
4. Client ID: để trống (Enter)
5. Client Secret: để trống (Enter)
6. Scope: chọn `1` (Full access)
7. Root folder ID: để trống
8. Service Account: để trống
9. Advanced config: `n`
10. Auto config: `n` (vì đang chạy trên server không có GUI)
11. Nó sẽ cho bạn 1 link → copy link đó, paste vào trình duyệt trên máy tính local
12. Đăng nhập Google, cho phép quyền truy cập
13. Copy code mà Google trả về, paste lại vào terminal
14. Shared drive: `n` (nếu không dùng Shared Drive)
15. Confirm: `y`

### Bước 2: Kiểm tra kết nối Google Drive

```bash
# Liệt kê các file/folder trên Drive
rclone ls gdrive: --max-depth 1

# Liệt kê folder cụ thể chứa video
rclone ls gdrive:"Tên Folder Video" --include "*.mp4" | head -20
```

### Bước 3: Tải video từ Google Drive

```bash
# Tạo thư mục project
mkdir -p ~/transcribe-project/videos
cd ~/transcribe-project

# Tải TẤT CẢ video mp4 từ folder trên Drive
# Thay "Tên Folder Video" bằng tên folder thực tế của bạn
rclone copy gdrive:"Tên Folder Video" ./videos \
    --include "*.mp4" \
    --progress \
    --transfers 5 \
    --checkers 5 \
    --stats 10s \
    -v

# Hoặc nếu muốn tải từng batch (ví dụ 10 video đầu tiên để test)
rclone copy gdrive:"Tên Folder Video" ./videos \
    --include "*.mp4" \
    --max-transfer 5G \
    --progress \
    -v
```

### Bước 4: Upload script lên VM

**Từ máy local (PowerShell):**
```powershell
# Upload các script lên VM
scp C:\Users\nghia\.gemini\antigravity\scratch\video-transcribe\transcribe.py vuhai8617@34.173.154.244:~/transcribe-project/
scp C:\Users\nghia\.gemini\antigravity\scratch\video-transcribe\import_to_db.py vuhai8617@34.173.154.244:~/transcribe-project/
scp C:\Users\nghia\.gemini\antigravity\scratch\video-transcribe\db_schema.sql vuhai8617@34.173.154.244:~/transcribe-project/
```

### Bước 5: Test với 1 video trước

```bash
cd ~/transcribe-project

# Test với 1 video
python3 transcribe.py --local-dir ./videos --model large-v3 --test

# Kiểm tra kết quả
ls -la output/
cat output/*.json | python3 -m json.tool | head -50
```

### Bước 6: Chạy transcription cho toàn bộ video

```bash
# Chạy trong screen/tmux để không bị mất khi disconnect
sudo apt install -y tmux
tmux new -s transcribe

# Chạy transcription toàn bộ
cd ~/transcribe-project
python3 transcribe.py --local-dir ./videos --model large-v3 --resume

# Nếu muốn chạy theo batch (50 video/batch)
python3 transcribe.py --local-dir ./videos --model large-v3 --batch-size 50 --resume

# Để thoát tmux mà không dừng script: Ctrl+B rồi D
# Để quay lại tmux: tmux attach -t transcribe
```

---

## ⏱️ Ước tính thời gian & chi phí

| Thông số | Giá trị |
|---|---|
| Số video | ~1500 |
| Độ dài TB | ~1 giờ/video |
| GPU | Tesla T4 (15GB VRAM) |
| Model | large-v3 |
| Tốc độ xử lý | ~10-15 phút/video 1 giờ (Real-time factor ~0.2x) |
| Tổng thời gian | ~250-375 giờ (~10-16 ngày) |
| Chi phí Spot VM | ~$0.35/giờ (N2-standard-4 + T4 Spot) |
| Tổng chi phí | ~$87-131 (trong $300 credit) |

**💡 Mẹo tiết kiệm:**
- Nếu muốn nhanh hơn, dùng `medium` model thay vì `large-v3` (nhanh gấp 2-3x, độ chính xác giảm nhẹ)
- Có thể tạo 2-3 VM Spot chạy song song (chia video ra)

---

## 📊 Cấu trúc JSON output (để import vào DB)

```json
{
  "video_name": "bai_giang_lich_su_01.mp4",
  "video_path": "./videos/bai_giang_lich_su_01.mp4",
  "full_text": "Xin chào các em, hôm nay chúng ta sẽ học về vua Quang Trung...",
  "duration": 3605.2,
  "language": "vi",
  "model_used": "large-v3",
  "total_segments": 450,
  "processing_time_seconds": 823.5,
  "transcribed_at": "2026-02-10T04:00:00",
  "segments": [
    {
      "segment_id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Xin chào các em, hôm nay chúng ta sẽ học về vua Quang Trung",
      "words": [
        {"word": "Xin", "start": 0.0, "end": 0.3, "probability": 0.95},
        {"word": "chào", "start": 0.3, "end": 0.6, "probability": 0.97},
        ...
      ]
    },
    {
      "segment_id": 1,
      "start": 5.2,
      "end": 11.8,
      "text": "Vua Quang Trung tên thật là Nguyễn Huệ",
      "words": [...]
    }
  ]
}
```

---

## 🔍 Ví dụ truy vấn tìm kiếm (sau khi import vào DB)

```sql
-- Tìm tất cả video có chứa "vua Quang Trung"
SELECT * FROM search_subtitles('vua Quang Trung');

-- Tìm tất cả timestamp trong 1 video cụ thể
SELECT * FROM get_video_matches(42, 'vua Quang Trung');

-- Tìm kiếm nhanh bằng ILIKE
SELECT v.video_name, ss.start_time, ss.end_time, ss.text
FROM subtitle_segments ss
JOIN videos v ON v.id = ss.video_id
WHERE ss.text ILIKE '%vua Quang Trung%'
ORDER BY v.video_name, ss.start_time;
```

---

## ⚠️ Lưu ý quan trọng

1. **Dùng tmux/screen**: Luôn chạy script trong tmux để không mất tiến trình khi mất kết nối SSH
2. **Auto-resume**: Script có checkpoint, nếu bị dừng giữa chừng, chạy lại với `--resume`
3. **Disk space**: Mỗi video 1 giờ ≈ 500MB-1GB. Đảm bảo có đủ disk (132GB hiện tại)
4. **Tải video theo batch**: Không cần tải hết 1500 video cùng lúc. Tải 50-100 video, xử lý, xóa, rồi tải tiếp
5. **Spot VM có thể bị tắt**: Google có thể tắt Spot VM bất kỳ lúc nào. Script có checkpoint nên không mất data

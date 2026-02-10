# Video Transcription Pipeline - Hướng Dẫn Chi Tiết

## 🎯 Mục tiêu
Trích xuất subtitle từ ~1500 video bài giảng tiếng Việt trên Google Drive,
lưu JSON/SRT/MP3 lên Drive, sau đó import vào PostgreSQL để tìm kiếm.

**Engine:** faster-whisper (CTranslate2) + Silero VAD → chính xác, nhanh gấp 4x vanilla Whisper.

---

## 📁 Cấu trúc project

### Trên VM (Google Cloud):
```
~/transcribe-work/           ← Thư mục làm việc tạm
├── download/                ← Video đang tải (xóa sau khi extract audio)
├── audio/                   ← Audio tạm (xóa sau khi transcribe + upload)
├── output/                  ← JSON/SRT tạm (xóa sau khi upload)
├── logs/                    ← Log files
│   └── transcribe_20260210_100000.log
└── checkpoint.json          ← Bản sao local (bản chính lưu trên Drive)
```

### Trên Google Drive:
```
Output/                      ← Kết quả (drive-output)
├── JSON/                    ← Full transcription data
│   ├── bai_giang_01.json
│   └── ...
├── SRT/                     ← Subtitle files
│   ├── bai_giang_01.srt
│   └── ...
├── Audio/                   ← MP3 files
│   ├── bai_giang_01.mp3
│   └── ...
└── checkpoint.json          ← CHECKPOINT (bản chính, an toàn khi VM bị tắt)
```

---

## 🔄 QUY TRÌNH XỬ LÝ (Pipeline)

```
┌─────────────────────────────────────────────────────────────────┐
│                   PIPELINE GỐI ĐẦU                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Thread Prefetcher (CPU/Network):                               │
│    ┌──────────┐   ┌──────────┐   ┌──────────┐                  │
│    │Download B│──▶│Extract B │──▶│Download C│──▶ ...            │
│    │from Drive│   │MP4→MP3   │   │from Drive│                   │
│    └──────────┘   └──────────┘   └──────────┘                  │
│                                                                 │
│  Thread Main (GPU):                                             │
│    ┌────────────┐   ┌──────────┐   ┌────────────┐              │
│    │Transcribe A│──▶│Upload A  │──▶│Transcribe B│──▶ ...       │
│    │(Whisper)   │   │JSON+SRT  │   │(Whisper)   │              │
│    │            │   │+MP3→Drive│   │            │              │
│    └────────────┘   └──────────┘   └────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quy trình từng video:

```
Step 1: [Prefetcher] Tải MP4 từ Google Drive về VM
         ↓
Step 2: [Prefetcher] Extract audio: MP4 → MP3 (ffmpeg)
         ↓
Step 3: [Prefetcher] Xóa MP4 (chỉ giữ MP3)
         ↓
Step 4: [Main] Whisper transcribe MP3 → segments (GPU)
         ↓
Step 5: [Main] Tạo file JSON (full data) + SRT (subtitle)
         ↓
Step 6: [Main] Upload JSON + SRT + MP3 lên Google Drive
         ↓
Step 7: [Main] Cập nhật checkpoint.json lên Google Drive
         ↓
Step 8: [Main] Xóa JSON + SRT + MP3 trên VM
         ↓
Step 9: → Tiếp tục video tiếp theo
```

---

## 🛡️ CHECKPOINT (An toàn cho Spot Instance)

Checkpoint được lưu **TRÊN GOOGLE DRIVE**, không phải local VM.

### Trạng thái video:
- **`processing`**: Đang xử lý (script ghi khi BẮT ĐẦU)
- **`done`**: Đã xong (script ghi khi HOÀN THÀNH)
- **`failed`**: Lỗi (script ghi khi GẶP LỖI)

### Khi VM khởi động lại:
1. Script đọc `checkpoint.json` từ Google Drive
2. Video có status `done` → **Bỏ qua**
3. Video có status `processing` → **Làm lại** (bị crash giữa chừng)
4. Video có status `failed` → **Thử lại**
5. Video chưa có trong checkpoint → **Xử lý mới**

### Format checkpoint.json:
```json
{
  "videos": {
    "BaiGiang/bai_01.mp4": {
      "status": "done",
      "started_at": "2026-02-10T10:00:00",
      "completed_at": "2026-02-10T10:15:00",
      "processing_time_seconds": 900.0,
      "output_json": "Output/JSON/bai_01.json",
      "output_srt": "Output/SRT/bai_01.srt",
      "output_mp3": "Output/Audio/bai_01.mp3"
    },
    "BaiGiang/bai_02.mp4": {
      "status": "processing",
      "started_at": "2026-02-10T10:15:00"
    }
  },
  "stats": {
    "total_processed": 1,
    "total_failed": 0,
    "last_updated": "2026-02-10T10:15:00"
  }
}
```

---

## 🚀 HƯỚNG DẪN CHẠY

### Bước 1: Upload script lên VM (từ Windows)

```powershell
cd D:\source-code\video-transcribe
scp transcribe.py vuhai8617@34.173.154.244:~/
```

### Bước 2: SSH vào VM và chạy

```bash
ssh vuhai8617@34.173.154.244

# Di chuyển script
mv ~/transcribe.py ~/transcribe.py

# Cài tmux (để giữ script chạy khi disconnect SSH)
sudo apt install -y tmux

# Tạo session tmux
tmux new -s transcribe
```

### Bước 3: Chạy Pipeline

```bash
# Test với 1 video trước
python3 ~/transcribe.py \
    --drive-input "TEN_FOLDER_VIDEO" \
    --drive-output "Output" \
    --test

# Nếu test OK, chạy toàn bộ
python3 ~/transcribe.py \
    --drive-input "TEN_FOLDER_VIDEO" \
    --drive-output "Output"

# Hoặc chạy theo batch 50 video
python3 ~/transcribe.py \
    --drive-input "TEN_FOLDER_VIDEO" \
    --drive-output "Output" \
    --batch-size 50
```

### Bước 4: Thoát tmux (giữ script chạy)

```
Ctrl+B rồi D    ← Thoát tmux, script vẫn chạy
tmux attach -t transcribe    ← Quay lại xem
```

---

## ⏱️ ƯỚC TÍNH THỜI GIAN & CHI PHÍ

| Thông số | Giá trị |
|---|---|
| Số video | ~1500 |
| Độ dài TB | ~1 giờ/video |
| GPU | Tesla T4 (15GB VRAM) |
| Model | large-v3 |
| Thời gian transcribe | ~3-8 phút/video (faster-whisper + VAD) |
| Thời gian download+extract | ~3-5 phút/video (chạy song song) |
| Tổng thời gian | ~100-200 giờ (~4-8 ngày) |
| Chi phí Spot VM | ~$0.35/giờ |
| Tổng chi phí | ~$35-70 (trong $300 credit) |

---

## 📊 FORMAT OUTPUT

### JSON (cho Database):
```json
{
  "video_name": "bai_giang_01.mp4",
  "drive_path": "BaiGiang/bai_giang_01.mp4",
  "full_text": "Xin chào các em, hôm nay...",
  "duration": 3605.2,
  "language": "vi",
  "model_used": "large-v3",
  "total_segments": 450,
  "processing_time_seconds": 823.5,
  "segments": [
    {
      "segment_id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Xin chào các em hôm nay chúng ta sẽ học về vua Quang Trung"
    }
  ]
}
```

### SRT (cho xem phụ đề):
```srt
1
00:00:00,000 --> 00:00:05,200
Xin chào các em hôm nay chúng ta sẽ học về vua Quang Trung

2
00:00:05,200 --> 00:00:11,800
Vua Quang Trung tên thật là Nguyễn Huệ
```

---

## 🔧 AUTO-START (Systemd Service)

Để script tự chạy khi VM khởi động lại (Spot Instance):

```bash
sudo tee /etc/systemd/system/transcribe.service << 'EOF'
[Unit]
Description=Video Transcription Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=vuhai8617
WorkingDirectory=/home/vuhai8617
ExecStart=/usr/bin/python3 /home/vuhai8617/transcribe.py --drive-input "TEN_FOLDER_VIDEO" --drive-output "Output"
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable transcribe.service
sudo systemctl start transcribe.service

# Xem log
sudo journalctl -u transcribe.service -f
```

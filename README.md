# Video Transcription Pipeline - Hướng Dẫn Chi Tiết

## 🎯 Mục tiêu
Trích xuất subtitle từ ~1500 video bài giảng tiếng Việt trên Google Drive,
lưu JSON/SRT lên Drive, sau đó import vào PostgreSQL để tìm kiếm.

**Engine:** faster-whisper (CTranslate2) + Silero VAD → chính xác, nhanh gấp 4x vanilla Whisper.

---

## 📁 Cấu trúc project

### Files:
```
├── convert_audio.py     ← BƯỚC 1: Video → MP3 → Upload lên Drive
├── transcribe.py        ← BƯỚC 2: MP3 → Transcribe → JSON/SRT lên Drive
├── import_to_db.py      ← BƯỚC 3: Import JSON vào PostgreSQL
├── db_schema.sql        ← Database schema
└── README.md            ← File này
```

### Trên VM (Google Cloud) - convert_audio.py:
```
~/convert-work/              ← Thư mục làm việc tạm (xóa sạch sau khi chạy xong)
├── download/                ← Video đang tải (xóa ngay sau khi extract audio)
├── audio/                   ← MP3 tạm (xóa ngay sau khi upload lên Drive)
├── logs/                    ← Log files
│   └── convert_20260210_100000.log
└── convert_checkpoint.json  ← Bản sao local (bản chính lưu trên Drive)
```

### Trên VM (Google Cloud) - transcribe.py:
```
~/transcribe-work/           ← Thư mục làm việc tạm
├── audio/                   ← MP3 tạm tải về (xóa sau khi transcribe + upload)
├── output/                  ← JSON/SRT tạm (xóa sau khi upload)
├── logs/                    ← Log files
│   └── transcribe_20260210_100000.log
└── checkpoint.json          ← Bản sao local (bản chính lưu trên Drive)
```

### Trên Google Drive:
```
Output/                             ← Kết quả
├── Audio/                          ← MP3 files (BƯỚC 1 output)
│   ├── bai_giang_01.mp3
│   └── ...
├── JSON/                           ← Full transcription data (BƯỚC 2 output)
│   ├── bai_giang_01.json
│   └── ...
├── SRT/                            ← Subtitle files (BƯỚC 2 output)
│   ├── bai_giang_01.srt
│   └── ...
├── convert_checkpoint.json         ← Checkpoint BƯỚC 1
└── checkpoint.json                 ← Checkpoint BƯỚC 2
```

---

## 🔄 QUY TRÌNH XỬ LÝ (2-Step Pipeline)

```
┌──────────────────────────────────────────────────────────────────────┐
│                      BƯỚC 1: convert_audio.py                        │
│       (Không cần GPU - Chạy trên bất kỳ VM nào có ffmpeg)           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Với mỗi video:                                                      │
│    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│    │Download  │──▶│Extract   │──▶│Upload    │──▶│Xóa sạch  │       │
│    │MP4 Drive │   │MP4→MP3   │   │MP3 Drive │   │VM files  │       │
│    └──────────┘   │128k/stereo│  └──────────┘   └──────────┘       │
│                   └──────────┘                                       │
│  Input:  Folder video trên Google Drive                              │
│  Output: Folder MP3 trên Google Drive                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      BƯỚC 2: transcribe.py                           │
│              (Cần GPU - Tesla T4 / A100 / v100)                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Thread Prefetcher (CPU/Network):                                    │
│    ┌──────────┐   ┌──────────┐                                      │
│    │Download B│──▶│Download C│──▶ ...                                │
│    │MP3 Drive │   │MP3 Drive │                                       │
│    └──────────┘   └──────────┘                                      │
│                                                                      │
│  Thread Main (GPU):                                                  │
│    ┌────────────┐   ┌──────────┐   ┌────────────┐                   │
│    │Transcribe A│──▶│Upload A  │──▶│Transcribe B│──▶ ...            │
│    │(Whisper)   │   │JSON+SRT  │   │(Whisper)   │                   │
│    └────────────┘   │→Drive    │   └────────────┘                   │
│                     └──────────┘                                     │
│  Input:  Folder MP3 trên Google Drive                                │
│  Output: JSON + SRT trên Google Drive                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Quy trình BƯỚC 1 (convert_audio.py) - Từng video:

```
Step 1: Tải MP4 từ Google Drive về VM
         ↓
Step 2: Extract audio: MP4 → MP3 (ffmpeg, 128kbps, stereo)
         ↓
Step 3: Xóa MP4 trên VM (chỉ giữ MP3)
         ↓
Step 4: Upload MP3 lên Google Drive
         ↓
Step 5: Xóa MP3 trên VM (không giữ lại bất cứ gì)
         ↓
Step 6: Cập nhật checkpoint → Tiếp tục video tiếp theo
```

### Quy trình BƯỚC 2 (transcribe.py) - Từng file MP3:

```
Step 1: [Prefetcher] Tải MP3 từ Google Drive về VM
         ↓
Step 2: [Main] Whisper transcribe MP3 → segments (GPU)
         ↓
Step 3: [Main] Tạo file JSON (full data) + SRT (subtitle)
         ↓
Step 4: [Main] Upload JSON + SRT lên Google Drive
         ↓
Step 5: [Main] Cập nhật checkpoint
         ↓
Step 6: [Main] Xóa MP3 + JSON + SRT trên VM → Tiếp tục
```

---

## 🛡️ CHECKPOINT (An toàn cho Spot Instance)

Mỗi bước có **checkpoint riêng** lưu **TRÊN GOOGLE DRIVE**, không phải local VM.

### BƯỚC 1 checkpoint (`convert_checkpoint.json`):
```json
{
  "videos": {
    "BaiGiang/bai_01.mp4": {
      "status": "done",
      "started_at": "2026-02-10T10:00:00",
      "completed_at": "2026-02-10T10:02:00",
      "processing_time_seconds": 120.5,
      "output_mp3": "Output/Audio/bai_01.mp3"
    }
  },
  "stats": {
    "total_processed": 100,
    "total_failed": 0,
    "last_updated": "2026-02-10T10:02:00"
  }
}
```

### BƯỚC 2 checkpoint (`checkpoint.json`):
```json
{
  "audios": {
    "bai_01.mp3": {
      "status": "done",
      "started_at": "2026-02-10T10:00:00",
      "completed_at": "2026-02-10T10:15:00",
      "processing_time_seconds": 900.0,
      "output_json": "Output/JSON/bai_01.json",
      "output_srt": "Output/SRT/bai_01.srt"
    }
  },
  "stats": {
    "total_processed": 100,
    "total_failed": 0,
    "last_updated": "2026-02-10T10:15:00"
  }
}
```

### Trạng thái video/audio:
- **`processing`**: Đang xử lý (script ghi khi BẮT ĐẦU)
- **`done`**: Đã xong (script ghi khi HOÀN THÀNH)
- **`failed`**: Lỗi (script ghi khi GẶP LỖI)

### Khi VM khởi động lại:
1. Script đọc checkpoint từ Google Drive
2. File có status `done` → **Bỏ qua**
3. File có status `processing` → **Làm lại** (bị crash giữa chừng)
4. File có status `failed` → **Thử lại**
5. File chưa có trong checkpoint → **Xử lý mới**

---

## 🚀 HƯỚNG DẪN CHẠY

### Bước 1: Upload scripts lên VM (từ Windows)

```powershell
cd D:\source-code\video-transcribe
scp convert_audio.py transcribe.py vuhai8617@34.75.41.187:~/
```

### Bước 2: SSH vào VM

```bash
ssh vuhai8617@34.75.41.187

# Cài tmux (để giữ script chạy khi disconnect SSH)
sudo apt install -y tmux
```

### Bước 3: Chạy BƯỚC 1 - Convert Video → MP3

```bash
# Tạo session tmux riêng
tmux new -s convert

# Test với 1 video trước
python3 ~/convert_audio.py \
    --drive-input "TEN_FOLDER_VIDEO" \
    --drive-output "Output/Audio" \
    --test

# Nếu test OK, chạy toàn bộ
python3 ~/convert_audio.py \
    --drive-input "TEN_FOLDER_VIDEO" \
    --drive-output "Output/Audio"

# Hoặc chạy theo batch 50 video
python3 ~/convert_audio.py \
    --drive-input "TEN_FOLDER_VIDEO" \
    --drive-output "Output/Audio" \
    --batch-size 50

# Tùy chỉnh audio settings
python3 ~/convert_audio.py \
    --drive-input "TEN_FOLDER_VIDEO" \
    --drive-output "Output/Audio" \
    --bitrate 192k \
    --channels 1 \
    --sample-rate 48000
```

### Bước 4: Chạy BƯỚC 2 - Transcribe MP3

```bash
# Tạo session tmux riêng
tmux new -s transcribe

# Test với 1 file trước
python3 ~/transcribe.py \
    --drive-input "Output/Audio" \
    --drive-output "Output" \
    --test

# Nếu test OK, chạy toàn bộ
python3 ~/transcribe.py \
    --drive-input "Output/Audio" \
    --drive-output "Output"

# Hoặc chạy theo batch 50 file
python3 ~/transcribe.py \
    --drive-input "Output/Audio" \
    --drive-output "Output" \
    --batch-size 50
```

### Bước 5: Thoát tmux (giữ script chạy)

```
Ctrl+B rồi D    ← Thoát tmux, script vẫn chạy
tmux attach -t convert      ← Quay lại xem BƯỚC 1
tmux attach -t transcribe   ← Quay lại xem BƯỚC 2
```

---

## ⏱️ ƯỚC TÍNH THỜI GIAN & CHI PHÍ

### BƯỚC 1 (Convert Audio):
| Thông số | Giá trị |
|---|---|
| Số video | ~1500 |
| Thời gian convert | ~1-3 phút/video |
| Tổng thời gian | ~25-75 giờ (~1-3 ngày) |
| Không cần GPU | Chạy trên VM CPU được |

### BƯỚC 2 (Transcribe):
| Thông số | Giá trị |
|---|---|
| Số MP3 | ~1500 |
| Độ dài TB | ~1 giờ/file |
| GPU | Tesla T4 (15GB VRAM) |
| Model | large-v3 |
| Thời gian transcribe | ~3-8 phút/file (faster-whisper + VAD) |
| Tổng thời gian | ~75-200 giờ (~3-8 ngày) |
| Chi phí Spot VM | ~$0.35/giờ |
| Tổng chi phí | ~$26-70 (trong $300 credit) |

---

## 📊 FORMAT OUTPUT

### JSON (cho Database):
```json
{
  "video_name": "Bai Giang 01.mp3",
  "drive_path": "bai_giang_01.mp3",
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

### BƯỚC 1 - Convert Audio Service:

```bash
sudo tee /etc/systemd/system/convert-audio.service << 'EOF'
[Unit]
Description=Video to MP3 Converter Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=vuhai8617
WorkingDirectory=/home/vuhai8617
ExecStart=/usr/bin/python3 /home/vuhai8617/convert_audio.py --drive-input "TEN_FOLDER_VIDEO" --drive-output "Output/Audio"
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable convert-audio.service
sudo systemctl start convert-audio.service

# Xem log
sudo journalctl -u convert-audio.service -f
```

### BƯỚC 2 - Transcribe Service:

```bash
sudo tee /etc/systemd/system/transcribe.service << 'EOF'
[Unit]
Description=Audio Transcription Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=vuhai8617
WorkingDirectory=/home/vuhai8617
ExecStart=/usr/bin/python3 /home/vuhai8617/transcribe.py --drive-input "Output/Audio" --drive-output "Output"
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

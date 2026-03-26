# 🚀 Hướng Dẫn Cấu Hình Vertex AI Cho Batch Transcribe

Hướng dẫn này giúp bạn chuyển từ **AI Studio (API Key)** sang **Vertex AI (Service Account)** để sử dụng **$300 Google Cloud Free Trial Credit**.

> **⚠️ LƯU Ý QUAN TRỌNG VỀ MODEL:**
> Tại thời điểm viết (03/2026), các model **Gemini 3.x preview** (gemini-3.1-pro-preview, gemini-3-pro-preview...) tuy hiển thị trong danh sách model nhưng **KHÔNG hoạt động qua API** trên project Free Trial (lỗi 404). Chỉ các model sau đã xác nhận hoạt động:
> - ✅ `gemini-2.5-flash` — **Recommended** (nhanh, rẻ, chất lượng tốt)
> - ✅ `gemini-2.5-pro` — Chất lượng cao hơn, chậm hơn, dễ bị rate limit
> - ✅ `gemini-2.5-flash-lite` — Rẻ nhất, nhanh nhất

---

## Mục lục

1. [Enable APIs](#bước-1-enable-apis-trên-google-cloud-console)
2. [Tạo Service Account](#bước-2-tạo-service-account)
3. [Tạo JSON Key](#bước-3-tạo-và-tải-json-key)
4. [Tạo GCS Bucket](#bước-4-tạo-google-cloud-storage-bucket)
5. [Cấu hình .env](#bước-5-cấu-hình-file-env)
6. [Cài đặt Dependencies](#bước-6-cài-đặt-dependencies)
7. [Chạy Script](#bước-7-chạy-script)
8. [So sánh AI Studio vs Vertex AI](#tóm-tắt-thay-đổi-so-với-bản-cũ)
9. [Troubleshooting](#troubleshooting)

---

## Bước 1: Enable APIs trên Google Cloud Console

Truy cập [Google Cloud Console](https://console.cloud.google.com) → chọn project của bạn.

### 1a. Enable Vertex AI API
1. Vào [APIs & Services → Library](https://console.cloud.google.com/apis/library)
2. Tìm kiếm **"Vertex AI API"**
3. Click **Enable**

### 1b. Enable Cloud Storage API
1. Tìm kiếm **"Cloud Storage API"** (hoặc **"Google Cloud Storage JSON API"**)
2. Click **Enable** (có thể đã enable sẵn)

---

## Bước 2: Tạo Service Account

1. Vào [IAM & Admin → Service Accounts](https://console.cloud.google.com/iam-admin/service-accounts)
2. Click **"+ CREATE SERVICE ACCOUNT"** (phía trên)
3. Điền thông tin:
   - **Service account name:** `transcribe-worker`
   - **Service account ID:** tự tạo (ví dụ: `transcribe-worker`)
   - **Description:** `Service account cho batch transcribe`
4. Click **"CREATE AND CONTINUE"**
5. **Gán roles** (quan trọng!):
   - Click **"+ ADD ANOTHER ROLE"** để thêm từng role:
     - **`Vertex AI User`** → cho phép gọi Gemini qua Vertex AI
     - **`Storage Object Admin`** → cho phép upload/xóa file trên GCS
   - Click **"CONTINUE"**
6. Click **"DONE"**

> **📝 Ghi chú:** Nếu muốn test nhanh, có thể gán role **Owner** cho service account. Tuy nhiên, trong production nên dùng quyền tối thiểu (Vertex AI User + Storage Object Admin).

---

## Bước 3: Tạo và Tải JSON Key

1. Trong danh sách Service Accounts, click vào **transcribe-worker** (vừa tạo)
2. Chọn tab **"KEYS"**
3. Click **"ADD KEY"** → **"Create new key"**
4. Chọn **JSON** → Click **"CREATE"**
5. File JSON sẽ tự động tải về máy (ví dụ: `transcribe-491102-xxxx.json`)

### Upload file key lên VPS

```bash
# Từ máy local, dùng scp để upload lên VPS:
scp ~/Downloads/transcribe-491102-xxxx.json vuhai@<VPS_IP>:/home/vuhai/video-transcribe/service-account-key.json

# Hoặc nếu đang trên VPS, copy nội dung file JSON và paste:
nano /home/vuhai/video-transcribe/service-account-key.json
# Paste nội dung JSON vào → Ctrl+O → Enter → Ctrl+X
```

> **🔒 BẢO MẬT:** Tuyệt đối **KHÔNG** commit file `service-account-key.json` lên Git!
> ```bash
> echo "service-account-key.json" >> /home/vuhai/video-transcribe/.gitignore
> ```

---

## Bước 4: Tạo Google Cloud Storage Bucket

Bucket dùng để upload tạm audio files trước khi gửi cho Gemini xử lý. Script sẽ **tự động xóa** file trên GCS sau khi transcribe xong.

### Cách 1: Qua Console (UI)
1. Vào [Cloud Storage → Buckets](https://console.cloud.google.com/storage/browser)
2. Click **"+ CREATE"**
3. Điền:
   - **Bucket name:** `transcribe-audio-<PROJECT_ID>` (tên phải unique toàn cầu)
   - **Location type:** Region
   - **Region:** `us-central1` (cùng region với Vertex AI)
   - **Storage class:** Standard
   - **Access control:** Uniform (mặc định)
4. Click **"CREATE"**

### Cách 2: Qua gcloud CLI
```bash
gcloud storage buckets create gs://transcribe-audio-<PROJECT_ID> \
    --project=<PROJECT_ID> \
    --location=us-central1 \
    --uniform-bucket-level-access
```

---

## Bước 5: Cấu hình file `.env`

Mở file `.env` trong thư mục project và thêm/sửa các dòng sau:

```env
# Google Gemini API Key (giữ lại cho script cũ nếu cần)
GOOGLE_API_KEY=your_api_key_here

# Model sử dụng cho transcription
# Recommended: gemini-2.5-flash (nhanh, rẻ) hoặc gemini-2.5-pro (chất lượng cao)
GEMINI_MODEL=gemini-2.5-flash

# === VERTEX AI CONFIG ===
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
GCS_BUCKET_NAME=your_bucket_name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Giải thích các biến:

| Biến | Mô tả | Ví dụ |
|---|---|---|
| `GEMINI_MODEL` | Model Gemini sử dụng | `gemini-2.5-flash` |
| `GOOGLE_CLOUD_PROJECT` | Project ID trên GCP | `transcribe-491102` |
| `GOOGLE_CLOUD_LOCATION` | Region cho Vertex AI | `us-central1` |
| `GCS_BUCKET_NAME` | Tên bucket GCS | `transcribe-audio-491102` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Đường dẫn file JSON key | `/home/vuhai/video-transcribe/service-account-key.json` |

---

## Bước 6: Cài đặt Dependencies

```bash
cd /home/vuhai/video-transcribe
pip install google-genai google-cloud-storage google-auth python-dotenv
```

### Danh sách packages cần thiết:

| Package | Mục đích |
|---|---|
| `google-genai` | SDK gọi Gemini qua Vertex AI |
| `google-cloud-storage` | Upload/xóa file trên GCS |
| `google-auth` | Xác thực với Service Account |
| `python-dotenv` | Đọc cấu hình từ file `.env` |

---

## Bước 7: Chạy Script

```bash
cd /home/vuhai/video-transcribe

# Chạy với cấu hình mặc định (5 workers)
python3 batch_transcribe_vertex.py

# Tùy chỉnh số workers (giảm nếu bị rate limit nhiều)
python3 batch_transcribe_vertex.py --workers 3

# Chỉ định thư mục input/output
python3 batch_transcribe_vertex.py --mp3-dir /path/to/mp3 --txt-dir /path/to/txt

# Chạy lại tất cả (kể cả file đã có TXT)
python3 batch_transcribe_vertex.py --force
```

### Lưu ý khi chạy:
- Script tự động **bỏ qua** file đã có TXT (resume-friendly)
- Log file được tạo tự động tại thư mục project: `batch_transcribe_vertex_YYYYMMDD_HHMMSS.log`
- Nếu gặp **rate limit** (429), script tự động retry với thời gian chờ tăng dần
- Nếu gặp **499 CANCELLED** hoặc **503 UNAVAILABLE**, script tự retry

### Khuyến nghị chọn model:

| Model | Tốc độ | Chi phí | Rate Limit | Phù hợp cho |
|---|---|---|---|---|
| `gemini-2.5-flash` | ⚡ Nhanh (~60s/file) | 💰 Rẻ | Ít bị | Batch lớn, tiết kiệm credit |
| `gemini-2.5-pro` | 🐢 Chậm (~80-120s/file) | 💰💰 Đắt hơn | Hay bị | Cần chất lượng cao |

---

## Tóm tắt thay đổi so với bản cũ

| | Bản cũ (AI Studio) | Bản mới (Vertex AI) |
|---|---|---|
| **Script** | `batch_transcribe.py` | `batch_transcribe_vertex.py` |
| **Xác thực** | API Key (`GOOGLE_API_KEY`) | Service Account JSON key |
| **Upload audio** | Gemini File API (`client.files.upload`) | Google Cloud Storage (GCS) |
| **Endpoint** | AI Studio (`generativelanguage.googleapis.com`) | Vertex AI (`aiplatform.googleapis.com`) |
| **Thanh toán** | AI Studio billing (không dùng $300 credit) | Google Cloud billing (dùng $300 credit ✅) |
| **File tạm** | Gemini cloud (auto xóa sau 48h) | GCS bucket (script tự xóa ngay sau khi xong) |

---

## Troubleshooting

### ❌ "Permission denied" khi upload GCS
→ Kiểm tra Service Account có role **Storage Object Admin** chưa.
Vào [IAM & Admin → IAM](https://console.cloud.google.com/iam-admin/iam) → tìm service account → thêm role.

### ❌ "Vertex AI API has not been enabled"
→ Quay lại [Bước 1a](#1a-enable-vertex-ai-api), enable Vertex AI API.

### ❌ "Could not automatically determine credentials"
→ Kiểm tra `GOOGLE_APPLICATION_CREDENTIALS` trong `.env` trỏ đúng path đến file JSON key.

### ❌ "Bucket not found"
→ Kiểm tra `GCS_BUCKET_NAME` trong `.env` đúng tên bucket đã tạo.

### ❌ "404 NOT_FOUND" cho model
→ Model không khả dụng trên project của bạn. Đổi `GEMINI_MODEL` sang `gemini-2.5-flash`.

Kiểm tra model nào hoạt động:
```bash
# Script test model
python3 -c "
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2 import service_account

load_dotenv('.env')
creds = service_account.Credentials.from_service_account_file(
    os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
client = genai.Client(vertexai=True, project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                      location='us-central1', credentials=creds)

for m in ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite']:
    try:
        r = client.models.generate_content(model=m, contents='Xin chào',
            config=types.GenerateContentConfig(max_output_tokens=20))
        print(f'✅ {m}: OK')
    except Exception as e:
        print(f'❌ {m}: {str(e)[:60]}')
"
```

### ❌ "429 RESOURCE_EXHAUSTED" (Rate Limit)
→ Đây là lỗi bình thường khi chạy nhiều workers. Script tự retry. Nếu xảy ra quá thường xuyên, giảm số workers:
```bash
python3 batch_transcribe_vertex.py --workers 3
```

### ❌ "499 CANCELLED"
→ Server timeout khi xử lý file lớn. Script tự retry. Nếu vẫn fail, thử chạy lại.

### ❌ "FAILED_PRECONDITION" (lần chạy đầu tiên)
→ Google đang setup Vertex AI service agents cho project. Chờ 2-3 phút rồi chạy lại.

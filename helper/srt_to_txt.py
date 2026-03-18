"""
Script chuyển đổi file phụ đề .srt sang .txt (plain text).
Logic:
- Loại bỏ số thứ tự phân đoạn (segment ID).
- Loại bỏ dòng thời gian (timestamp).
- Giữ lại nội dung text.
- Mỗi block subtitle sẽ được chuyển thành 1 dòng text trong file output.

Sử dụng:
    python srt_to_txt.py "input_file.srt"
    python srt_to_txt.py "input_file.srt" -o "output_file.txt"
"""

import sys
import re
import argparse
from pathlib import Path

def clean_srt_text(text_lines):
    """
    Xử lý danh sách các dòng text của một segment:
    - Loại bỏ các thẻ HTML (nếu có).
    - Nối các dòng lại thành một câu hoàn chỉnh (nếu segment bị ngắt dòng).
    """
    # Gộp các dòng lại, cách nhau bằng khoảng trắng
    text = " ".join(line.strip() for line in text_lines if line.strip())
    # Loại bỏ tags HTML (ví dụ <i>, <b>, <font>)
    text = re.sub(r'<[^>]+>', '', text)
    return text

def convert_srt_to_txt(input_path, output_path=None):
    if not output_path:
        output_path = str(Path(input_path).with_suffix('.txt'))

    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Lỗi: Không tìm thấy file '{input_path}'")
        return

    try:
        # Đọc file với encoding utf-8 (có thể handle BOM)
        content = input_file.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        # Fallback nếu không phải utf-8
        content = input_file.read_text(encoding='latin-1')

    # Chuẩn hóa xuống dòng để xử lý dễ hơn
    content = content.replace('\r\n', '\n')
    
    # Tách file thành các blocks dựa trên 2 dấu xuống dòng liên tiếp
    # Regex \n\n+ dùng để bắt trường hợp có nhiều hơn 2 dòng trống
    blocks = re.split(r'\n\n+', content.strip())
    
    output_lines = []
    
    for block in blocks:
        lines = block.split('\n')
        # Block chuẩn thường có ít nhất 3 phần: ID, Timestamp, Text
        # Nhưng ta sẽ tìm dòng Timestamp để làm mốc chắc chắn
        
        timestamp_index = -1
        for i, line in enumerate(lines):
            if '-->' in line:
                timestamp_index = i
                break
        
        if timestamp_index != -1:
            # Text là phần sau dòng timestamp
            text_part = lines[timestamp_index + 1:]
            clean_text = clean_srt_text(text_part)
            
            if clean_text:
                output_lines.append(clean_text)

    # Ghi ra file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"✅ Đã chuyển đổi thành công!")
        print(f"   Input:  {input_path}")
        print(f"   Output: {output_path}")
        print(f"   Số dòng: {len(output_lines)}")
    except Exception as e:
        print(f"Lỗi khi ghi file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SRT to TXT")
    parser.add_argument("input_file", help="Đường dẫn file .srt")
    parser.add_argument("-o", "--output", help="Đường dẫn file .txt đầu ra (tùy chọn)")
    
    args = parser.parse_args()
    convert_srt_to_txt(args.input_file, args.output)

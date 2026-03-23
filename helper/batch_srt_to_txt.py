"""
Script batch chuyển đổi tất cả file .srt trong một folder sang .txt.
Sử dụng logic từ srt_to_txt.py.

Sử dụng:
    python batch_srt_to_txt.py
"""

import re
from pathlib import Path

INPUT_DIR = Path(r"D:\source-code\video-transcribe\data\srt")
OUTPUT_DIR = Path(r"D:\source-code\video-transcribe\data\txt")


def clean_srt_text(text_lines):
    text = " ".join(line.strip() for line in text_lines if line.strip())
    text = re.sub(r'<[^>]+>', '', text)
    return text


def convert_srt_to_txt(input_path, output_path):
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"  ❌ Không tìm thấy file '{input_path}'")
        return False

    try:
        content = input_file.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        content = input_file.read_text(encoding='latin-1')

    content = content.replace('\r\n', '\n')
    blocks = re.split(r'\n\n+', content.strip())

    output_lines = []

    for block in blocks:
        lines = block.split('\n')

        timestamp_index = -1
        for i, line in enumerate(lines):
            if '-->' in line:
                timestamp_index = i
                break

        if timestamp_index != -1:
            text_part = lines[timestamp_index + 1:]
            clean_text = clean_srt_text(text_part)

            if clean_text:
                output_lines.append(clean_text)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        return True
    except Exception as e:
        print(f"  ❌ Lỗi khi ghi file: {e}")
        return False


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    srt_files = sorted(INPUT_DIR.glob("*.srt"))
    total = len(srt_files)
    success = 0
    failed = 0

    print(f"📂 Tìm thấy {total} file SRT trong: {INPUT_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("-" * 60)

    for idx, srt_file in enumerate(srt_files, 1):
        txt_file = OUTPUT_DIR / srt_file.with_suffix('.txt').name
        result = convert_srt_to_txt(srt_file, txt_file)
        if result:
            success += 1
            print(f"  ✅ [{idx}/{total}] {srt_file.name}")
        else:
            failed += 1
            print(f"  ❌ [{idx}/{total}] {srt_file.name}")

    print("-" * 60)
    print(f"🎉 Hoàn tất! Thành công: {success}/{total}, Thất bại: {failed}")

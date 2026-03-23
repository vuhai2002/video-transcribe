#!/usr/bin/env python3
"""
Verify SRT - Kiểm tra nội dung text trong SRT có khớp 100% với TXT
So sánh 2 cấp độ:
  1. Nội dung tổng thể (gộp tất cả text lại) - QUAN TRỌNG NHẤT
  2. Cấu trúc dòng (số dòng, thứ tự) - ít quan trọng hơn

WhisperX alignment có thể tách 1 dòng TXT thành 2+ subtitle SRT,
điều này KHÔNG phải lỗi mất text.

Usage:
    python3 verify_srt.py
    python3 verify_srt.py --detail          # Hiển thị chi tiết
    python3 verify_srt.py --strict          # Bắt buộc khớp từng dòng
"""

import os
import sys
import argparse
from difflib import unified_diff

# ============================================================
# CẤU HÌNH
# ============================================================
DEFAULT_TXT_DIR = "/home/vuhai/data/txt"
DEFAULT_SRT_DIR = "/home/vuhai/data/srt"


def parse_srt(srt_path):
    """Parse file SRT, trả về list các text line (bỏ index và timestamp)."""
    lines = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    for block in blocks:
        block_lines = block.strip().split("\n")
        if len(block_lines) >= 3:
            text = "\n".join(block_lines[2:]).strip()
            if text:
                lines.append(text)
    return lines


def load_txt(txt_path):
    """Load file TXT, trả về list các dòng (bỏ dòng trống)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def compare_texts(txt_lines, srt_lines, name, show_detail=False, strict=False):
    """
    So sánh text giữa TXT và SRT.
    
    Level 1 (Critical): Nội dung gộp lại có khớp không? (mất text = lỗi nghiêm trọng)
    Level 2 (Info): Số dòng có khớp không? (tách dòng = bình thường)
    """
    result = {
        "name": name,
        "txt_lines": len(txt_lines),
        "srt_lines": len(srt_lines),
        "content_match": True,   # Gộp text lại có khớp?
        "line_match": True,      # Từng dòng có khớp?
        "issues": [],
        "level": "OK",           # OK, INFO, WARNING, ERROR
    }

    # ==== LEVEL 1: So sánh nội dung tổng thể (QUAN TRỌNG NHẤT) ====
    txt_full = " ".join(txt_lines)
    srt_full = " ".join(srt_lines)
    
    txt_words = txt_full.split()
    srt_words = srt_full.split()
    
    result["txt_words"] = len(txt_words)
    result["srt_words"] = len(srt_words)

    if txt_full == srt_full:
        result["content_match"] = True
    else:
        # Kiểm tra kỹ hơn: so sánh từng từ
        missing_words = []
        extra_words = []
        
        # Dùng diff trên danh sách từ
        txt_word_set = txt_full
        srt_word_set = srt_full
        
        if txt_word_set != srt_word_set:
            result["content_match"] = False
            result["level"] = "ERROR"
            
            # Tìm vị trí khác đầu tiên
            min_len = min(len(txt_full), len(srt_full))
            first_diff_pos = None
            for i in range(min_len):
                if txt_full[i] != srt_full[i]:
                    first_diff_pos = i
                    break
            
            if first_diff_pos is not None:
                ctx_start = max(0, first_diff_pos - 30)
                ctx_end = min(min_len, first_diff_pos + 30)
                result["issues"].append(
                    f"🔴 NỘI DUNG KHÁC NHAU! Vị trí ký tự ~{first_diff_pos}"
                )
                if show_detail:
                    result["issues"].append(
                        f"   TXT: ...{txt_full[ctx_start:ctx_end]}..."
                    )
                    result["issues"].append(
                        f"   SRT: ...{srt_full[ctx_start:ctx_end]}..."
                    )
            elif len(txt_full) != len(srt_full):
                diff = len(srt_full) - len(txt_full)
                if diff > 0:
                    result["issues"].append(
                        f"🔴 SRT có thêm {diff} ký tự ở cuối"
                    )
                else:
                    result["issues"].append(
                        f"🔴 SRT thiếu {abs(diff)} ký tự ở cuối"
                    )

            # Đếm từ khác
            word_diff = len(srt_words) - len(txt_words)
            if word_diff != 0:
                result["issues"].append(
                    f"   Chênh lệch: {abs(word_diff)} từ ({'thừa' if word_diff > 0 else 'thiếu'} trong SRT)"
                )

    # ==== LEVEL 2: So sánh cấu trúc dòng (tách dòng) ====
    if len(txt_lines) != len(srt_lines):
        result["line_match"] = False
        diff_lines = len(srt_lines) - len(txt_lines)
        
        if result["content_match"]:
            # Nội dung khớp nhưng số dòng khác = chỉ bị tách dòng
            result["level"] = max(result["level"], "INFO") if result["level"] != "ERROR" else "ERROR"
            result["issues"].append(
                f"🟡 Tách dòng: TXT={len(txt_lines)} → SRT={len(srt_lines)} (chênh {diff_lines:+d} dòng, text vẫn khớp 100%)"
            )
        else:
            result["issues"].append(
                f"Khác số dòng: TXT={len(txt_lines)}, SRT={len(srt_lines)}"
            )

    # Strict mode: kiểm tra từng dòng
    if strict:
        min_len = min(len(txt_lines), len(srt_lines))
        diff_count = 0
        for i in range(min_len):
            if txt_lines[i] != srt_lines[i]:
                diff_count += 1
        if diff_count > 0:
            result["line_match"] = False
            result["issues"].append(f"{diff_count} dòng khác nhau (strict mode)")

    return result


def run_verify(txt_dir, srt_dir, show_detail=False, strict=False):
    """Chạy verify toàn bộ."""

    txt_files = {os.path.splitext(f)[0]: f for f in os.listdir(txt_dir) if f.endswith('.txt')}
    srt_files = {os.path.splitext(f)[0]: f for f in os.listdir(srt_dir) if f.endswith('.srt')}

    common_names = sorted(set(txt_files.keys()) & set(srt_files.keys()))
    missing_srt = sorted(set(txt_files.keys()) - set(srt_files.keys()))

    print("=" * 70)
    print("🔍 VERIFY SRT vs TXT - Kiểm tra nội dung text")
    print(f"   TXT dir: {txt_dir}")
    print(f"   SRT dir: {srt_dir}")
    print(f"   Tổng cặp file: {len(common_names)}")
    print(f"   Mode: {'STRICT (từng dòng phải khớp)' if strict else 'NORMAL (chỉ check nội dung tổng thể)'}")
    if missing_srt:
        print(f"   ⚠️  {len(missing_srt)} file TXT chưa có SRT")
    print("=" * 70)
    print()

    perfect_count = 0           # Khớp hoàn toàn (cả nội dung lẫn dòng)
    content_ok_count = 0        # Nội dung khớp (có thể tách dòng)
    content_error_count = 0     # Nội dung bị mất/thêm
    error_files = []
    split_files = []

    for idx, name in enumerate(common_names, 1):
        txt_path = os.path.join(txt_dir, txt_files[name])
        srt_path = os.path.join(srt_dir, srt_files[name])

        txt_lines = load_txt(txt_path)
        srt_lines = parse_srt(srt_path)

        result = compare_texts(txt_lines, srt_lines, name, show_detail, strict)

        if result["content_match"] and result["line_match"]:
            perfect_count += 1
            print(f"  ✅ [{idx}/{len(common_names)}] {name} ({result['txt_lines']} dòng)")
        elif result["content_match"] and not result["line_match"]:
            content_ok_count += 1
            split_files.append(result)
            print(f"  🟡 [{idx}/{len(common_names)}] {name}")
            for issue in result["issues"]:
                print(f"     {issue}")
        else:
            content_error_count += 1
            error_files.append(result)
            print(f"  ❌ [{idx}/{len(common_names)}] {name}")
            for issue in result["issues"]:
                print(f"     {issue}")

    # Tổng kết
    print()
    print("=" * 70)
    print("📊 TỔNG KẾT")
    print(f"   ✅ Khớp hoàn toàn (nội dung + số dòng): {perfect_count}/{len(common_names)}")
    print(f"   🟡 Nội dung khớp 100% nhưng bị tách dòng: {content_ok_count}/{len(common_names)}")
    print(f"   ❌ NỘI DUNG BỊ MẤT/THÊM TEXT:             {content_error_count}/{len(common_names)}")
    print()
    
    total_ok = perfect_count + content_ok_count
    print(f"   📝 Tổng file text khớp 100%: {total_ok}/{len(common_names)} ({total_ok/len(common_names)*100:.1f}%)")
    
    if content_error_count > 0:
        print()
        print("🔴 DANH SÁCH FILE BỊ MẤT/THÊM TEXT (cần kiểm tra lại):")
        for r in error_files:
            print(f"   - {r['name']}")
            print(f"     TXT: {r['txt_words']} từ | SRT: {r['srt_words']} từ")
            for issue in r["issues"]:
                print(f"     {issue}")

    if show_detail and split_files:
        print()
        print(f"🟡 DANH SÁCH FILE BỊ TÁCH DÒNG ({len(split_files)} file, text vẫn OK):")
        for r in split_files:
            diff = r["srt_lines"] - r["txt_lines"]
            print(f"   - {r['name']}: +{diff} dòng (TXT={r['txt_lines']} → SRT={r['srt_lines']})")

    if missing_srt:
        print()
        print(f"📋 {len(missing_srt)} FILE TXT CHƯA CÓ SRT:")
        for name in missing_srt[:10]:
            print(f"   - {name}")
        if len(missing_srt) > 10:
            print(f"   ... và {len(missing_srt) - 10} file khác")

    print("=" * 70)

    return 0 if content_error_count == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify SRT vs TXT - Kiểm tra text khớp 100%")
    parser.add_argument("--txt-dir", default=DEFAULT_TXT_DIR, help="Thư mục chứa file TXT")
    parser.add_argument("--srt-dir", default=DEFAULT_SRT_DIR, help="Thư mục chứa file SRT")
    parser.add_argument("--detail", action="store_true", help="Hiển thị chi tiết")
    parser.add_argument("--strict", action="store_true", help="Kiểm tra từng dòng phải khớp (không chỉ nội dung)")

    args = parser.parse_args()
    exit_code = run_verify(args.txt_dir, args.srt_dir, show_detail=args.detail, strict=args.strict)
    sys.exit(exit_code)

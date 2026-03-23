#!/bin/bash
# Script: Xóa các file trong "1-audio" mà đã tồn tại ở "run-script-3"
# Dùng rclone để lấy danh sách file từ run-script-3, rồi xóa file trùng trong 1-audio

REMOTE="ggdrive:"
SRC_FOLDER="1-audio"
REF_FOLDER="run-script-3"

echo "=== Lấy danh sách file từ '${REF_FOLDER}'..."
mapfile -t files < <(rclone lsf "${REMOTE}${REF_FOLDER}/" --files-only)

total=${#files[@]}
echo "=== Tìm thấy ${total} file trong '${REF_FOLDER}'"
echo ""

deleted=0
errors=0

for i in "${!files[@]}"; do
    file="${files[$i]}"
    idx=$((i + 1))
    echo "[${idx}/${total}] Đang xóa: ${file}"
    
    if rclone deletefile "${REMOTE}${SRC_FOLDER}/${file}" 2>/dev/null; then
        ((deleted++))
    else
        echo "  ⚠ Lỗi hoặc không tìm thấy: ${file}"
        ((errors++))
    fi
done

echo ""
echo "=== Hoàn tất ==="
echo "Đã xóa: ${deleted} file"
echo "Lỗi/Không tìm thấy: ${errors} file"

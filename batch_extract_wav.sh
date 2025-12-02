#!/bin/bash

ROOT_DIR="raw_IEMOCAP"

echo "开始批量提取 AVI → WAV ..."

# 遍历所有 AVI 文件
find "$ROOT_DIR" -type f -name "*.avi" | while read -r avi_file; do
    wav_file="${avi_file%.avi}.wav"

    # 如果 wav 已存在，跳过
    if [ -f "$wav_file" ]; then
        echo "[跳过] 已存在: $wav_file"
        continue
    fi

    echo "[处理] $avi_file → $wav_file"

    ffmpeg -y -i "$avi_file" -ac 1 -ar 16000 "$wav_file"
done

echo "全部 AVI 文件已成功转换为 WAV！"
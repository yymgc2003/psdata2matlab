#!/bin/bash

# 入力と出力のディレクトリを設定
INPUT_DIR="W:\2023\Ultrasound\012"  # サーバーの.psdataファイルのパス
OUTPUT_DIR="Z:\tmp"                 # 出力先のパス

# 変換対象のファイル数をカウント（cmd.exeを使用）
total_files=$(cmd.exe /c "dir /b ${INPUT_DIR}\*.psdata 2>nul | find /c /v """")
echo "合計${total_files}個のファイルを変換します..."

# カウンター初期化
current=0

# .psdataファイルを処理（cmd.exeを使用してファイル一覧を取得）
while IFS= read -r file; do
    # 空行をスキップ
    [ -z "$file" ] && continue
    
    # 進捗状況の表示
    ((current++))
    filename=$(basename "$file")
    echo "[${current}/${total_files}] 変換中: $filename"
    
    # 出力ファイル名の生成（.psdataを.matに置換）
    output_file="${OUTPUT_DIR}/${filename%.*}.mat"
    
    # 変換実行
    if picoscope /c "$file" /f mat /d "$output_file" /q /b 1; then
        echo "✓ 変換成功: $filename"
    else
        echo "✗ 変換失敗: $filename"
    fi
done < <(cmd.exe /c "dir /b ${INPUT_DIR}\*.psdata 2>nul")

echo "変換完了" 
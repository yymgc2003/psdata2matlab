# 入力と出力のディレクトリを設定
$TARGET_DIR = "W:\2023\Ultrasound\012"  # 作業対象のディレクトリ
$OUTPUT_DIR = "Z:\tmp"                   # 出力先のパス
$TIMEOUT_SECONDS = 30000                  # タイムアウト時間（秒）

# picoscopeのパスを確認
$PICOSCOPE_PATHS = @(
    "C:\Program Files\Pico Technology\SDK\bin",
    "C:\Program Files (x86)\Pico Technology\SDK\bin",
    "C:\Program Files\Pico Technology\PicoScope6",
    "C:\Program Files (x86)\Pico Technology\PicoScope6"
)

# 現在のPATHを保存
$original_path = $env:Path

# PicoScopeのパスを環境変数に追加
foreach ($path in $PICOSCOPE_PATHS) {
    if (Test-Path $path) {
        $env:Path = "$path;$env:Path"
        Write-Output "Added to PATH: $path"
    }
}

# 現在のディレクトリを保存
$original_location = Get-Location

Write-Output "Current directory: $original_location"
Write-Output "Target directory: $TARGET_DIR"
Write-Output "Output directory: $OUTPUT_DIR"
Write-Output "Timeout: $TIMEOUT_SECONDS seconds"

# 出力ディレクトリの確認と作成
Write-Output "`nChecking output directory..."
if (!(Test-Path -Path $OUTPUT_DIR)) {
    Write-Output "Creating output directory: $OUTPUT_DIR"
    try {
        New-Item -ItemType Directory -Path $OUTPUT_DIR -Force
        Write-Output "Output directory created successfully"
    } catch {
        Write-Output "Error creating output directory: $_"
        $env:Path = $original_path
        exit 1
    }
} else {
    Write-Output "Output directory exists: $OUTPUT_DIR"
    # 書き込み権限の確認
    try {
        $testFile = Join-Path $OUTPUT_DIR "test.tmp"
        [IO.File]::WriteAllText($testFile, "test")
        Remove-Item $testFile
        Write-Output "Output directory is writable"
    } catch {
        Write-Output "Warning: Cannot write to output directory: $_"
    }
}

# 対象ディレクトリに移動
Write-Output "`nChanging to target directory..."
try {
    Set-Location -Path $TARGET_DIR
    Write-Output "Successfully changed to: $(Get-Location)"
} catch {
    Write-Output "Error: Could not change to target directory: $_"
    Set-Location -Path $original_location
    $env:Path = $original_path
    exit 1
}

# .psdataファイルを取得
$psdata_files = Get-ChildItem -Filter "*.psdata"
$total_files = $psdata_files.Count

Write-Output "`nFound $total_files .psdata files in current directory"

# カウンター初期化
$current = 0

# 各ファイルを処理
foreach ($file in $psdata_files) {
    # カウンターを増やす
    $current++
    
    # 進捗状況の表示
    Write-Output "`n[$current/$total_files] Converting: $($file.Name)"
    $output_file = Join-Path $OUTPUT_DIR ($file.BaseName + ".mat")
    Write-Output "Output file: $output_file"
    
    # 変換実行（タイムアウト付き）
    try {
        # コマンドをCMDを通して実行
        $cmd = "cmd.exe /c `"picoscope /c $($file.Name) /f mat /d $output_file /q /b 1`""
        Write-Output "Executing: $cmd"
        
        # ジョブを開始
        $job = Start-Job -ScriptBlock {
            param($cmd)
            $result = Invoke-Expression $cmd
            $LASTEXITCODE
        } -ArgumentList $cmd
        
        # タイムアウトまで待機
        $completed = Wait-Job -Job $job -Timeout $TIMEOUT_SECONDS
        
        if ($completed -eq $null) {
            # タイムアウト発生
            Write-Output "Timeout: Operation took longer than $TIMEOUT_SECONDS seconds"
            Stop-Job -Job $job
            Remove-Job -Job $job -Force
            Write-Output "Skipping: $($file.Name)"
            continue
        }
        
        # ジョブの結果を取得
        $exitCode = Receive-Job -Job $job
        Remove-Job -Job $job
        
        if ($exitCode -eq 0) {
            Write-Output "Success: $($file.Name)"
            if (Test-Path $output_file) {
                $fileInfo = Get-Item $output_file
                Write-Output "Created file size: $($fileInfo.Length) bytes"
            } else {
                Write-Output "Warning: Output file not found despite successful return code"
            }
        } else {
            Write-Output "Failed: $($file.Name) with exit code $exitCode"
            Write-Output "Try running the command manually in cmd.exe to see detailed error messages"
        }
    } catch {
        Write-Output "Error: $($file.Name) - $($_.Exception.Message)"
        Write-Output "Exception details: $_"
    }
}

# 元のディレクトリに戻る
Write-Output "`nReturning to original directory..."
Set-Location -Path $original_location
$env:Path = $original_path
Write-Output "Current directory: $(Get-Location)"

Write-Output "`nConversion completed" 
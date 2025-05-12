# PsData to MATLAB変換ツール

このツールは、.psdataファイルを.matファイルに一括変換するためのPowerShellスクリプトです。

## 必要条件

- Windows PowerShell 5.1以上
- picoscopeコマンドラインツール
Picoscope7を使え（遺言）

## セットアップ

1. PowerShellの実行ポリシーを設定（初回のみ）:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

2. スクリプト内の設定を変更:
- `$INPUT_DIR`: .psdataファイルが保存されているサーバーのパス
- `$OUTPUT_DIR`: 変換後の.matファイルを保存するディレクトリのパス

## 使用方法

PowerShellを開いて以下のコマンドを実行：
```powershell
.\convert_psdata_to_mat.ps1
```

## 機能

- 指定されたディレクトリ内の全ての.psdataファイルを検索
- 各ファイルを.matファイルに変換
- 進捗状況のカラー表示
- エラーハンドリング

## 注意事項

- 変換先のディレクトリに十分な空き容量があることを確認してください
- ネットワーク接続が安定していることを確認してください
- 必要に応じてネットワークドライブのマッピングを行ってください 
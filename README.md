# 信号波形データ処理レポジトリ

このレポジトリは、信号波形データをスクリプト言語で扱えるようにすることを目的としています。Picoscopeで取得した`.psdata`ファイルを`.mat`形式に変換し、MATLABやPythonなどのスクリプト言語で解析できるようにします。

## 目的

- `.psdata`ファイルを`.mat`ファイルに変換する
- Picoscopeではなく、プログラミング環境（Jupyter Notebookなど）でデータを確認・分析できるようにする
- 波形データの処理を自動化・効率化する

## 必要環境

### オペレーティングシステム
- Windows 10/11（推奨）
- macOS（一部機能に制限あり）
- Linux（一部機能に制限あり）

### 必須ソフトウェア
- **Picoscope 7**：`.psdata`ファイルの生成と変換に必要
- **PowerShell 5.1以上**：変換スクリプトの実行環境
- **MATLAB**（R2019b以上推奨）または **Python 3.7以上**：変換後のデータ処理用

### 推奨ソフトウェア
- **Jupyter Notebook**：対話的なデータ分析用
- **Git**：バージョン管理用

## インストール方法

1. このレポジトリをクローンまたはダウンロード
2. PowerShellの実行ポリシーを設定（初回のみ）:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. 必要に応じて設定ファイルを編集

## 使用方法

詳細な使用方法は[documents/batchconvert.md](documents/batchconvert.md)を参照してください。

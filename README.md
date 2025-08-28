# 信号波形データ処理

このレポジトリは、信号波形を処理し機械学習用データセットを作成することを目的としています。Picoscopeで取得した`.psdata`ファイルはプログラミング言語で読み込むことができないため、汎用的な`.mat`形式に変換します。シミュレーションで生成したデータも`.mat`形式で保存されています。  
ただし、これらはファイル拡張子が同じでも、構造が異なるために波形の可視化には別々の関数を必要とします。そこで、これらを統一的に処理する枠組みとして`.npz`形式を採用します。実機・シミュレーションそれぞれのファイルを`.npz`に変換し保存した後、これらを適切に統合し、機械学習用データセット`x_train.npy,t_train.npy`, `x_test.npy`,`t_test.npy`を作成しています。  
実行方法は`main.ipynb`をご覧ください。


## 目的

- `.psdata`ファイルを`.mat`ファイルに変換する
- `.mat`ファイルを適切に処理し、統一形式として`.npz`形式のデータへと変換する。
- 機械学習用の`x_train.npy`, `t_train.npy`等を作成する。
- Picoscopeではなく、プログラミング環境でデータを確認・分析できるようにする
- 波形データの処理を自動化・効率化する

## 必要環境

### OS
- Ubuntu & Windows 10,11

### 必須ソフトウェア
- **Picoscope 7**：`.psdata`ファイルの生成と変換に必要
- **PowerShell 5.1以上**：変換スクリプトの実行環境
- **MATLAB**（R2019b以上推奨）または **Python 3.7以上**：変換後のデータ処理用
- **python** (3.8以上が望ましい)

## 実行方法
- picoscope to mat  
   PicoscopeがインストールされているPC上で、以下のコマンドをWindows Powershellで実行してください。これにより、指定のフォルダ内に存在する.psdataを一括で.matに変換できます。
   ```powershell
   Start-Process -FilePath <path/to/PicoScope.exe> -ArgumentList "BatchConvert", <path/to/input folder>, <path/to/output folder>, ".mat" -Wait -NoNewWindow
   ```
   （shellscript作成予定）

- mat to npz,npy
   このレポジトリを
   ```
   git clone https://github.com/apetrasc/psdata2matlab.git
   ```
   でクローンしてアクセス、そのあと`src`ディレクトリの中にある`mat2npz.py`を使って変換してください。使用例は`main.ipynb`に記載してあります。

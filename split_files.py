import csv
import re

# 入力ファイルを読み込み、新しいCSVファイルを作成
with open('matfiles.csv', 'r') as infile, open('separated_files.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # ヘッダー行を書き込み
    writer.writerow(['Patient_ID_Numbers', 'DateTime'])
    
    # 最初の行（ヘッダー）をスキップ
    next(reader)
    
    for row in reader:
        filename = row[0]
        # .matを除去
        name_without_ext = filename.replace('.mat', '')
        
        # ハイフンで分割
        if '-' in name_without_ext:
            parts = name_without_ext.split('-', 1)
            patient_part = parts[0]
            datetime_part = parts[1]
            
            # A列から数字のみを抽出（アルファベットを除去）
            patient_numbers = re.sub(r'[^0-9]', '', patient_part)
            
            writer.writerow([patient_numbers, datetime_part])
        else:
            # ハイフンがない場合は数字のみを抽出
            patient_numbers = re.sub(r'[^0-9]', '', name_without_ext)
            writer.writerow([patient_numbers, ''])

print("separated_files.csv が作成されました（A列からアルファベットを除去）") 
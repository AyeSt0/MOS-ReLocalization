import json
import csv

def extract_json_to_csv(json_file, csv_file):
    # 打开 JSON 文件并读取
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 打开 CSV 文件准备写入
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入 CSV 表头（原文 + 各语言翻译列）
        header = ['Original Text', 'English', 'Chinese', 'French', 'German', 'Spanish', 'Other Languages...']
        writer.writerow(header)
        
        # 遍历 JSON 数据并写入 CSV
        for key, value in data.items():
            original_text = value[1]  # 假设原文位于数组的第二个元素
            translations = value[2:]  # 后续元素是翻译内容
            
            # 将原文和翻译内容写入 CSV 文件
            row = [original_text] + translations
            writer.writerow(row)

if __name__ == "__main__":
    json_file = "data/language_dict.json"  # 游戏的原始 JSON 文件路径
    csv_file = "output/translations.csv"   # 输出的 CSV 文件路径
    
    extract_json_to_csv(json_file, csv_file)
    print(f"Data successfully extracted to {csv_file}")

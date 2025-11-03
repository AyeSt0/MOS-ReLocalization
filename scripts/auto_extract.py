import importlib
import subprocess
import sys
import json
import csv
import random
from pathlib import Path
from collections import Counter

# ========== 1️⃣ 自动安装依赖 ==========
def ensure_package(pkg):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"📦 Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# 必要依赖
for package in ["langdetect", "pandas", "openpyxl"]:
    ensure_package(package)

from langdetect import detect

# ========== 2️⃣ 基础路径配置 ==========
INPUT_JSON = Path("data/language_dict.json")
OUTPUT_CSV = Path("output/translations_auto.csv")
OUTPUT_CONFIG = Path("scripts/config_auto.py")
SAMPLE_SIZE = 300  # 抽样检测的行数

# ========== 3️⃣ 语言检测函数 ==========
def detect_language_map(json_path, sample_size=300):
    """自动检测 JSON 每列对应的语言分布"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keys = list(data.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    column_langs = {}

    for key in sample_keys:
        arr = data[key]
        for i, text in enumerate(arr):
            if not text or len(text.strip()) < 2:
                continue
            try:
                lang = detect(text)
            except Exception:
                lang = "unknown"
            column_langs.setdefault(i, []).append(lang)

    summary = {}
    for idx, langs in column_langs.items():
        counter = Counter(langs)
        top_lang, count = counter.most_common(1)[0]
        summary[idx] = {
            "most_common": top_lang,
            "confidence": round(count / len(langs), 2),
            "distribution": dict(counter),
        }

    print("\n=== Language Column Detection Summary ===")
    for idx, info in summary.items():
        print(
            f"Index {idx:>2}: {info['most_common']:>7} ({info['confidence']*100:.0f}% confidence)"
        )
    return summary


# ========== 4️⃣ 自动生成配置文件 ==========
def generate_config(summary, path):
    lines = ["# 自动生成的语言索引配置", "LANG_INDEX = {"]
    for idx, info in summary.items():
        lang = info["most_common"]
        lang = lang.replace("zh-cn", "zh").replace("zh-tw", "zh-TW")
        lines.append(f"    {idx}: '{lang}',  # confidence {info['confidence']*100:.0f}%")
    lines.append("}\n")
    lines.append("\nTARGET_LANG = 'zh'  # 默认优化中文\n")
    text = "\n".join(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n✅ 已生成配置文件: {path}")


# ========== 5️⃣ 导出 CSV ==========
def extract_json_to_csv(json_file, csv_file, lang_index):
    """根据语言索引导出 CSV"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["key"] + [lang_index[i] for i in sorted(lang_index.keys())]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for key, arr in data.items():
            row = [key]
            for i in sorted(lang_index.keys()):
                row.append(arr[i] if i < len(arr) else "")
            writer.writerow(row)

    print(f"✅ 已生成对齐良好的 CSV 文件：{csv_file}")


# ========== 6️⃣ 主流程入口 ==========
def main():
    print("🔍 正在检测语言列分布...")
    summary = detect_language_map(INPUT_JSON, sample_size=SAMPLE_SIZE)

    print("\n🧩 正在生成 config_auto.py ...")
    generate_config(summary, OUTPUT_CONFIG)

    print("\n📤 正在导出 CSV ...")
    extract_json_to_csv(
        INPUT_JSON, OUTPUT_CSV, {k: v["most_common"] for k, v in summary.items()}
    )

    print("\n🎉 完成！请查看以下文件：")
    print(f"  1️⃣ {OUTPUT_CONFIG}")
    print(f"  2️⃣ {OUTPUT_CSV}")
    print("\n✨ 你现在可以打开 CSV 对中文列进行优化，然后再回填 JSON。")


if __name__ == "__main__":
    main()

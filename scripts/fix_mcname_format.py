# scripts/fix_mcname_format.py
import json
import re
from pathlib import Path

# ===== 文件路径 =====
DATA_PATH = Path("output/language_dict_fixed.json")
OUTPUT_PATH = Path("output/language_dict_mcname_fixed.json")
REPORT_PATH = Path("output/mcname_fix_report.txt")

# ===== 正则定义 =====
# 匹配各种错误形式（全角、花括号、空格、嵌套等）
BAD_MCNAMES = re.compile(
    r"[\{\}\（\）\[\]［］｛｝＜＞⟨⟩]*\s*[mM][cC]\s*[nN][aA][mM][eE]\s*[\{\}\（\）\[\]［］｛｝＜＞⟨⟩]*"
)

# 匹配正确形式 [mcname]（大小写统一）
GOOD_MCNAME = "[mcname]"

def normalize_mcname(text: str) -> str:
    """将各种错误形式修正为 [mcname]"""
    if not text:
        return text
    # 修正各种括号错误和空格
    fixed = BAD_MCNAMES.sub(GOOD_MCNAME, text)
    return fixed

def main():
    if not DATA_PATH.exists():
        print(f"❌ 找不到文件: {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified_count = 0
    report_lines = []

    for key, arr in data.items():
        new_arr = []
        for i, text in enumerate(arr):
            if isinstance(text, str) and "mcname" in text.lower():
                fixed = normalize_mcname(text)
                if fixed != text:
                    modified_count += 1
                    report_lines.append(f"[{key}] 列 {i}:\n  原: {text}\n  改: {fixed}\n")
                new_arr.append(fixed)
            else:
                new_arr.append(text)
        data[key] = new_arr

    # 输出修正结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 输出报告
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"✅ 修正完成：共修复 {modified_count} 条异常 [mcname] 引用。")
    print(f"📁 已保存修正后文件: {OUTPUT_PATH}")
    print(f"📘 修正报告: {REPORT_PATH}")

if __name__ == "__main__":
    main()

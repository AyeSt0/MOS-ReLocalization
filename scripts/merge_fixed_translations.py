import json
import re
from pathlib import Path

DATA_PATH = Path("output/language_dict_translated.json")
FIXED_PATH = Path("output/review_english_mixed/mixed_entries_fixed.json")
OUTPUT_PATH = Path("output/language_dict_merged.json")

EN_PATTERN = re.compile(r"[A-Za-z]")
MCNAME_PATTERN = re.compile(r"[\[\{]+mcname[\]\}]+", re.IGNORECASE)

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def contains_meaningful_english(text: str) -> bool:
    """检测是否含非 mcname 的英文字符"""
    if not isinstance(text, str):
        return False
    tmp = MCNAME_PATTERN.sub("", text)
    return bool(EN_PATTERN.search(tmp))

def merge_fixes(data, fixed):
    updated_count = 0
    warn_count = 0
    for key, cols in fixed.items():
        if key not in data:
            print(f"⚠️ 警告：主数据中不存在 key = {key}，已跳过。")
            continue
        for col_str, new_text in cols.items():
            col = int(col_str)
            if len(data[key]) <= col:
                data[key].extend([""] * (col - len(data[key]) + 1))
            if contains_meaningful_english(new_text):
                print(f"⚠️ 修正文本仍含英文（key={key}, col={col}）: {new_text[:40]}")
                warn_count += 1
            data[key][col] = new_text.strip()
            updated_count += 1
    return updated_count, warn_count

def main():
    data = load_json(DATA_PATH)
    fixed = load_json(FIXED_PATH)

    updated, warns = merge_fixes(data, fixed)

    save_json(OUTPUT_PATH, data)
    print(f"\n✅ 合并完成，共更新 {updated} 条。")
    if warns > 0:
        print(f"⚠️ 其中 {warns} 条修正文本仍含英文，请再次检查。")
    print(f"📁 输出已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

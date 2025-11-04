import json
import re
from pathlib import Path

DATA_PATH = Path("output/language_dict_translated.json")
LANG_MAP_PATH = Path("data/language_map.json")
EXPORT_DIR = Path("output/review_english_mixed")

# 匹配英文字符
EN_PATTERN = re.compile(r"[A-Za-z]")
# 匹配各种形式的 mcname： [mcname]、{mcname}、{{mcname}}、[[mcname]] 等
MCNAME_PATTERN = re.compile(r"[\[\{]+mcname[\]\}]+", re.IGNORECASE)

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def export_mixed_entries(data, lang_map):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    chinese_cols = [int(k) for k, v in lang_map.items() if "Chinese" in v]
    print(f"✅ 识别到中文列: {chinese_cols}")

    mixed_entries = {}
    total_checked = 0

    for key, arr in data.items():
        for col in chinese_cols:
            if len(arr) > col:
                text = arr[col]
                if not isinstance(text, str):
                    continue
                if not text.strip():
                    continue
                # 包含英文，但排除仅含 mcname 的句子
                if EN_PATTERN.search(text):
                    # 若去掉 mcname 后仍有英文字符，才导出
                    tmp = MCNAME_PATTERN.sub("", text)
                    if EN_PATTERN.search(tmp):
                        if key not in mixed_entries:
                            mixed_entries[key] = {}
                        mixed_entries[key][str(col)] = text
                        total_checked += 1

    if not mixed_entries:
        print("✅ 没有检测到含英文的中文翻译。")
        return

    out_path = EXPORT_DIR / "mixed_entries.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mixed_entries, f, ensure_ascii=False, indent=2)

    print(f"✅ 导出完成，共发现 {len(mixed_entries)} 条含英文的中文译文，共 {total_checked} 处。")
    print(f"📁 文件已保存到: {out_path}")

if __name__ == "__main__":
    data = load_json(DATA_PATH)
    lang_map = load_json(LANG_MAP_PATH)
    export_mixed_entries(data, lang_map)

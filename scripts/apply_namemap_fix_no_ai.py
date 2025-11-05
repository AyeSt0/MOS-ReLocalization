import json
import re
import time
from pathlib import Path

# ========== 文件路径 ==========
DATA_PATH = Path("output/language_dict_namemap_applied.json")
LANG_MAP_PATH = Path("data/language_map.json")
NAMEMAP_PATH = Path("data/name_map.json")
OUTPUT_PATH = Path("output/language_dict_name_fixed.json")
REPORT_PATH = Path("output/name_fix_report.txt")

# ========== 加载函数 ==========
def load_json(path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

# ========== 核心替换函数 ==========
def normalize_chinese_names(text, name_map):
    """
    将文本中出现的中英俄混合专名替换为 name_map 中定义的中文译名。
    """
    if not text:
        return text, []

    replaced = []
    fixed = text
    for key, val in sorted(name_map.items(), key=lambda kv: -len(kv[0])):  # 长词优先
        # 模糊匹配：英文、俄文名或已有中译名
        variants = set([key, val])
        if key.lower() != key:
            variants.add(key.lower())
        if key.upper() != key:
            variants.add(key.upper())

        for var in variants:
            if not var or var == val:
                continue
            pattern = re.compile(re.escape(var))
            if re.search(pattern, fixed):
                fixed = pattern.sub(val, fixed)
                replaced.append((var, val))
    return fixed, replaced

# ========== 主逻辑 ==========
def main():
    data = load_json(DATA_PATH)
    lang_map = load_json(LANG_MAP_PATH)
    name_map = load_json(NAMEMAP_PATH)

    total = len(data)
    ru_col = next((int(k) for k, v in lang_map.items() if "Russian" in v), None)
    en_col = next((int(k) for k, v in lang_map.items() if "English" in v), None)
    zh_col = next((int(k) for k, v in lang_map.items() if "Chinese" in v), None)

    if None in (ru_col, en_col, zh_col):
        print("❌ 未检测到完整的三语列，请检查 language_map.json。")
        return

    print(f"✅ 数据载入成功：共 {total} 条。俄文列={ru_col}，英文列={en_col}，中文列={zh_col}")
    print(f"📘 name_map 中有 {len(name_map)} 条专名映射。")

    modified = 0
    report_lines = []

    start = time.time()

    for i, (key, row) in enumerate(data.items(), 1):
        if len(row) <= max(ru_col, en_col, zh_col):
            continue
        ru, en, cn = row[ru_col] or "", row[en_col] or "", row[zh_col] or ""
        # 若该行的俄文或英文包含任何 name_map 键，则触发修正
        if any(k in ru or k in en for k in name_map.keys()):
            new_cn, replaced = normalize_chinese_names(cn, name_map)
            if new_cn != cn:
                data[key][zh_col] = new_cn
                modified += 1
                report_lines.append(
                    f"【{i}】发现修正：{replaced}\n原文：{cn}\n修正：{new_cn}\n"
                )

        if i % 200 == 0:
            print(f"⏳ 进度 {i}/{total} | 已修正 {modified}")
            save_json(OUTPUT_PATH, data)

    save_json(OUTPUT_PATH, data)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ 修正完成，共 {modified} 条。")
    print(f"📘 报告保存至: {REPORT_PATH}")
    print(f"📁 输出文件: {OUTPUT_PATH}")
    print(f"耗时: {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()

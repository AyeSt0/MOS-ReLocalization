# scripts/fix_mcname_tags.py
import json
import re
from pathlib import Path

INPUT_PATH = Path("output/language_dict_merged.json")  # 翻译后文件
OUTPUT_PATH = Path("output/language_dict_fixed.json")      # 修复后输出

# 定义所有可能被AI误改的形式
MCNAME_VARIANTS = [
    r"\{\s*mcname\s*\}",          # {mcname} / { mcname }
    r"\{\{\s*mcname\s*\}\}",      # {{mcname}}
    r"\[\s*mcname\s*\]",          # [ mcname ]
    r"<\s*mcname\s*>",            # <mcname>
    r"＜\s*mcname\s*＞",          # 全角尖括号
    r"｛\s*mcname\s*｝",           # 全角花括号
    r"【\s*mcname\s*】",          # 方括号
    r"\(mcname\)",                # (mcname)
]

# 统一替换为 [mcname]
FIX_TO = "[mcname]"

def fix_text(text: str) -> str:
    new_text = text
    for pattern in MCNAME_VARIANTS:
        new_text = re.sub(pattern, FIX_TO, new_text, flags=re.IGNORECASE)
    return new_text

def fix_json(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f"❌ 找不到输入文件: {input_path}")
        return

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total, changed = 0, 0
    for key, arr in data.items():
        for i in range(len(arr)):
            old = arr[i]
            if isinstance(old, str) and "mcname" in old.lower():
                new = fix_text(old)
                total += 1
                if new != old:
                    arr[i] = new
                    changed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 修复完成，共检测到 {total} 条含 mcname 的文本；修正 {changed} 条。")
    print(f"📁 修复后的文件已保存至: {output_path}")

if __name__ == "__main__":
    fix_json(INPUT_PATH, OUTPUT_PATH)

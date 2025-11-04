# ===================================
# 语言自动检测（游戏版）
# 支持：葡语方言、简繁中文区分、游戏UI命名映射
# ===================================

import json
from pathlib import Path
from langdetect import detect, DetectorFactory

# ===== 配置 =====
DATA_PATH = Path("data/language_dict.json")
OUTPUT_PATH = Path("output/language_map.json")

# ===== 初始化 =====
DetectorFactory.seed = 0


# ===== 方言判定 =====
def guess_portuguese_region(samples):
    """区分葡萄牙语与巴西葡语"""
    br_markers = ["você", "pra", "tá", "legal", "cara", "aí", "beleza", "obrigado", "garota"]
    pt_markers = ["tu", "fixe", "gajo", "rapariga", "estás", "pois", "obrigada", "prenda"]
    br_count = sum(any(m in s.lower() for m in br_markers) for s in samples)
    pt_count = sum(any(m in s.lower() for m in pt_markers) for s in samples)
    if br_count > pt_count:
        return "Brazilian Port."
    elif pt_count > br_count:
        return "Portuguese"
    else:
        return "Portuguese"

# ===== 游戏语言映射表 =====
GAME_LANG_MAP = {
    "RU": "Russian",
    "EN": "English",
    "DE": "German",
    "FR": "French",
    "IT": "Italian",
    "ES": "Spanish",
    "TR": "Turkish",
    "PL": "Polish",
    "CS": "Czech",
    "UK": "Ukrainian",
    "AR": "Arabian",
    "FA": "Persian",
    "HU": "Hungarian",
    "PT": "Portuguese",
    "PT-BR": "Brazilian Port.",
    "ZH-CN": "Chinese (Simplified Chinese)",
    "ZH-TW": "Chinese (Traditional Chinese)",
    "META": "META",
    "UNKNOWN": "Unknown"
}

# ===== 检测函数 =====
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "UNKNOWN"


# ===== 主流程 =====
def detect_all_languages_pro(verbose=True):
    if not DATA_PATH.exists():
        print(f"[❌] 文件不存在: {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    max_cols = max(len(v) for v in data.values())
    print(f"✅ 总条目数: {total}，检测列数: {max_cols}")

    col_texts = {i: [] for i in range(max_cols)}
    for v in data.values():
        for i, val in enumerate(v):
            if isinstance(val, str) and val.strip():
                col_texts[i].append(val.strip())

    result_map = {}

    for col, texts in col_texts.items():
        if col == 0:
            print(f"\n🧩 列 {col:02d}: META (跳过语言检测)")
            result_map[col] = "META"
            continue

        if not texts:
            result_map[col] = "Unknown"
            print(f"\n🧩 列 {col:02d}: Unknown (0%) | 样本数: 0")
            continue

        # 按字符长度排序，取最长的200条用于检测
        sorted_texts = sorted(texts, key=len, reverse=True)[:200]
        detected = [detect_language(t).upper() for t in sorted_texts if detect_language(t)]

        if not detected or all(l == "UNKNOWN" for l in detected):
            display_name = "Unknown"
        else:
            main_lang = max(set(detected), key=detected.count)
            display_name = GAME_LANG_MAP.get(main_lang, "Unknown")

            # 特殊处理葡语
            if main_lang == "PT":
                display_name = guess_portuguese_region(sorted_texts)

        # 百分比 = 有效样本数 / 总条目数
        pct = (len(texts) / total) * 100

        result_map[col] = display_name
        print(f"\n🧩 列 {col:02d}: {display_name} ({pct:.0f}%) | 样本数: {len(texts)}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result_map, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 检测完成，共 {max_cols} 列；唯一语言数: {len(set(result_map.values()))}")
    print(f"📁 结果已保存至: {OUTPUT_PATH}")


# ===== 程序入口 =====
if __name__ == "__main__":
    detect_all_languages_pro(verbose=True)
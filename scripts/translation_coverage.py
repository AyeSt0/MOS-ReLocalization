import json
import argparse
from pathlib import Path

DATA_PATH = Path("data/language_dict.json")
OUTPUT_PATH = Path("output/translation_coverage.json")

def analyze_translation_coverage(save=False):
    """检测语言翻译完成度（每列填充率 + 总体进度）"""
    if not DATA_PATH.exists():
        print(f"[❌] 文件不存在: {DATA_PATH}")
        return

    print(f"🔍 正在读取: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"✅ 总条目数: {total}")

    # 自动检测最大列数
    max_cols = max(len(v) for v in data.values())
    print(f"📏 检测到最大列数: {max_cols}")

    filled_count = [0] * max_cols
    empty_count = [0] * max_cols

    # 遍历统计每列的非空与空
    for values in data.values():
        for i in range(max_cols):
            val = values[i] if i < len(values) else ""
            if str(val).strip():
                filled_count[i] += 1
            else:
                empty_count[i] += 1

    print("\n📊 ===== 翻译覆盖率报告 =====")
    coverage_report = {}
    for i in range(max_cols):
        filled = filled_count[i]
        empty = empty_count[i]
        coverage = round(filled / total * 100, 2)
        coverage_report[i] = {
            "filled": filled,
            "empty": empty,
            "coverage_%": coverage
        }

        # 可视化进度条
        bar = "█" * int(coverage // 2) + "-" * (50 - int(coverage // 2))
        print(f"列 {i:02d}: {coverage:5.2f}% |{bar}|  ({filled}/{total})")

    # 平均覆盖率
    avg_coverage = round(sum(filled_count) / (total * max_cols) * 100, 2)
    print(f"\n🌍 平均翻译完成度: {avg_coverage}%")

    # 找出最完整与最不完整的列
    best_col = max(range(max_cols), key=lambda i: filled_count[i])
    worst_col = min(range(max_cols), key=lambda i: filled_count[i])
    print(f"🏆 填充最多的列: {best_col}（{filled_count[best_col]}/{total}）")
    print(f"⚠️ 填充最少的列: {worst_col}（{filled_count[worst_col]}/{total}）")

    if save:
        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(coverage_report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存: {OUTPUT_PATH}")
    else:
        print("\n⚙️ 未保存文件（如需保存，请使用参数 --save）")

    return coverage_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检测语言翻译覆盖率（完成度分析）")
    parser.add_argument("--save", "-s", action="store_true", help="是否保存检测结果为 JSON 文件")
    args = parser.parse_args()

    analyze_translation_coverage(save=args.save)

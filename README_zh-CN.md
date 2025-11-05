<div align="center">

# MOS-ReLocalization 🌐
<em>专为 <strong>《MILFs of Sunville》</strong> 打造的 JSON 再本地化工具集。</em>

[English](./README.md) | [中文](./README_zh-CN.md)

<p>
  <img src="https://img.shields.io/badge/Python-3.13+-blue">
  <img src="https://img.shields.io/badge/License-MIT-green">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen">
  <img src="https://img.shields.io/badge/AI-OpenAI%20%7C%20DeepSeek-purple">
  <img src="https://img.shields.io/badge/Data-JSON-orange">
</p>

</div>

MOS-ReLocalization 是一个面向 JSON 的再本地化工具集，支持《MILFs of Sunville》或其它基于 JSON 的本地化项目。提供 AI 翻译、占位符修复、中文中夹英审校、专名统一与覆盖率统计，产出更自然、连贯的一致译文。

---

## 功能特点

- 语言列自动识别：自动生成 列→语言 映射（`scripts/detect_all_languages.py` → `data/language_map.json`）
- AI 翻译（俄/英 → 中文）：基于上下文生成自然中文（`scripts/ai_translate.py` → `output/language_dict_translated.json`）
- 中文夹英审校与合并：导出待修条目→人工修正→合并回主表（`scripts/export_english_in_chinese.py` → 修正 → `scripts/merge_fixed_translations.py`）
- 占位符修复：规范化 `[mcname]` / `[mcsurname]` 及其变体（`scripts/fix_mcname_tags.py`、`scripts/fix_mcname_format.py`、`scripts/fix_mcsurname_tags.py`）
- 专名统一：基于 `data/name_map.json`，可直接应用或结合 AI 辅助
  - 应用 + AI 质检：`scripts/ai_apply_namemap_unify.py`
  - 一体化（合并候选 + 应用）：`scripts/unify_namemap_and_apply_ai.py`
  - 另一套应用器：`scripts/ai_apply_namemap_aiunify.py`
  - 候选合并并生成统一映射：`scripts/ai_name_unify_with_namemap.py`
  - 纯替换（不经 AI）：`scripts/apply_namemap_fix_no_ai.py`
- 覆盖率统计：按列统计填充率（`scripts/translation_coverage.py`）

---

## 环境准备

1) Python 3.13+

2) 安装依赖
```bash
pip install -U openai python-dotenv langdetect
```

3) 配置 `.env`（示例）
```ini
# 引擎选择
MODEL_PROVIDER=deepseek          # 或 chatgpt

# OpenAI
OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini

# DeepSeek（OpenAI 兼容）
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.siliconflow.cn
DEEPSEEK_MODEL=deepseek-ai/DeepSeek-R1

# 可选：限速/并发
RPM=1000
TPM=100000
ASYNC_CONCURRENCY=100
PRINT_EVERY=50
BATCH_FLUSH=200
```

---

## 推荐流程

1) 检测语言列
```bash
python scripts/detect_all_languages.py
```
输出：`data/language_map.json`

2) AI 翻译为中文（结合俄/英语境）
```bash
python scripts/ai_translate.py
```
输出：`output/language_dict_translated.json`

3) 中文夹英审校并合并回主表
```bash
# 导出含英文字的中文条目
python scripts/export_english_in_chinese.py
# 人工修正后保存到：output/review_english_mixed/mixed_entries_fixed.json
# 合并回主表
python scripts/merge_fixed_translations.py
```
输出：`output/language_dict_merged.json`

4) 占位符修复与规范化
```bash
python scripts/fix_mcname_tags.py            # → output/language_dict_fixed.json
python scripts/fix_mcname_format.py          # → output/language_dict_mcname_fixed.json
python scripts/fix_mcsurname_tags.py         # → output/language_dict_mcsurname_fixed.json
```

5) 专名统一（任选其一）
- 路径 A：按 `name_map` 应用并经 AI 质检
```bash
python scripts/ai_apply_namemap_unify.py     # → output/language_dict_namemap_applied.json
```
- 路径 B：一体化（候选合并 + 应用）
```bash
python scripts/unify_namemap_and_apply_ai.py # → output/language_dict_name_final.json
```
- 其他（进阶）
```bash
python scripts/ai_apply_namemap_aiunify.py
python scripts/ai_name_unify_with_namemap.py
python scripts/apply_namemap_fix_no_ai.py
```

6) 覆盖率统计
```bash
python scripts/translation_coverage.py --save
```
输出：`output/translation_coverage.json`

---

## 数据与产物

- 输入数据：`data/language_dict.json`、`data/name_map.json`
- 语言映射：`data/language_map.json`（由脚本生成）
- 中间/最终产物：位于 `output/`（JSON 文件 + 报告，如 `name_unify_report.txt`）
- 过程缓存：位于 `cache/`

---

## 注意事项

- 限速：DeepSeek 典型 RPM≈1000 / TPM≈100000；遇到 429 会自动降速重试
- 占位符：请保持 `[mcname]`、`[mcsurname]` 不变
- 专名：先完善 `data/name_map.json`，再执行统一以保证全局一致

---

## 许可

MIT License © 2025 AyeSt0

> 重铸语言 —— 面向《MILFs of Sunville》的再本地化。


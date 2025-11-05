<div align="center">

# MOS-ReLocalization 🌐
<em>A re-localization toolkit for <strong>MILFs of Sunville</strong> (JSON-based).</em>

[English](./README.md) | [中文](./README_zh-CN.md)

<p>
  <img src="https://img.shields.io/badge/Python-3.13+-blue">
  <img src="https://img.shields.io/badge/License-MIT-green">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen">
  <img src="https://img.shields.io/badge/AI-OpenAI%20%7C%20DeepSeek-purple">
  <img src="https://img.shields.io/badge/Data-JSON-orange">
</p>

</div>

MOS-ReLocalization is a Python toolkit for re-localizing MILFs of Sunville (or any JSON-based localization). It provides AI translation, placeholder fixing, English-in-Chinese review, proper‑noun unification, and coverage analytics — producing cleaner, more natural Chinese output.

---

## Features

- Language column detection: auto build column→language map (`scripts/detect_all_languages.py` → `data/language_map.json`)
- AI translation (RU/EN → ZH): context‑aware Chinese localization (`scripts/ai_translate.py` → `output/language_dict_translated.json`)
- English‑in‑Chinese review and merge: export mixed entries, manually fix, then merge (`scripts/export_english_in_chinese.py` → fix → `scripts/merge_fixed_translations.py`)
- Placeholder fixes: normalize `[mcname]` / `[mcsurname]` and variants (`scripts/fix_mcname_tags.py`, `scripts/fix_mcname_format.py`, `scripts/fix_mcsurname_tags.py`)
- Proper‑noun unification: apply `data/name_map.json` and optionally AI‑assisted unification
  - Apply + AI QA: `scripts/ai_apply_namemap_unify.py`
  - Unified flow (merge + apply): `scripts/unify_namemap_and_apply_ai.py`
  - Alternative applier: `scripts/ai_apply_namemap_aiunify.py`
  - Merge candidates & generate mapping: `scripts/ai_name_unify_with_namemap.py`
  - No‑AI quick apply (pure replace): `scripts/apply_namemap_fix_no_ai.py`
- Coverage analytics: per‑column fill ratio (`scripts/translation_coverage.py`)

---

## Setup

1) Python 3.13+

2) Install dependencies
```bash
pip install -U openai python-dotenv langdetect
```

3) Configure environment via `.env` (examples)
```ini
# Engine selection
MODEL_PROVIDER=deepseek          # or: chatgpt

# OpenAI
OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini

# DeepSeek (OpenAI-compatible)
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.siliconflow.cn
DEEPSEEK_MODEL=deepseek-ai/DeepSeek-R1

# Optional rate limiting / concurrency
RPM=1000
TPM=100000
ASYNC_CONCURRENCY=100
PRINT_EVERY=50
BATCH_FLUSH=200
```

---

## Recommended Workflow

1) Detect language columns
```bash
python scripts/detect_all_languages.py
```
Output: `data/language_map.json`

2) AI translation to Chinese (using RU + EN)
```bash
python scripts/ai_translate.py
```
Output: `output/language_dict_translated.json`

3) Review English‑in‑Chinese and merge back
```bash
# Export mixed entries in Chinese columns
python scripts/export_english_in_chinese.py
# Manually fix: output/review_english_mixed/mixed_entries_fixed.json
# Merge back
python scripts/merge_fixed_translations.py
```
Output: `output/language_dict_merged.json`

4) Fix placeholders and normalize formats
```bash
python scripts/fix_mcname_tags.py            # → output/language_dict_fixed.json
python scripts/fix_mcname_format.py          # → output/language_dict_mcname_fixed.json
python scripts/fix_mcsurname_tags.py         # → output/language_dict_mcsurname_fixed.json
```

5) Unify proper nouns (pick one path)
- Path A: Apply `name_map` with AI QA
```bash
python scripts/ai_apply_namemap_unify.py     # → output/language_dict_namemap_applied.json
```
- Path B: Merge candidates + apply in one go
```bash
python scripts/unify_namemap_and_apply_ai.py # → output/language_dict_name_final.json
```
- Alternatives (advanced)
```bash
python scripts/ai_apply_namemap_aiunify.py
python scripts/ai_name_unify_with_namemap.py
python scripts/apply_namemap_fix_no_ai.py
```

6) Coverage report
```bash
python scripts/translation_coverage.py --save
```
Output: `output/translation_coverage.json`

---

## Data & Outputs

- Input data: `data/language_dict.json`, `data/name_map.json`
- Language map: `data/language_map.json` (generated)
- Intermediate/final outputs under `output/` (JSON + reports like `name_unify_report.txt`)
- Caches under `cache/`

---

## Notes

- Rate limits: DeepSeek typical RPM≈1000 / TPM≈100000; scripts auto back‑off on 429.
- Placeholders: keep `[mcname]`, `[mcsurname]` intact in translations.
- Proper nouns: curate `data/name_map.json` first, then run unification to ensure consistency.

---

## License

MIT License © 2025 AyeSt0

> Re‑forging words — re‑localizing MILFs of Sunville.


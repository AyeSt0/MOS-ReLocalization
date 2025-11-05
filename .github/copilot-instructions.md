# MOS-ReLocalization AI 助手指南

## 项目概述

MOS-ReLocalization 是一个专门为《MILFs of Sunville》游戏设计的多语言翻译工具集。它主要处理 JSON 格式的翻译文件，实现:

- 多语言文本提取与分析
- AI 辅助翻译与润色
- 专有名词映射与统一
- JSON 翻译文件重建

## 关键概念

### 数据结构

- `/data/` - 原始多语言数据
  - `language_dict.json` - 主翻译字典
  - `language_map.json` - 语言列映射配置
  - `name_map.json` - 专有名词映射表

### 专名处理

1. `[mcname]` 标记处理
2. 专有名词统一映射
3. 基于上下文的智能专名翻译

### 翻译流程

1. 基础翻译: ru_col (情感/语气) + en_col (语义) → zh_col
2. 专名统一: 使用 `name_map.json` 进行专有名词映射
3. 质量检查: 检测英文混入、标记格式等问题

## 核心工作流

### 翻译与润色

```python
DATA_PATH = "data/language_dict.json"
LANG_MAP_PATH = "data/language_map.json"
NAME_MAP_PATH = "data/name_map.json"

# 1. 载入数据
data = load_json(DATA_PATH)
lang_map = load_json(LANG_MAP_PATH)

# 2. 定位语言列
ru_col = pick_col_by_lang(lang_map, "Russian")
en_col = pick_col_by_lang(lang_map, "English")
zh_col = pick_col_by_lang(lang_map, "Chinese")

# 3. 处理翻译/润色
for key, row in data.items():
    ru, en = row[ru_col], row[en_col]
    # 处理翻译...
```

### 命名规范

- 使用 Path 对象处理文件路径
- JSON 保存时使用临时文件避免损坏
- 缓存使用 sha1 作为键值
- 异步操作使用信号处理确保安全退出

## 常见场景

### 1. 专名统一

检查并应用 `name_map.json` 中的映射规则，确保专有名词翻译一致性。

### 2. 格式修正

- `[mcname]` 标记规范化
- 删除多余空格和换行
- 处理特殊字符

### 3. 质量检查

- 检测中文中的英文混入
- 验证标记格式正确性
- 确认翻译完整性

## 开发提示

1. 使用 `.env` 配置环境变量:
   - `OPENAI_API_KEY`/`DEEPSEEK_API_KEY`
   - `MODEL_PROVIDER`
   - `RPM`/`TPM` 速率限制

2. 异步并发处理:
   - 使用 `asyncio.Semaphore` 控制并发
   - 实现 `RateLimiter` 限速
   - 定期保存进度

3. 文件处理:
   - 总是使用 `utf-8` 编码
   - 使用临时文件 + 替换方式保存
   - 保持输入输出路径一致性
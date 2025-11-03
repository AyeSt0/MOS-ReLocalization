<div align="center">

# MOS-ReLocalization 🌐
*专为《MILFs of Sunville》打造的再本地化工具集。*

[English](./README.md) | [中文](./README.zh-CN.md)

<p>
  <img src="https://img.shields.io/badge/Python-3.13+-blue">
  <img src="https://img.shields.io/badge/License-MIT-green">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen">
</p>

</div>

**MOS-ReLocalization** 是一个基于 Python 的 **再本地化（Re-localization）** 工具集，专为 *MILFs of Sunville* 游戏设计。  
它可以提取多语言数据，将其导出为可编辑的 CSV 文件，用于 AI 辅助或人工翻译润色，并能自动将优化后的译文重新写入 JSON 文件，从而让游戏文本更加自然流畅、更具代入感。

---

## ✨ 功能特点

- 🧩 **JSON 提取器** – 自动识别多语言 JSON 字典 / 数组结构并导出为 CSV  
- 🪄 **翻译润色器** – 使用 AI 工具或人工方式提升文本语气与自然度  
- 🔁 **JSON 回填器** – 将润色后的译文自动插入原始 JSON 对应语言位置  
- 🕹 **游戏适配输出** – 生成可直接替换至游戏目录的 JSON 翻译文件  

---

## 🚀 快速开始

### 1. 环境要求
安装 **Python 3.13 或更高版本**  
```bash
python --version
```
### 2. 克隆仓库
```bash
git clone https://github.com/<your-username>/MOS-ReLocalization.git
cd MOS-ReLocalization
```
### 3. 运行脚本
```bash
python scripts/extract_to_csv.py
python scripts/rebuild_from_csv.py
```
脚本运行后，会在 `/output` 目录中生成对应的输出文件，并在控制台显示日志信息。

---

## 📁 项目结构

```text
MOS-ReLocalization/
│
├── data/           # 原始多语言 JSON 文件  
├── output/         # 处理后或回填后的文件  
├── scripts/        # Python 脚本目录  
│   ├── extract_to_csv.py      # 提取原文与译文  
│   ├── rebuild_from_csv.py    # 从 CSV 回写 JSON  
│   └── utils.py               # 通用函数模块（可选）  
├── README.md
└── README.zh-CN.md
```

---

## 🧭 开发计划（Roadmap）

- [ ] 支持多层嵌套翻译结构
- [ ] 支持批量文件夹处理
- [ ] 自动备份与差异日志记录
- [ ] 可选集成 DeepL / OpenAI 翻译 API
- [ ] 提供图形化（GUI）版本

---

## 🪪 License

MIT License © 2025 **AyeSt0**

---

> *重铸语言，重构世界 —— Sunville 的再本地化工程。*


<div align="center">

# MOS-ReLocalization 🌐
*A re-localization toolkit specifically made for **MILFs of Sunville**.*

[English](./README.md) | [中文](./README.zh-CN.md)

<p>
  <img src="https://img.shields.io/badge/Python-3.13+-blue">
  <img src="https://img.shields.io/badge/License-MIT-green">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen">
</p>

</div>

**MOS-ReLocalization** is a Python-based toolkit for the **re-localization** of *MILFs of Sunville* — or any game that uses JSON-based translation files.  
It extracts multilingual data, converts them into editable CSV format for AI-assisted or manual translation, and then rebuilds updated JSON files — enabling smoother, more natural translations and greater immersion across all languages.

---

## ✨ Features

- 🧩 **JSON Extractor** – automatically detect multilingual JSON dictionary/array structures and export them to CSV  
- 🪄 **Translation Refiner** – improve or rewrite translations using AI tools or manual editing  
- 🔁 **JSON Rebuilder** – insert the refined translations back into the corresponding language fields  
- 🕹 **Game-Ready Output** – produce JSON files ready to be re-imported into the game  

---

## 🚀 Getting Started

### 1. Prerequisites
Install **Python 3.13 or later**  
```bash
python --version
```
### 2. Clone the repository
```bash
git clone https://github.com/<your-username>/MOS-ReLocalization.git
cd MOS-ReLocalization
```
### 3. Run the scripts
```bash
python scripts/extract_to_csv.py
python scripts/rebuild_from_csv.py
```
Each script will log its actions and generate files inside the `/output` folder.

---

## 📁 Project Structure

```text
MOS-ReLocalization/
│
├── data/           # Original multilingual JSON files
├── output/         # Processed / rebuilt files
├── scripts/        # Python scripts
│   ├── extract_to_csv.py      # Extracts text and translations
│   ├── rebuild_from_csv.py    # Rebuilds JSON from edited CSV
│   └── utils.py               # Shared functions (if needed)
├── README.md
└── README.zh-CN.md
```

---

## 🧭 Roadmap

- [ ] Add support for nested translation structures  
- [ ] Add batch folder processing  
- [ ] Add automatic backup and diff log  
- [ ] Integrate DeepL / OpenAI API for optional AI translation  
- [ ] Add web/GUI version for non-developers  

---

## 🪪 License

MIT License © 2025 **AyeSt0**

---

> *Re-forging words, rebuilding worlds — the multilingual re-localization of Sunville.*


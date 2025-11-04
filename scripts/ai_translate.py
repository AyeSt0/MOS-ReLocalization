import os, json, time, sys, signal, threading, re
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== 环境加载 =====
load_dotenv()

# ===== 路径配置 =====
DATA_JSON = Path("data/language_dict.json")
LANG_MAP_JSON = Path("output/language_map.json")
OUTPUT_JSON = Path("output/language_dict_translated.json")
NAME_MAP_JSON = Path("output/name_map.json")

# ===== 读取 ENV =====
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "chatgpt").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
THREADS = int(os.getenv("THREADS", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
BATCH_FLUSH = int(os.getenv("BATCH_FLUSH", "50"))

# ===== 模型客户端 =====
if MODEL_PROVIDER == "chatgpt":
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
elif MODEL_PROVIDER == "deepseek":
    import openai
    openai.api_key = DEEPSEEK_API_KEY
    openai.base_url = f"{DEEPSEEK_BASE_URL}/v1"
    client = openai
else:
    raise RuntimeError("❌ MODEL_PROVIDER 必须为 chatgpt 或 deepseek")

# ===== 基础函数 =====
def load_json(path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def ensure_column_capacity(arr: list, n: int):
    while len(arr) < n:
        arr.append("")

# ===== 全局退出钩子 =====
def graceful_exit(signum, frame):
    print("\n⚠️ 捕获中断信号，安全落盘后退出……")
    save_json(OUTPUT_JSON, data)
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

# ===== Prompt 模板 =====
def build_prompt(text, target_lang):
    return f"""
You are a professional game localization translator.
Translate the following English text into natural, immersive Chinese ({target_lang}),
making it appropriate for a visual novel game.
Keep dialogue fluent and emotionally expressive; preserve tone and intent.
Do not omit or summarize details.
Text:
{text}
""".strip()

# ===== 翻译接口 =====
def translate_once(text, target_lang):
    if not text.strip():
        return ""

    for attempt in range(5):
        try:
            if MODEL_PROVIDER == "chatgpt":
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional translator."},
                        {"role": "user", "content": build_prompt(text, target_lang)},
                    ],
                    timeout=REQUEST_TIMEOUT,
                )
                return (resp.choices[0].message.content or "").strip()

            else:  # DeepSeek
                resp = client.ChatCompletion.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional translator."},
                        {"role": "user", "content": build_prompt(text, target_lang)},
                    ],
                    timeout=REQUEST_TIMEOUT,
                )
                return (resp.choices[0].message["content"] or "").strip()
        except Exception as e:
            wait = 2 * (attempt + 1)
            print(f"⏳ 重试 {attempt+1}/5：{type(e).__name__}，{wait}s 后再试...")
            time.sleep(wait)

    print(f"❌ 翻译失败（已达重试上限），返回空：{text[:40]}")
    return ""

# ===== 专有名词统一映射 =====
def apply_name_map(text, name_map):
    for k, v in name_map.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text

def update_name_map(original, translated, name_map):
    english_tokens = re.findall(r'\b[A-Z][a-zA-Z]+\b', original)
    for token in english_tokens:
        if token not in name_map and token.lower() not in ["the", "a", "an"]:
            if translated and not re.search(r"[A-Za-z]", translated):
                name_map[token] = translated
                print(f"🧩 新增专有名词映射：{token} → {translated}")
                save_json(NAME_MAP_JSON, name_map)
    return name_map

# ===== 新增语言列 =====
def add_new_language_column(data, lang_map):
    max_col = max(int(k) for k in lang_map.keys())
    print(f"\n📊 当前最大列号为 {max_col}。")
    choice = input("是否要新增语言列？(y/n)：").strip().lower()
    if choice != "y":
        return lang_map, None

    new_col_input = input(f"请输入要插入的新列号（默认 {max_col+1}）：").strip()
    new_col = int(new_col_input) if new_col_input else max_col + 1

    for i in range(max_col + 1, new_col):
        lang_map[str(i)] = "Unknown"

    new_lang_name = input("请输入新语言名称（例如：Japanese、Korean）：").strip() or "Unknown"
    lang_map[str(new_col)] = new_lang_name

    for v in data.values():
        ensure_column_capacity(v, new_col + 1)
        v[new_col] = ""

    save_json(DATA_JSON, data)
    save_json(LANG_MAP_JSON, lang_map)
    print(f"✅ 已新增列 {new_col}：{new_lang_name}")
    return lang_map, new_col

# ===== 可翻译列选择 =====
def pick_target_column(lang_map, data):
    english_col = None
    total = len(data)
    candidates, unknowns = [], []

    for k, v in sorted(lang_map.items(), key=lambda kv: int(kv[0])):
        if v == "META":
            continue
        count = sum(1 for row in data.values() if int(k) < len(row) and row[int(k)].strip())
        pct = (count / total) * 100
        if v == "English":
            english_col = int(k)
        if v == "Unknown":
            unknowns.append((int(k), v, pct))
        else:
            candidates.append((int(k), v, pct))

    print("\n可翻译列如下：")
    for c in candidates:
        print(f"  - 列 {c[0]}: {c[1]} ({c[2]:.1f}%)")
    if unknowns:
        print("\n🟡 检测到 Unknown 列，可选择创建新语言翻译：")
        for c in unknowns:
            print(f"  - 列 {c[0]}: {c[1]} ({c[2]:.1f}%)")

    tgt_col_input = input("\n👉 请输入要进行本地化翻译的目标列号（或回车新增语言）：").strip()
    if not tgt_col_input:
        lang_map, new_col = add_new_language_column(data, lang_map)
        if new_col is None:
            raise RuntimeError("未选择翻译目标列。")
        return english_col, new_col, lang_map
    return english_col, int(tgt_col_input), lang_map

# ===== 翻译执行 =====
def translate_all(data, english_col, target_col, target_lang, name_map):
    total = len(data)
    idx_lock = threading.Lock()
    counter = {"count": 0}

    def worker(k):
        with idx_lock:
            idx = counter["count"] + 1
            counter["count"] += 1

        arr = data[k]
        ensure_column_capacity(arr, target_col + 1)
        src = arr[english_col].strip()
        if not src:
            return
        src_with_replacement = apply_name_map(src, name_map)
        result = translate_once(src_with_replacement, target_lang)
        name_map = update_name_map(src, result, name_map)
        with idx_lock:
            arr[target_col] = result
            if idx % 10 == 0:
                save_json(OUTPUT_JSON, data)
            print(f"🔄 正在翻译第 {idx}/{total} 条...\n  原文: {src}\n  译文: {result}\n")

    with ThreadPoolExecutor(max_workers=THREADS) as exe:
        futures = [exe.submit(worker, k) for k in data.keys()]
        for _ in as_completed(futures):
            pass

# ===== 主程序入口 =====
def main():
    global data
    data = load_json(DATA_JSON, {})
    lang_map = load_json(LANG_MAP_JSON, {})
    name_map = load_json(NAME_MAP_JSON, {})

    print(f"✅ 加载完成，共 {len(data)} 条记录。")
    english_col, target_col, lang_map = pick_target_column(lang_map, data)

    print(f"\n🌍 将从列 {english_col}（English） 翻译到列 {target_col}（{lang_map[str(target_col)]}）")
    mode = input("\n选择模式：1=继续翻译（补空） / 2=强制翻译（清空重来）：").strip()
    if mode == "2":
        confirm = input("⚠️ 确认要清空该列的所有翻译吗？(y/n)：").strip().lower()
        if confirm == "y":
            for arr in data.values():
                ensure_column_capacity(arr, target_col + 1)
                arr[target_col] = ""
            print("🧹 已清空目标列。")

    not_done = {k: v for k, v in data.items() if len(v) <= target_col or not v[target_col].strip()}
    print(f"📦 待翻译: {len(not_done)} 条。")

    translate_all(data, english_col, target_col, lang_map[str(target_col)], name_map)
    save_json(OUTPUT_JSON, data)
    print(f"\n🎉 翻译完成，结果已保存至 {OUTPUT_JSON}")
    print(f"📘 专有名词映射表已更新 -> {NAME_MAP_JSON}")

if __name__ == "__main__":
    main()

# scripts/ai_translate.py
import os, sys, json, time, signal, re, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# ========== 初始化与路径 ==========
load_dotenv()

DATA_PATH        = Path("data/language_dict.json")
LANG_MAP_PATH    = Path("data/language_map.json")
NAME_MAP_PATH    = Path("data/name_map.json")
OUTPUT_PATH      = Path("output/language_dict_translated.json")

THREADS          = int(os.getenv("THREADS", "5"))
BATCH_FLUSH      = int(os.getenv("BATCH_FLUSH", "50"))
REQUEST_TIMEOUT  = int(os.getenv("REQUEST_TIMEOUT", "30"))

DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "chatgpt").strip().lower()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = os.getenv("MODEL", "gpt-4o-mini").strip()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASEURL = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn").strip()
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1").strip()

stop_requested = False
NAME_LOCK = threading.Lock()

# ========== 信号处理 ==========
def handle_signal(signum, frame):
    global stop_requested
    stop_requested = True
    print("\n⚠️ 捕获到中断信号，正在安全落盘并退出……")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ========== 工具函数 ==========
def load_json(path: Path, default=None):
    if default is None: default = {}
    if not path.exists(): return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def ensure_row_len(row: list, length: int):
    if len(row) <= length:
        row.extend([""] * (length - len(row) + 1))

# ========== 模型客户端 ==========
def choose_provider():
    print("\n请选择翻译引擎：")
    print("  1) ChatGPT (OpenAI)")
    print("  2) DeepSeek (OpenAI兼容)")
    choice = input(f"👉 输入 1 或 2 (默认: {DEFAULT_PROVIDER})：").strip()
    provider = "deepseek" if (choice == "2" or (not choice and DEFAULT_PROVIDER == "deepseek")) else "chatgpt"
    print(f"🧠 使用 {provider.upper()} 模型引擎\n")
    return provider

def build_client(provider: str):
    from openai import OpenAI
    if provider == "deepseek":
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASEURL), DEEPSEEK_MODEL
    return OpenAI(api_key=OPENAI_API_KEY), OPENAI_MODEL

# ========== Prompt ==========
def build_prompt(text, target_lang):
    return f"""
You are a professional Chinese localizer specializing in adult visual novels.
Translate the following Russian line into fluent, natural Chinese ({target_lang})
for the game "MILFs of Sunville".

Guidelines:
- Preserve sensuality, emotion, and tone.
- Keep idiomatic phrasing natural for modern Chinese dialogue.
- Only output the translation itself (no explanations, no quotes).
- Retain variables like {{mcname}}, [var], <tag>.

Text:
{text}
""".strip()

def build_name_prompt(names: list):
    joined = ", ".join(names)
    return f"""
Translate the following Russian or English person names into natural, culturally consistent Chinese names.
Output valid JSON only, e.g. {{"原名": "译名", ...}}, without extra text.

Names: {joined}
""".strip()

def clean_output(s):
    if not s: return ""
    s = s.strip().strip("`").strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[0] if lines else ""

# ========== 翻译器 ==========
class Translator:
    def __init__(self, provider):
        self.provider = provider
        self.client, self.model = build_client(provider)
        from openai import APIError, RateLimitError, APITimeoutError, BadRequestError
        self.APIError, self.RateLimitError, self.APITimeoutError, self.BadRequestError = APIError, RateLimitError, APITimeoutError, BadRequestError

    def chat(self, sys_prompt, user_prompt):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.5, timeout=REQUEST_TIMEOUT
        )
        return resp.choices[0].message.content or ""

    def translate_name_batch(self, names):
        prompt = build_name_prompt(names)
        try:
            out = self.chat("You are a precise transliteration assistant.", prompt)
            return json.loads(out)
        except Exception:
            try:
                cleaned = re.search(r"\{.*\}", out, re.S)
                return json.loads(cleaned.group(0)) if cleaned else {}
            except Exception:
                return {}

# ========== 专名热词训练 ==========
def pretrain_name_map(data, lang_map, translator):
    print("🧠 启动专名热词训练模式...\n")
    src_col = next((int(k) for k, v in lang_map.items() if "Russian" in v), None)
    if src_col is None:
        raise RuntimeError("language_map.json 未检测到 Russian 列。")

    name_map = load_json(NAME_MAP_PATH, default={})
    words = set()

    for arr in data.values():
        if len(arr) <= src_col: continue
        text = arr[src_col]
        if not text or not isinstance(text, str): continue
        tokens = re.findall(r"[А-ЯЁA-Z][а-яёa-z]{2,}", text)
        for t in tokens:
            if len(t) <= 2 or t.lower() in ("она","это","мой","мама"): continue
            words.add(t)

    unknown = [w for w in sorted(words) if w not in name_map]
    print(f"📊 发现 {len(unknown)} 个潜在专名。\n")

    new_map = {}
    batch_size = 20
    for i in range(0, len(unknown), batch_size):
        batch = unknown[i:i+batch_size]
        result = translator.translate_name_batch(batch)
        if isinstance(result, dict):
            new_map.update(result)
            print(f"✅ 已处理 {min(i+batch_size, len(unknown))}/{len(unknown)} 专名")
        save_json(NAME_MAP_PATH, {**name_map, **new_map})

    print("\n🎉 专名热词训练完成，结果已写入 -> output/name_map.json")

# ========== 主流程 ==========
def main():
    provider = choose_provider()
    translator = Translator(provider)

    data = load_json(DATA_PATH)
    lang_map = load_json(LANG_MAP_PATH)
    name_map = load_json(NAME_MAP_PATH)

    choice = input("是否进行专名热词训练？(y/n)：").strip().lower()
    if choice == "y":
        pretrain_name_map(data, lang_map, translator)
        print("✅ 专名热词训练完成，可重新运行进行翻译。")
        return

    total = len(data)
    src_col = next((int(k) for k, v in lang_map.items() if "Russian" in v), None)
    if src_col is None:
        print("❌ 未检测到 Russian 列。")
        return

    # 选择目标列
    print("\n可翻译列如下：")
    for k, v in lang_map.items():
        if v == "META": continue
        print(f"  - 列 {k}: {v}")
    tgt_col = int(input("\n👉 请输入目标列号：").strip() or 5)
    target_lang = lang_map.get(str(tgt_col), "Chinese (Simplified Chinese)")

    # 模式选择
    mode = input("选择模式：1=继续翻译 / 2=强制翻译：").strip()
    if mode == "2":
        confirm = input("⚠️ 确认清空目标列所有翻译吗？(y/n)：").strip().lower()
        if confirm == "y":
            for row in data.values():
                ensure_row_len(row, tgt_col)
                row[tgt_col] = ""
            save_json(DATA_PATH, data)
            print("🧹 已清空目标列。")

    # 翻译任务
    todo = []
    for key, row in data.items():
        ensure_row_len(row, max(src_col, tgt_col))
        src = (row[src_col] or "").strip()
        tgt = (row[tgt_col] or "").strip()
        if src and not tgt:
            todo.append((key, src))
    print(f"\n📦 待翻译 {len(todo)} 条。\n")

    processed = 0
    last_flush = 0

    def worker(key, text):
        pre = text
        out = translator.chat("You are a professional translator.", build_prompt(pre, target_lang))
        return key, text, clean_output(out)

    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [ex.submit(worker, key, src) for key, src in todo]
        for fut in as_completed(futures):
            if stop_requested: break
            try:
                key, src, out = fut.result()
            except Exception as e:
                print(f"❌ 执行错误：{e}")
                continue
            data[key][tgt_col] = out
            processed += 1
            print(f"🔄 正在翻译第 {processed}/{len(todo)} 条...\n  原文: {src}\n  译文: {out}\n")

            if processed - last_flush >= BATCH_FLUSH:
                save_json(OUTPUT_PATH, data)
                last_flush = processed
                print(f"💾 自动保存进度 -> {OUTPUT_PATH}")

    save_json(OUTPUT_PATH, data)
    print(f"\n🎉 翻译完成 -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

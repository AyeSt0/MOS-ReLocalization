import os
import json
import time
import asyncio
import math
import signal
import re
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

# ========== 环境 ==========
load_dotenv()

DATA_PATH   = Path("output/language_dict_mcsurname_fixed.json")
LANG_MAP    = Path("data/language_map.json")
NAME_MAP    = Path("data/name_map.json")
OUTPUT_PATH = Path("output/language_dict_namemap_applied.json")
REPORT_PATH = Path("output/name_unify_report.txt")

# 模型配置
DEFAULT_PROVIDER  = os.getenv("MODEL_PROVIDER", "deepseek").lower()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASEURL  = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn")
OPENAI_MODEL      = os.getenv("MODEL", "gpt-4o-mini")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")

# 限速
ASYNC_CONCURRENCY = int(os.getenv("ASYNC_CONCURRENCY", "100"))
RPM               = int(os.getenv("RPM", "1000"))
TPM               = int(os.getenv("TPM", "100000"))

# 全局状态
stop_requested = False
RATE_LIMIT_HITS = 0
MAX_CONCURRENCY = ASYNC_CONCURRENCY
PROCESSED = 0
LOCK = asyncio.Lock()


# ========== 工具 ==========
def load_json(p, default=None):
    if default is None:
        default = {}
    if not Path(p).exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        tmp.replace(p)
    except PermissionError:
        print("⚠️ 文件写入被占用，等待重试…")
        time.sleep(1)
        tmp.replace(p)

def pick_col(lang_map, key):
    for k, v in lang_map.items():
        if key.lower() in v.lower():
            return int(k)
    return -1

def clean_output(s: str):
    if not s:
        return ""
    lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
    return lines[0] if lines else ""

def handle_signal(signum, frame):
    global stop_requested
    stop_requested = True
    print("\n⚠️ 收到中断信号，准备安全退出…")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ========== 限速 ==========
class RateLimiter:
    def __init__(self, rpm, tpm):
        self.rpm = rpm
        self.tpm = tpm
        self.lock = asyncio.Lock()
        self.window_start = time.monotonic()
        self.req_used = 0
        self.tok_used = 0

    def _maybe_reset(self):
        now = time.monotonic()
        if now - self.window_start >= 60:
            self.window_start = now
            self.req_used = 0
            self.tok_used = 0

    async def acquire(self, text: str):
        est = len(text) // 3 + 50
        while True:
            async with self.lock:
                self._maybe_reset()
                if self.req_used + 1 <= self.rpm and self.tok_used + est <= self.tpm:
                    self.req_used += 1
                    self.tok_used += est
                    return
                wait = max(0.0, 60.0 - (time.monotonic() - self.window_start))
            await asyncio.sleep(min(1.0, wait))


# ========== 提示构建 ==========
def build_prompt(ru, en, cn, name_map):
    name_pairs = "\n".join([f"{k} → {v}" for k, v in name_map.items()])
    return f"""
You are a professional localization editor.
Task: refine the Chinese line to keep all proper nouns consistent with the following mappings.
Use Russian & English for context when necessary, but output ONLY the corrected Chinese.

Mappings:
{name_pairs}

Russian: {ru}
English: {en}
Chinese: {cn}

Rules:
- Replace all proper nouns to match the mapping.
- Keep emotion and tone natural.
- Output ONLY the corrected Chinese line (no commentary).
""".strip()


# ========== 主逻辑 ==========
async def main_async():
    global RATE_LIMIT_HITS, MAX_CONCURRENCY, PROCESSED

    provider = input("\n请选择引擎（1=ChatGPT，2=DeepSeek）：").strip()
    if provider == "1":
        model = OPENAI_MODEL
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        print("🧠 使用 CHATGPT 模型引擎")
    else:
        model = DEEPSEEK_MODEL
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASEURL)
        print("🧠 使用 DEEPSEEK 模型引擎")

    data = load_json(DATA_PATH)
    lang_map = load_json(LANG_MAP)
    name_map = load_json(NAME_MAP)

    ru_col = pick_col(lang_map, "Russian")
    en_col = pick_col(lang_map, "English")
    zh_col = pick_col(lang_map, "Chinese")

    total = len(data)
    print(f"✅ 数据载入成功：共 {total} 条。俄文列={ru_col}，英文列={en_col}，中文列={zh_col}")
    print(f"📘 name_map 中有 {len(name_map)} 条专名映射。")

    limiter = RateLimiter(RPM, TPM)

    tasks = []
    for key, row in data.items():
        if len(row) <= max(ru_col, en_col, zh_col):
            continue
        ru, en, cn = row[ru_col], row[en_col], row[zh_col]
        if not cn:
            continue
        if any(k in ru or k in en for k in name_map.keys()):
            tasks.append((key, ru, en, cn))
    print(f"📦 待修正句子数：{len(tasks)}（仅检测英文/俄文含专名行）")

    start_time = time.time()

    async def worker(idx, key, ru, en, cn):
        global PROCESSED, MAX_CONCURRENCY, RATE_LIMIT_HITS
        try:
            await limiter.acquire(cn)
            prompt = build_prompt(ru, en, cn, name_map)
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a careful localization QA assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                timeout=30,
            )
            out = clean_output(resp.choices[0].message.content)
        except Exception as e:
            if "RateLimit" in str(e) or "429" in str(e):
                RATE_LIMIT_HITS += 1
                if RATE_LIMIT_HITS % 10 == 0:
                    MAX_CONCURRENCY = max(5, MAX_CONCURRENCY // 2)
                    print(f"⚙️ 降速：并发={MAX_CONCURRENCY}")
                await asyncio.sleep(10)
                out = cn
            else:
                print(f"⚠️ 第{idx}条出错：{e}")
                out = cn

        async with LOCK:
            data[key][zh_col] = out
            PROCESSED += 1
            if PROCESSED % 100 == 0:
                elapsed = time.time() - start_time
                speed = PROCESSED / elapsed
                pct = PROCESSED / len(tasks)
                eta = (len(tasks) - PROCESSED) / max(1, speed)
                done = math.floor(pct * 25)
                bar = "█" * done + "-" * (25 - done)
                print(f"⏳ [{bar}] {PROCESSED}/{len(tasks)} | ETA: {eta:0.0f}s | Speed: {speed:.1f}/s")
            if PROCESSED % 500 == 0:
                save_json(OUTPUT_PATH, data)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async def sem_worker(i, key, r, e, c):
        async with sem:
            await worker(i, key, r, e, c)

    await asyncio.gather(*[sem_worker(i, k, r, e, c) for i, (k, r, e, c) in enumerate(tasks, 1)])

    save_json(OUTPUT_PATH, data)
    print(f"\n🎉 统一完成，共修正 {PROCESSED} 条。结果已保存至 {OUTPUT_PATH}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

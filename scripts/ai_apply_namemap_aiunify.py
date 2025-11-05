# scripts/ai_apply_namemap_unify.py
import os
import json
import re
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from openai import AsyncOpenAI

# ========== 环境加载 ==========
load_dotenv()

DATA_PATH = Path("output/language_dict_mcname_fixed.json")
LANG_MAP_PATH = Path("data/language_map.json")
NAME_MAP_PATH = Path("data/name_map.json")
OUTPUT_PATH = Path("output/language_dict_namemap_applied.json")
REPORT_PATH = Path("output/namemap_apply_report.txt")

# 模型与限速配置
DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "deepseek").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("MODEL", "gpt-4o-mini")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3.2-Exp")

RPM = int(os.getenv("RPM", "700"))
TPM = int(os.getenv("TPM", "80000"))
ASYNC_CONCURRENCY = int(os.getenv("ASYNC_CONCURRENCY", "30"))
BATCH_FLUSH = int(os.getenv("BATCH_FLUSH", "200"))
PRINT_EVERY = int(os.getenv("PRINT_EVERY", "50"))

# ========== 工具函数 ==========
def load_json(path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

# ========== 限速器 ==========
class RateLimiter:
    def __init__(self, rpm, tpm):
        self.rpm = rpm
        self.tpm = tpm
        self.reset_time = time.monotonic()
        self.req = 0
        self.tok = 0
        self.lock = asyncio.Lock()

    def _maybe_reset(self):
        now = time.monotonic()
        if now - self.reset_time >= 60:
            self.reset_time = now
            self.req = 0
            self.tok = 0

    def _estimate_tokens(self, text):
        return max(8, int(len(text) / 3.5) + 50)

    async def acquire(self, text):
        est = self._estimate_tokens(text)
        while True:
            async with self.lock:
                self._maybe_reset()
                if self.req + 1 <= self.rpm and self.tok + est <= self.tpm:
                    self.req += 1
                    self.tok += est
                    return
                wait = max(0.1, 60 - (time.monotonic() - self.reset_time))
            await asyncio.sleep(wait)

# ========== 客户端 ==========
@asynccontextmanager
async def build_async_client(provider: str):
    if provider == "deepseek":
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        model = DEEPSEEK_MODEL
        yield client, model
        await client.close()
    else:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        model = OPENAI_MODEL
        yield client, model
        await client.close()

def choose_provider():
    print("\n请选择引擎（1=ChatGPT，2=DeepSeek）：", end="")
    choice = input().strip()
    if choice == "1":
        print("🧠 使用 CHATGPT 模型引擎")
        return "chatgpt"
    print("🧠 使用 DEEPSEEK 模型引擎")
    return "deepseek"

# ========== Prompt 构建 ==========
def build_prompt(ru_text, en_text, zh_text, name_map):
    map_text = "\n".join([f"{k}: {v}" for k, v in name_map.items()])
    return f"""
你是游戏本地化编辑，任务是统一专有名词译名。

以下是三语对照的文本：
---
俄文: {ru_text}
英文: {en_text}
中文: {zh_text}
---

已知统一专名表（不可更改）：
{map_text}

请检查中文句子中的专名是否存在译法不统一、遗漏或混乱的情况。
如果需要修正，请用统一译法替换错误的部分，使整体流畅自然。
只返回修正后的中文译文（不得添加解释、括号、说明或引号）。
""".strip()

# ========== 清洗函数 ==========
def clean_output(txt: str):
    if not txt:
        return ""
    txt = txt.strip().strip("`").strip()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    return lines[0] if lines else ""

# ========== 主逻辑 ==========
async def main_async():
    provider = choose_provider()
    data = load_json(DATA_PATH)
    lang_map = load_json(LANG_MAP_PATH)
    name_map = load_json(NAME_MAP_PATH)

    total = len(data)
    print(f"✅ 数据载入成功：共 {total} 条。")

    # 定位列
    ru_col = next((int(k) for k, v in lang_map.items() if "Russian" in v), 1)
    en_col = next((int(k) for k, v in lang_map.items() if "English" in v), 2)
    zh_col = next((int(k) for k, v in lang_map.items() if "Chinese" in v), 5)
    print(f"俄文列={ru_col}，英文列={en_col}，中文列={zh_col}")
    print(f"📘 name_map 中有 {len(name_map)} 条专名映射。")

    # 检测候选
    candidates = []
    for k, row in data.items():
        if len(row) <= max(ru_col, en_col, zh_col):
            continue
        ru, en, zh = row[ru_col], row[en_col], row[zh_col]
        if not zh.strip():
            continue
        for n in name_map.keys():
            if n in ru or n in en:
                candidates.append((k, ru, en, zh))
                break

    print(f"📦 待修正句子数：{len(candidates)}（仅检测英文/俄文含专名行）")

    limiter = RateLimiter(RPM, TPM)
    sem = asyncio.Semaphore(ASYNC_CONCURRENCY)
    report_lines = []
    processed = 0

    async with build_async_client(provider) as (client, model):
        async def worker(i, key, ru, en, zh):
            nonlocal processed
            try:
                await limiter.acquire(zh)
                sys_msg = "You are a professional localization QA editor. Output translation only."
                user_msg = build_prompt(ru, en, zh, name_map)
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.4,
                    timeout=30,
                )
                out = clean_output(resp.choices[0].message.content)
                if out and out != zh:
                    data[key][zh_col] = out
                    report_lines.append(f"【{key}】修正:\n原:{zh}\n新:{out}\n")
            except Exception as e:
                if "429" in str(e):
                    await asyncio.sleep(5)
                else:
                    report_lines.append(f"⚠️ 第{i}条出错：{type(e).__name__} → {e}")
            processed += 1
            if processed % PRINT_EVERY == 0:
                print(f"⏳ [{processed}/{len(candidates)}] {en[:40]} → {data[key][zh_col][:40]}")
            if processed % BATCH_FLUSH == 0:
                save_json(OUTPUT_PATH, data)
                print("💾 自动保存进度...")

        await asyncio.gather(*(worker(i, k, r, e, c) for i, (k, r, e, c) in enumerate(candidates, 1)))

    save_json(OUTPUT_PATH, data)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n✅ 修正完成，共 {processed} 条。")
    print(f"📘 报告: {REPORT_PATH}")
    print(f"📁 输出文件: {OUTPUT_PATH}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

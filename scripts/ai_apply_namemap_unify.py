import os
import json
import time
import asyncio
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Dict

# ================== 环境配置 ==================
load_dotenv()

DATA_PATH = Path("output/language_dict_mcsurname_fixed.json")
LANG_MAP_PATH = Path("data/language_map.json")
NAME_MAP_PATH = Path("data/name_map.json")
OUTPUT_PATH = Path("output/language_dict_namemap_applied.json")
CACHE_PATH = Path("cache/namemap_apply_cache.json")

# ================== 模型配置 ==================
DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "").strip().lower()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("MODEL", "gpt-4o-mini")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASEURL = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn")
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")

# ================== 性能参数 ==================
ASYNC_CONCURRENCY = 60
BATCH_FLUSH = 500
PRINT_EVERY = 50
REQUEST_TIMEOUT = 40
SAVE_LOCK = asyncio.Lock()

# ================== 工具函数 ==================
def load_json(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

async def async_save_json(path: Path, data):
    """异步安全写入"""
    async with SAVE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if path.exists():
                os.remove(path)
            os.replace(tmp, path)
        except Exception as e:
            print(f"⚠️ 写入 {path.name} 失败：{e}")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def pick_col_by_lang(lang_map: Dict[str, str], label_contains: str) -> int:
    for k, v in lang_map.items():
        if v and label_contains.lower() in v.lower():
            return int(k)
    return -1

# ================== 模型客户端 ==================
@asynccontextmanager
async def build_async_client(provider: str):
    from openai import AsyncOpenAI
    if provider == "deepseek":
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASEURL)
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
    if choice == "2" or (not choice and DEFAULT_PROVIDER == "deepseek"):
        provider = "deepseek"
    else:
        provider = "chatgpt"
    print(f"🧠 使用 {provider.upper()} 模型引擎")
    return provider

# ================== Prompt ==================
def build_prompt(russian: str, english: str, chinese: str, name_map: Dict[str, str]) -> str:
    name_list = ", ".join([f"{k}:{v}" for k, v in name_map.items()])
    return f"""
你是一名资深的本地化编辑，负责成人视觉小说的中文译文一致性修正。
请使用映射表中的专名，统一下列文本的译名，保持自然、前后一致、符合中文语境。

---
映射表（节选）：
{name_list[:4000]}

俄文原句：{russian}
英文原句：{english}
当前译文：{chinese}
---

规则：
1. 输出仅为修正后的中文句子。
2. 不添加任何解释或标点。
3. 若无需修改，原样输出。
4. 优化人名、地名、校名等专名译法。
""".strip()

def clean_model_output(s: str) -> str:
    if not s:
        return ""
    lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
    return lines[0] if lines else ""

# ================== 实时进度条 ==================
def progress_bar(current: int, total: int, start_time: float):
    bar_len = 30
    filled_len = int(bar_len * current / total)
    bar = "█" * filled_len + "-" * (bar_len - filled_len)
    elapsed = time.monotonic() - start_time
    speed = current / elapsed if elapsed > 0 else 0
    remaining = (total - current) / speed if speed > 0 else 0
    eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
    print(f"\r⏳ [{bar}] {current}/{total} | ETA: {eta} | Speed: {speed:.1f}/s", end="", flush=True)

# ================== 主逻辑 ==================
async def main_async():
    provider = choose_provider()

    # 加载文件
    data = load_json(DATA_PATH)
    lang_map = load_json(LANG_MAP_PATH)
    name_map = load_json(NAME_MAP_PATH, default={})
    cache = load_json(CACHE_PATH, default={})
    total = len(data)

    ru_col = pick_col_by_lang(lang_map, "Russian")
    en_col = pick_col_by_lang(lang_map, "English")
    zh_col = pick_col_by_lang(lang_map, "Chinese")

    if min(ru_col, en_col, zh_col) < 0:
        print("❌ language_map.json 未检测到完整的 Russian / English / Chinese 列")
        return

    print(f"✅ 数据载入成功：共 {total} 条。俄文列={ru_col}，英文列={en_col}，中文列={zh_col}")
    print(f"📘 name_map 中有 {len(name_map)} 条专名映射。")

    # 仅取含专名的行
    keywords = list(name_map.keys())
    print(f"🔍 检测专名：共 {len(keywords)} 个")

    tasks = []
    for i, (key, row) in enumerate(data.items(), 1):
        if len(row) <= max(ru_col, en_col, zh_col):
            continue
        ru = str(row[ru_col] or "")
        en = str(row[en_col] or "")
        cn = str(row[zh_col] or "")
        if not cn.strip():
            continue
        if any(k in ru or k in en for k in keywords):
            tasks.append((i, key, ru, en, cn))

    print(f"📦 待修正句子数：{len(tasks)}（仅检测英文/俄文含专名行）")

    async with build_async_client(provider) as (client, model):
        sem = asyncio.Semaphore(ASYNC_CONCURRENCY)
        modified = 0
        last_flush_time = time.monotonic()
        start_time = time.monotonic()

        async def worker(i: int, key: str, ru: str, en: str, cn: str):
            nonlocal modified, last_flush_time
            async with sem:
                ck = sha1(f"{ru}|{en}|{cn}")
                if ck in cache:
                    new_cn = cache[ck]
                else:
                    prompt = build_prompt(ru, en, cn, name_map)
                    try:
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a localization editor. Output corrected Chinese only."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.4,
                            timeout=REQUEST_TIMEOUT
                        )
                        new_cn = clean_model_output(resp.choices[0].message.content)
                        cache[ck] = new_cn
                    except Exception as e:
                        print(f"\n⚠️ 第{i}条出错：{type(e).__name__} → {e}")
                        new_cn = cn

                if new_cn and new_cn != cn:
                    modified += 1
                    data[key][zh_col] = new_cn
                    print(f"\n🔧 修正 {i}/{len(tasks)}：{cn[:40]} → {new_cn[:40]}")

                progress_bar(i, len(tasks), start_time)

                if i % BATCH_FLUSH == 0 or (time.monotonic() - last_flush_time > 90):
                    await async_save_json(OUTPUT_PATH, data)
                    await async_save_json(CACHE_PATH, cache)
                    last_flush_time = time.monotonic()
                    print(f"\n💾 自动保存进度 ({i}/{len(tasks)})")

        # 异步执行
        await asyncio.gather(*[worker(*task) for task in tasks])

        # 最终保存
        await async_save_json(OUTPUT_PATH, data)
        await async_save_json(CACHE_PATH, cache)

    print(f"\n🎉 修正完成，共修改 {modified} 条。结果已保存至 {OUTPUT_PATH}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

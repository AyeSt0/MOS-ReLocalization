# scripts/ai_namemap_unify_pipeline_v2.py
# -*- coding: utf-8 -*-
import os
import re
import json
import time
import asyncio
import hashlib
import difflib
import signal
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# ========= 环境 =========
load_dotenv()

# —— 路径（按你的项目结构）——
DATA_PATH       = Path("output/language_dict_mcsurname_fixed.json")  # 你的最新中文列文件
LANG_MAP_PATH   = Path("data/language_map.json")
NAME_MAP_PATH   = Path("data/name_map.json")

CANDIDATES_PATH   = Path("output/name_ai_candidates.json")
INCONSISTENT_PATH = Path("output/name_ai_inconsistent.json")
FIX_LOG_PATH      = Path("output/name_ai_fixes.jsonl")
OUTPUT_PATH       = Path("output/language_dict_name_unified.json")
REPORT_PATH       = Path("output/name_unify_report.txt")

CACHE_PATH      = Path("cache/ai_namemap_cache.json")

# —— 并发 / 限速 / 落盘 ——
ASYNC_CONCURRENCY = int(os.getenv("ASYNC_CONCURRENCY", "30"))    # 并发
REQUEST_TIMEOUT   = int(os.getenv("REQUEST_TIMEOUT", "30"))
BATCH_FLUSH       = int(os.getenv("BATCH_FLUSH", "500"))
PRINT_EVERY       = int(os.getenv("PRINT_EVERY", "100"))

# —— 引擎选择 & 模型（与你现有 env 保持一致）——
DEFAULT_PROVIDER  = os.getenv("MODEL_PROVIDER", "").strip().lower()  # chatgpt / deepseek
# ChatGPT
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL      = os.getenv("MODEL", "gpt-4o-mini").strip()
# DeepSeek（OpenAI兼容接口）
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASEURL  = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn").strip()
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3.2-Exp").strip()

# —— 限速（DeepSeek：RPM=1000 / TPM=100000，已在 .env 里给了 700/80000 更稳）——
RPM               = int(os.getenv("RPM", "700"))
TPM               = int(os.getenv("TPM", "80000"))

# —— 仅处理含英/俄字的正则（检测中文里夹杂）——
HAS_LATIN_OR_CYR  = re.compile(r"[A-Za-zА-Яа-яЁё]")

stop_requested = False

def handle_signal(signum, frame):
    global stop_requested
    stop_requested = True
    print("\n⚠️ 捕获到中断信号，将安全落盘并退出…")
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ========= I/O =========
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

def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def pick_col_by_lang(lang_map: Dict[str, str], label_contains: str) -> int:
    for k, v in lang_map.items():
        if v and label_contains.lower() in v.lower():
            return int(k)
    return -1

# ========= 进度工具 =========
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ========= 速率限制器 =========
class RateLimiter:
    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self._lock = asyncio.Lock()
        self._win_start = time.monotonic()
        self._req_used = 0
        self._tok_used = 0
        self._decay = 1.0  # 动态降速因子（命中 429 时调低）

    def _maybe_reset(self):
        now = time.monotonic()
        if now - self._win_start >= 60.0:
            self._win_start = now
            self._req_used = 0
            self._tok_used = 0

    def _estimate_tokens(self, ru_text: str, en_text: str, zh_text: str, std_map_json: str) -> int:
        base = len(ru_text) + len(en_text) + len(zh_text) + len(std_map_json) + 300
        return max(10, int(base / 3.5))

    async def acquire(self, ru_text: str, en_text: str, zh_text: str, std_map_json: str):
        est = self._estimate_tokens(ru_text, en_text, zh_text, std_map_json)
        while True:
            async with self._lock:
                self._maybe_reset()
                rpm_ok = (self._req_used + 1) <= int(self.rpm * self._decay)
                tpm_ok = (self._tok_used + est) <= int(self.tpm * self._decay)
                if rpm_ok and tpm_ok:
                    self._req_used += 1
                    self._tok_used += est
                    return
                wait = max(0.0, 60.0 - (time.monotonic() - self._win_start))
            await asyncio.sleep(min(1.0, wait))

    async def cool_down(self):
        async with self._lock:
            self._decay = max(0.2, self._decay * 0.85)  # 命中 429 就更保守
        await asyncio.sleep(2.0)

    async def relax(self):
        async with self._lock:
            self._decay = min(1.0, self._decay + 0.05)

# ========= 异步客户端 =========
@asynccontextmanager
async def build_async_client(provider: str):
    from openai import AsyncOpenAI
    if provider == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY")
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASEURL)
        model = DEEPSEEK_MODEL
        try:
            yield client, model
        finally:
            await client.close()
    else:
        if not OPENAI_API_KEY:
            raise RuntimeError("缺少 OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        model = OPENAI_MODEL
        try:
            yield client, model
        finally:
            await client.close()

def choose_provider() -> str:
    print("\n请选择引擎（1=ChatGPT，2=DeepSeek）：", end="")
    choice = input().strip()
    if choice == "2" or (not choice and DEFAULT_PROVIDER == "deepseek"):
        print("🧠 使用 DEEPSEEK 模型引擎")
        return "deepseek"
    print("🧠 使用 CHATGPT 模型引擎")
    return "chatgpt"

# ========= 缓存 =========
class SimpleCache:
    def __init__(self, path: Path):
        self.path = path
        self._data = load_json(path, default={})
        self._lock = asyncio.Lock()

    def _key(self, provider: str, model: str, ru: str, en: str, zh: str, std_map_json: str) -> str:
        raw = f"{provider}|{model}|{ru}|{en}|{zh}|{std_map_json}"
        return sha1(raw)

    async def get(self, provider, model, ru, en, zh, std_json):
        k = self._key(provider, model, ru, en, zh, std_json)
        async with self._lock:
            return self._data.get(k, "")

    async def set(self, provider, model, ru, en, zh, std_json, val: str):
        if not val:
            return
        k = self._key(provider, model, ru, en, zh, std_json)
        async with self._lock:
            if k not in self._data:
                self._data[k] = val

    async def flush(self):
        async with self._lock:
            save_json(self.path, self._data)

# ========= 文本清洗 & 占位符修正 =========
def clean_output(s: str) -> str:
    if not s: return ""
    s = s.strip().strip("`").strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[0] if lines else ""

def fix_placeholders(s: str) -> str:
    # 把各种括号形式的 mcname 统一成半角方括号 [mcname]
    s = re.sub(r"[{\[（【]\s*mcname\s*[}\]）】]", "[mcname]", s, flags=re.IGNORECASE)
    return s

# ========= 阶段 1：抽取候选 =========
def phase_extract(data: Dict[str, List[str]], lang_map: Dict[str, str], name_map: Dict[str, str]) -> List[Tuple[str, str, str, str]]:
    ru_col = pick_col_by_lang(lang_map, "Russian")
    en_col = pick_col_by_lang(lang_map, "English")
    zh_col = pick_col_by_lang(lang_map, "Chinese")
    if min(ru_col, en_col, zh_col) < 0:
        raise RuntimeError("language_map.json 未检测到 Russian / English / Chinese 列")

    keys = list(name_map.keys())
    candidates = []
    for key, row in data.items():
        if len(row) <= max(ru_col, en_col, zh_col):
            continue
        ru, en, zh = row[ru_col] or "", row[en_col] or "", row[zh_col] or ""
        # 命中专名键（俄或英）
        if any(k in ru or k in en for k in keys):
            candidates.append((key, ru, en, zh))

    save_json(CANDIDATES_PATH, [{"key": k, "ru": r, "en": e, "zh": z} for k, r, e, z in candidates])
    print(f"🧲 候选抽取完成：{len(candidates)} 条 → {CANDIDATES_PATH}")
    return candidates

# ========= 阶段 2：本地模糊检测 =========
def zh_needs_fix(zh: str, std_vals: List[str]) -> bool:
    # 如果标准译名已经在句中，通常不需要修
    if any(std in zh for std in std_vals):
        return False
    # 含英/俄字符，可能是没替干净
    if HAS_LATIN_OR_CYR.search(zh):
        return True
    # 与任一标准名相似度较高，说明可能是变体（但未命中标准本身）
    for std in std_vals:
        ratio = difflib.SequenceMatcher(None, zh, std).ratio()
        if ratio > 0.35:  # 句级粗判阈值
            return True
    return False

def phase_detect(candidates: List[Tuple[str, str, str, str]], name_map: Dict[str, str]) -> List[Tuple[str, str, str, str, Dict[str, str]]]:
    # 针对每条候选，汇总其“行内命中的标准表”（可能命中多个键 → 可能需多名统一）
    result = []
    for key, ru, en, zh in candidates:
        std_pairs = {}
        for k, v in name_map.items():
            if (k in ru) or (k in en):
                std_pairs[k] = v
        if not std_pairs:
            continue
        std_values = list(set(std_pairs.values()))
        if zh_needs_fix(zh, std_values):
            result.append((key, ru, en, zh, std_pairs))

    # 保存检测清单
    payload = [{"key": k, "ru": r, "en": e, "zh": z, "std_pairs": sp} for k, r, e, z, sp in result]
    save_json(INCONSISTENT_PATH, payload)
    print(f"🔍 本地检测完成：疑似不统一 {len(result)} 条 → {INCONSISTENT_PATH}")
    return result

# ========= AI 修正 =========
def build_fix_prompt(ru: str, en: str, zh: str, std_pairs: Dict[str, str]) -> str:
    # 只替换专名；不改其他内容；输出完整修正中文
    std_json = json.dumps(std_pairs, ensure_ascii=False)
    return (
        "你是中文本地化一致性修正助手。请仅在下列中文句子中，将专有名词统一为给定的标准译名。\n"
        "要求：\n"
        "1) 只替换对应专名的各种变体为标准译名；不改变其他文字、语序与标点。\n"
        "2) 保留占位符与标签（如 [mcname]、{变量}、<tag> 等）原样。\n"
        "3) 不要输出解释或注释，只输出最后的完整中文句子。\n\n"
        f"专名-标准译名表：{std_json}\n\n"
        f"俄文：{ru}\n"
        f"英文：{en}\n"
        f"中文原句：\n{zh}\n"
        "输出："
    )

async def ai_fix_once(client, model: str, limiter: RateLimiter, provider: str,
                      ru: str, en: str, zh: str, std_pairs: Dict[str, str]) -> str:
    std_json = json.dumps(std_pairs, ensure_ascii=False)
    await limiter.acquire(ru, en, zh, std_json)

    sys_msg  = "You are a precise Chinese localization consistency assistant. Output only the corrected Chinese line."
    user_msg = build_fix_prompt(ru, en, zh, std_pairs)

    for attempt in range(1, 6):
        if stop_requested:
            return ""
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                timeout=REQUEST_TIMEOUT,
            )
            out = clean_output(resp.choices[0].message.content or "")
            out = fix_placeholders(out)
            await limiter.relax()
            return out
        except Exception as e:
            ename = type(e).__name__
            estr = str(e)
            if "429" in estr or "RateLimit" in ename:
                print(f"⚠️ RateLimit：冷却中（attempt {attempt})")
                await limiter.cool_down()
                await asyncio.sleep(min(60.0, 2 ** attempt))
                continue
            wait = min(30.0, 1.8 ** attempt)
            print(f"⏳ 重试 {attempt}/5：{ename}，{wait:.1f}s 后再试…")
            await asyncio.sleep(wait)
    return ""

# ========= 阶段 3：AI 批量修正并落盘 =========
async def phase_fix(data: Dict[str, List[str]], lang_map: Dict[str, str],
                    inconsistent: List[Tuple[str, str, str, str, Dict[str, str]]],
                    provider: str):

    zh_col = pick_col_by_lang(lang_map, "Chinese")
    if zh_col < 0:
        raise RuntimeError("language_map.json 未检测到 Chinese 列")

    limiter = RateLimiter(RPM, TPM)
    cache   = SimpleCache(CACHE_PATH)
    fixed = 0
    total = len(inconsistent)
    last_flush = 0

    async with build_async_client(provider) as (client, model):
        sem = asyncio.Semaphore(ASYNC_CONCURRENCY)

        async def worker(i: int, key: str, ru: str, en: str, zh: str, std_pairs: Dict[str, str]):
            nonlocal fixed, last_flush
            std_json = json.dumps(std_pairs, ensure_ascii=False)
            cached = await cache.get(provider, model, ru, en, zh, std_json)
            if cached:
                new_zh = cached
            else:
                async with sem:
                    new_zh = await ai_fix_once(client, model, limiter, provider, ru, en, zh, std_pairs)
                await cache.set(provider, model, ru, en, zh, std_json, new_zh)

            # 写回（仅中文列）
            if new_zh and new_zh != zh:
                data[key][zh_col] = new_zh
                fixed += 1
                append_jsonl(FIX_LOG_PATH, {
                    "idx": i, "key": key, "ru": ru, "en": en,
                    "zh_old": zh, "zh_new": new_zh, "std_pairs": std_pairs
                })

            # 进度
            if i % PRINT_EVERY == 0:
                preview_src = (zh[:28] + "…") if len(zh) > 28 else zh
                preview_new = (new_zh[:28] + "…") if new_zh and len(new_zh) > 28 else (new_zh or "")
                print(f"⏳ [{i}/{total}] {preview_src} → {preview_new}")

            # 批量落盘
            if i - last_flush >= BATCH_FLUSH:
                save_json(OUTPUT_PATH, data)
                await cache.flush()
                last_flush = i
                print("💾 自动保存进度…")

        tasks = [worker(i, k, ru, en, zh, sp) for i, (k, ru, en, zh, sp) in enumerate(inconsistent, 1)]
        try:
            for chunk_start in range(0, len(tasks), 10000):
                chunk = tasks[chunk_start:chunk_start+10000]
                await asyncio.gather(*chunk)
                if stop_requested:
                    break
        finally:
            save_json(OUTPUT_PATH, data)
            await cache.flush()

    print(f"✅ AI 修正完成：共替换 {fixed}/{total} 条 → {OUTPUT_PATH}")
    return fixed

# ========= 主入口（多阶段） =========
def main():
    parser = argparse.ArgumentParser(description="专名统一管线（基于 name_map，三阶段：extract / detect / fix / all）")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["extract", "detect", "fix", "all"],
                        help="选择执行阶段")
    args = parser.parse_args()

    provider = choose_provider()

    data     = load_json(DATA_PATH, default={})
    lang_map = load_json(LANG_MAP_PATH, default={})
    name_map = load_json(NAME_MAP_PATH, default={})

    total = len(data)
    ru_col = pick_col_by_lang(lang_map, "Russian")
    en_col = pick_col_by_lang(lang_map, "English")
    zh_col = pick_col_by_lang(lang_map, "Chinese")

    if min(ru_col, en_col, zh_col) < 0:
        print("❌ 未检测到完整的三语列（Russian/English/Chinese），请检查 language_map.json")
        return

    print(f"✅ 数据载入成功：共 {total} 条。俄文列={ru_col}，英文列={en_col}，中文列={zh_col}")
    print(f"📘 name_map 中有 {len(name_map)} 条专名映射。")
    print(f"🔍 检测专名：共 {len(set(name_map.keys()))} 个")

    if args.phase in ("extract", "all"):
        candidates = phase_extract(data, lang_map, name_map)
    else:
        # 若不是 extract，从文件复用
        if CANDIDATES_PATH.exists():
            cjson = load_json(CANDIDATES_PATH, default=[])
            candidates = [(x["key"], x["ru"], x["en"], x["zh"]) for x in cjson]
        else:
            candidates = phase_extract(data, lang_map, name_map)

    if args.phase in ("detect", "all"):
        inconsistent = phase_detect(candidates, name_map)
    else:
        # 若不是 detect，从文件复用
        if INCONSISTENT_PATH.exists():
            incjson = load_json(INCONSISTENT_PATH, default=[])
            inconsistent = [(x["key"], x["ru"], x["en"], x["zh"], x["std_pairs"]) for x in incjson]
        else:
            inconsistent = phase_detect(candidates, name_map)

    print(f"📦 待修正句子数：{len(inconsistent)}（仅检测英文/俄文含专名行）")

    if args.phase in ("fix", "all") and inconsistent:
        asyncio.run(phase_fix(data, lang_map, inconsistent, provider))
        # 最终报告
        report = [
            f"总记录数：{total}",
            f"候选（含专名）行：{len(candidates)}",
            f"疑似不统一行：{len(inconsistent)}",
            f"输出：{OUTPUT_PATH}",
            f"修复日志：{FIX_LOG_PATH}"
        ]
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
        print(f"📘 报告保存至: {REPORT_PATH}")
    else:
        # 非 fix 阶段也落库一份未改动的数据，便于比对
        save_json(OUTPUT_PATH, data)
        print(f"📝 已保存当前数据快照（未改动）：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()

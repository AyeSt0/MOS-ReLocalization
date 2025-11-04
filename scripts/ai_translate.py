# scripts/ai_translate.py
import os
import sys
import json
import time
import signal
import re
import asyncio
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Tuple, List
from contextlib import asynccontextmanager

# ================== 环境与常量 ==================
load_dotenv()

# 路径
DATA_PATH        = Path("data/language_dict.json")
LANG_MAP_PATH    = Path("data/language_map.json")
NAME_MAP_PATH    = Path("data/name_map.json")
OUTPUT_PATH      = Path("output/language_dict_translated.json")
CACHE_PATH       = Path("cache/cache.json")

# 并发 / 限速 / 落盘
ASYNC_CONCURRENCY = int(os.getenv("ASYNC_CONCURRENCY", "100"))   # 并发度（DeepSeek 可拉高）
REQUEST_TIMEOUT   = int(os.getenv("REQUEST_TIMEOUT", "30"))
BATCH_FLUSH       = int(os.getenv("BATCH_FLUSH", "200"))         # 多少条落盘一次
PRINT_EVERY       = int(os.getenv("PRINT_EVERY", "50"))          # 多少条打印一次简要进度

# 引擎选择
DEFAULT_PROVIDER  = os.getenv("MODEL_PROVIDER", "").strip().lower()  # chatgpt / deepseek
# ChatGPT
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL      = os.getenv("MODEL", "gpt-4o-mini").strip()
# DeepSeek（OpenAI兼容）
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASEURL  = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn").strip()
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1").strip()

# 限速参数（可在 .env 覆盖）
# DeepSeek 官方：RPM=1000；TPM=100000
RPM               = int(os.getenv("RPM", "1000"))
TPM               = int(os.getenv("TPM", "100000"))

# 始终启用专名预替换（但翻译逻辑已优化：仅已登记的专名强制；其余根据上下文）
ALWAYS_PREMAP_NAMES = True

# Honorific/大写停用词（避免把敬称当成专名）
HONORIFICS = {
    "Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Professor", "Coach",
    "Sir", "Madam", "Lady", "Lord", "Captain", "Principal", "Dean",
    "I", "The", "A", "An", "OK", "TV", "USA", "EU", "ID"
}

stop_requested = False

# ================== 工具函数 ==================
def load_json(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
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

def non_empty_count_of_col(data: dict, col_idx: int) -> int:
    cnt = 0
    for arr in data.values():
        if len(arr) > col_idx and str(arr[col_idx]).strip():
            cnt += 1
    return cnt

def pick_col_by_lang(lang_map: Dict[str, str], label_contains: str) -> int:
    for k, v in lang_map.items():
        if v and label_contains.lower() in v.lower():
            return int(k)
    return -1

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ================== 终止信号 ==================
def handle_signal(signum, frame):
    global stop_requested
    stop_requested = True
    print("\n⚠️ 收到中断，准备落盘并安全退出…")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ================== Prompt ==================
# 成人向本地化：俄文情绪/语气 + 英文语义 → 流畅、撩人、带身体感的中文
def build_prompt(ru_text: str, en_text: str, target_lang_label: str) -> str:
    return f"""You are an experienced localization writer for adult visual novels.
Your job: translate & localize the following Russian and English lines into seductive,
natural, and immersive Chinese ({target_lang_label}) for the game "MILFs of Sunville".

Use English for literal meaning. Use Russian for mood and sensual nuance.
Localize for Chinese players: smooth rhythm, emotional tension, and erotic subtext — not literal.

Rules:
- Keep sexual details explicit but tasteful; no censorship or summary.
- Keep placeholders exactly ({{{{mcname}}}}, [var], <tag>).
- Translate person/place names only if they already have a fixed Chinese form in the given text.
- Output ONLY the final translation — no explanations, no quotes.

Russian: {ru_text}
English: {en_text}""".strip()

# ================== 文本清洗 ==================
def clean_model_output(s: str) -> str:
    if not s:
        return ""
    s = s.strip().strip("`").strip()
    # 只取第一行非空内容（避免模型多余解释）
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[0] if lines else ""

# 变量/标签保护（判定是否包含敏感占位）
def has_protected_tokens(s: str) -> bool:
    return any(tok in s for tok in ("{", "}", "[", "]", "<", ">"))

# ================== 名称预替换（仅已登记专名） ==================
def apply_name_map_pre(src_text: str, name_map: Dict[str, str]) -> str:
    if not ALWAYS_PREMAP_NAMES or not name_map:
        return src_text
    # 长匹配优先，避免短词误替
    for k, v in sorted(name_map.items(), key=lambda kv: -len(kv[0])):
        if not k or not v:
            continue
        # 仅替换“完整词/短语”出现的情况，不动变量及标签内部
        if k in src_text and not has_protected_tokens(k):
            src_text = src_text.replace(k, v)
    return src_text

# ================== 速率限制器（RPM & TPM） ==================
class RateLimiter:
    """
    简单令牌桶：每分钟请求上限 RPM；每分钟令牌上限 TPM。
    请求到来时，如果超限则等待至下一窗口可用。
    """
    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self._lock = asyncio.Lock()
        self._window_start = time.monotonic()
        self._req_used = 0
        self._tok_used = 0

    def _maybe_reset(self):
        now = time.monotonic()
        if now - self._window_start >= 60.0:
            self._window_start = now
            self._req_used = 0
            self._tok_used = 0

    def _estimate_tokens(self, ru_text: str, en_text: str) -> int:
        # 近似估算 tokens：中英俄混合粗估（保守一点）
        # 假设 1 token ≈ 4 chars（英文），俄文更密集，取 3.5。统一取 3.5 更稳妥
        ln = len(ru_text) + len(en_text) + 180  # +prompt 开销粗估
        return max(8, int(ln / 3.5))

    async def acquire(self, ru_text: str, en_text: str):
        est = self._estimate_tokens(ru_text, en_text)
        while True:
            async with self._lock:
                self._maybe_reset()
                can_req = (self._req_used + 1) <= self.rpm
                can_tok = (self._tok_used + est) <= self.tpm
                if can_req and can_tok:
                    self._req_used += 1
                    self._tok_used += est
                    return  # 许可通过
                # 否则计算等待时间：直到 60s 窗口重置
                wait = max(0.0, 60.0 - (time.monotonic() - self._window_start))
            await asyncio.sleep(min(1.0, wait))

# ================== 客户端适配（异步） ==================
@asynccontextmanager
async def build_async_client(provider: str):
    from openai import AsyncOpenAI
    if provider == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置。")
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASEURL)
        model = DEEPSEEK_MODEL
        yield client, model
        await client.close()
    else:
        if not OPENAI_API_KEY:
            raise RuntimeError("缺少 OPENAI_API_KEY，请在 .env 中配置。")
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        model = OPENAI_MODEL
        yield client, model
        await client.close()

def choose_provider() -> str:
    print("\n请选择翻译引擎：")
    print("  1) ChatGPT (OpenAI)")
    print("  2) DeepSeek (OpenAI兼容, RPM=1000, TPM=100000)")
    default_hint = f"(默认: {DEFAULT_PROVIDER or 'ChatGPT'})"
    choice = input(f"👉 输入 1 或 2 {default_hint}：").strip()
    if choice == "2" or (not choice and DEFAULT_PROVIDER == "deepseek"):
        provider = "deepseek"
    else:
        provider = "chatgpt"
    print(f"🧠 使用 {('DeepSeek' if provider=='deepseek' else 'ChatGPT')} 模型引擎")
    return provider

# ================== 缓存 ==================
class SimpleCache:
    def __init__(self, path: Path):
        self.path = path
        self._data = load_json(path, default={})
        self._lock = asyncio.Lock()

    def _key(self, provider: str, model: str, tgt_label: str, ru_text: str, en_text: str) -> str:
        raw = f"{provider}|{model}|{tgt_label}|{ru_text}|{en_text}"
        return sha1(raw)

    async def get(self, provider: str, model: str, tgt_label: str, ru_text: str, en_text: str) -> str:
        k = self._key(provider, model, tgt_label, ru_text, en_text)
        async with self._lock:
            return self._data.get(k, "")

    async def set(self, provider: str, model: str, tgt_label: str, ru_text: str, en_text: str, value: str):
        if not value:
            return
        k = self._key(provider, model, tgt_label, ru_text, en_text)
        async with self._lock:
            if k not in self._data:
                self._data[k] = value

    async def flush(self):
        async with self._lock:
            save_json(self.path, self._data)

# ================== 翻译核心（异步） ==================
async def translate_once(client, model: str, limiter: RateLimiter, provider: str,
                         ru_text: str, en_text: str, target_lang_label: str) -> str:
    # 限速：先取许可
    await limiter.acquire(ru_text, en_text)

    # 生成消息
    sys_msg  = "You are a professional adult-visual-novel localization translator. Output translation only."
    user_msg = build_prompt(ru_text, en_text, target_lang_label)

    # 调用
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
                temperature=0.6,
                timeout=REQUEST_TIMEOUT,
            )
            out = resp.choices[0].message.content or ""
            return clean_model_output(out)
        # 在 translate_once 的 except Exception 块里加：
        except Exception as e:
            if "RateLimitError" in type(e).__name__ or "429" in str(e):
                cooldown = min(60.0, 2 ** attempt)
                print(f"⚠️ 全局冷却 {cooldown:.1f}s：RateLimitError（已触发防抖机制）")
                await asyncio.sleep(cooldown)
            else:
                wait = min(30.0, 1.8 ** attempt)
                print(f"⏳ 重试 {attempt}/5：{type(e).__name__}，{wait:.1f}s 后再试...")
                await asyncio.sleep(wait)
    return ""

# ================== 主流程 ==================
async def main_async():
    # 选择引擎
    provider = choose_provider()

    # 读数据
    data     = load_json(DATA_PATH, default={})
    lang_map = load_json(LANG_MAP_PATH, default={})
    name_map = load_json(NAME_MAP_PATH, default={})
    cache    = SimpleCache(CACHE_PATH)

    total = len(data)
    print(f"✅ 加载完成，共 {total} 条记录。\n")

    # 定位列：俄文、英文
    ru_col = pick_col_by_lang(lang_map, "Russian")
    en_col = pick_col_by_lang(lang_map, "English")
    if ru_col < 0 or en_col < 0:
        print("❌ 无法识别源语言或目标语言列（需要 Russian 与 English），请检查 language_map.json")
        return

    # 展示可翻译列（排除 META）
    print("可翻译列如下：")
    for k, v in sorted(lang_map.items(), key=lambda kv: int(kv[0])):
        if v == "META":
            continue
        col_idx = int(k)
        pct = (non_empty_count_of_col(data, col_idx) / total * 100.0) if total else 0.0
        print(f"  - 列 {k}: {v} ({pct:.1f}%)")

    # Unknown 列提示
    unknowns = [(int(k), v) for k, v in lang_map.items() if v == "Unknown"]
    if unknowns:
        print("\n🟡 检测到 Unknown 列，可选择创建新语言翻译：")
        for k, v in sorted(unknowns, key=lambda x: x[0]):
            pct = (non_empty_count_of_col(data, k) / total * 100.0) if total else 0.0
            print(f"  - 列 {k}: {v} ({pct:.1f}%)")

    # 选择目标列或新增
    raw = input("\n👉 请输入要进行本地化翻译的目标列号（留空以新增语言列）：").strip()
    if raw == "":
        want_col  = input("🆕 请输入要新增的列号（留空则追加到末尾）：").strip()
        lang_name = input("🆕 请输入新增列的语言名（例如 Chinese (Simplified Chinese)）：").strip() or "Unknown"
        max_col = max(map(int, lang_map.keys())) if lang_map else -1
        if want_col:
            tgt_col = int(want_col)
            if tgt_col > max_col + 1:
                for c in range(max_col + 1, tgt_col):
                    lang_map[str(c)] = "Unknown"
            lang_map[str(tgt_col)] = lang_name
        else:
            tgt_col = max_col + 1
            lang_map[str(tgt_col)] = lang_name
        save_json(LANG_MAP_PATH, lang_map)
        print(f"🆕 新增列 {tgt_col}: {lang_map[str(tgt_col)]}")
    else:
        tgt_col = int(raw)

    target_lang_label = lang_map.get(str(tgt_col), "Unknown")
    print(f"\n🌍 将从列 {ru_col}（Russian） + 列 {en_col}（English） 翻到 列 {tgt_col}（{target_lang_label}）")

    # 模式
    mode = input("\n选择模式：1=继续翻译（补空） / 2=强制翻译（清空重来）：").strip()
    if mode == "2":
        confirm = input("⚠️ 确认要清空该列的所有翻译吗？(y/n)：").strip().lower()
        if confirm == "y":
            for row in data.values():
                ensure_row_len(row, tgt_col)
                row[tgt_col] = ""
            save_json(DATA_PATH, data)
            print("🧹 已清空目标列。")
        else:
            print("已取消清空操作，转为继续翻译（补空）模式。")
            mode = "1"

    # 构建任务：只翻译“源有内容 且 目标为空”的行
    todo: List[Tuple[int, str, str, str]] = []
    index = 0
    for key, row in data.items():
        index += 1
        ensure_row_len(row, max(ru_col, en_col, tgt_col))
        ru_text = (row[ru_col] or "").strip()
        en_text = (row[en_col] or "").strip()
        tgt_text = (row[tgt_col] or "").strip()
        # 只有源文本存在才翻；继续模式：仅补空；强制时：上面已清空
        if ru_text or en_text:
            if mode == "1" and tgt_text:
                continue
            todo.append((index, key, ru_text, en_text))

    print(f"\n📦 待翻译: {len(todo)} 条。")

    if not todo:
        save_json(OUTPUT_PATH, data)
        print(f"🎉 翻译完成，结果已保存至 {OUTPUT_PATH}")
        return

    # 深/浅引擎限速（若使用 ChatGPT，可在 .env 单独配置 RPM/TPM；否则用默认）
    limiter = RateLimiter(RPM, TPM)

    processed = 0
    last_flush = 0

    # 异步客户端
    async with build_async_client(provider) as (client, model):
        sem = asyncio.Semaphore(ASYNC_CONCURRENCY)

        async def one_job(idx: int, key: str, ru_text: str, en_text: str):
            nonlocal processed, last_flush

            # 名称预替换（仅针对英文列，俄文列不替，避免破坏语境）
            en_pre = apply_name_map_pre(en_text, name_map) if en_text else en_text
            ru_pre = ru_text  # 俄文不预替

            # 缓存查询
            cached = await cache.get(provider, model, target_lang_label, ru_pre, en_pre)
            if cached:
                out = cached
            else:
                # 限速 + 调用
                async with sem:
                    out = await translate_once(client, model, limiter, provider, ru_pre, en_pre, target_lang_label)
                out = clean_model_output(out)
                await cache.set(provider, model, target_lang_label, ru_pre, en_pre, out)

            # 写回
            row = data[key]
            ensure_row_len(row, tgt_col)
            row[tgt_col] = out

            processed += 1
            if processed % PRINT_EVERY == 0:
                short_src = (en_text or ru_text)[:60].replace("\n", " ")
                short_out = (out or "")[:60].replace("\n", " ")
                print(f"🔄 进度 {processed}/{len(todo)} | 源: {short_src} | 译: {short_out}")

            # 批量落盘
            if processed - last_flush >= BATCH_FLUSH:
                save_json(OUTPUT_PATH, data)
                await cache.flush()
                last_flush = processed
                print(f"💾 自动保存进度 -> {OUTPUT_PATH}")

        tasks = [one_job(idx, key, ru, en) for (idx, key, ru, en) in todo]

        try:
            for chunk_start in range(0, len(tasks), 10000):
                # 分块并发，避免过多任务一次性注入事件循环
                chunk = tasks[chunk_start:chunk_start+10000]
                await asyncio.gather(*chunk)
                if stop_requested:
                    break
        finally:
            # 最终落盘
            save_json(OUTPUT_PATH, data)
            await cache.flush()

    print(f"\n🎉 翻译完成，结果已保存至 {OUTPUT_PATH}")

def choose_provider_cli_default() -> str:
    # 供外部脚本调用（保持行为一致）
    return choose_provider()

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

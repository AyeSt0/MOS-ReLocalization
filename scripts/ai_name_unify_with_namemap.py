# scripts/ai_name_unify_with_namemap.py
import os
import re
import json
import time
import math
import signal
import asyncio
from pathlib import Path
from typing import Dict, Tuple, List, Set
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# ================== 环境与路径 ==================
load_dotenv()

# 数据输入（只改中文列）
DATA_PATH        = Path(os.getenv("UNIFY_DATA_PATH", "output/language_dict_mcname_fixed.json"))
LANG_MAP_PATH    = Path(os.getenv("LANG_MAP_PATH", "data/language_map.json"))
NAME_MAP_PATH    = Path(os.getenv("NAME_MAP_PATH", "data/name_map.json"))
OUTPUT_PATH      = Path(os.getenv("UNIFY_OUTPUT_PATH", "output/language_dict_name_unified.json"))
REPORT_PATH      = Path(os.getenv("UNIFY_REPORT_PATH", "output/name_unify_report.txt"))

# 缓存
CACHE_DIR        = Path(os.getenv("CACHE_DIR", "cache"))
CACHE_PATH       = CACHE_DIR / "name_unify_cache.json"

# 引擎选择（默认读取 .env）
DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "").strip().lower()  # chatgpt / deepseek
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = os.getenv("MODEL", "gpt-4o-mini").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASEURL = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn").strip()
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1").strip()

# 并发与限速（DeepSeek: RPM=1000, TPM=100000）
ASYNC_CONCURRENCY = int(os.getenv("UNIFY_ASYNC_CONCURRENCY", "120"))
REQUEST_TIMEOUT   = int(os.getenv("UNIFY_REQUEST_TIMEOUT", "30"))
RPM               = int(os.getenv("UNIFY_RPM", "1000"))
TPM               = int(os.getenv("UNIFY_TPM", "100000"))

# 打印/落盘
PRINT_EVERY       = int(os.getenv("UNIFY_PRINT_EVERY", "50"))
FLUSH_EVERY       = int(os.getenv("UNIFY_FLUSH_EVERY", "500"))

# 候选筛选（仅 1~2 词的短专名）
MAX_TOKENS_IN_PHRASE = 2

# 英文/俄文词正则（仅字母/连字符/句点）
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё.\-']+")

# 忽略词（称谓/冠词等；不当作专名）
STOPWORDS_EN = {
    "the","a","an","mr","mrs","ms","miss","dr","prof","professor","coach",
    "sir","madam","lady","lord","captain","principal","dean","ok","tv","id","usa","eu"
}
STOPWORDS_RU = {
    "мистер","миссис","мисс","проф","профессор","господин","госпожа","капитан","директор"
}

# 只对中文列进行修改
TARGET_LANG_KEYWORD = "Chinese"  # 用于 language_map.json 查中文列

stop_requested = False

# ================== 基础工具 ==================
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

def ensure_row_len(row: list, idx: int):
    if len(row) <= idx:
        row.extend([""] * (idx - len(row) + 1))

def pick_col_by_lang(lang_map: Dict[str, str], label_contains: str) -> int:
    for k, v in lang_map.items():
        if v and label_contains.lower() in v.lower():
            return int(k)
    return -1

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text or "")

def normalize_token(tok: str) -> str:
    return re.sub(r"[^\w\-\.']", "", tok).strip().lower()

def is_stopword(tok_norm: str, is_ru: bool) -> bool:
    return tok_norm in (STOPWORDS_RU if is_ru else STOPWORDS_EN)

def chunked(iterable, size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

def eta_str(done:int, total:int, start_ts:float) -> str:
    if done == 0:
        return "--:--"
    elapsed = time.time() - start_ts
    rate = done / max(1e-9, elapsed)
    remain = max(0, total - done) / max(1e-9, rate)
    m, s = divmod(int(remain), 60)
    return f"{m}m{s:02d}s"

# ================== 信号安全退出 ==================
def handle_signal(signum, frame):
    global stop_requested
    stop_requested = True
    print("\n⚠️ 捕获中断信号，准备安全落盘退出…")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ================== Rate Limiter（自适应） ==================
class RateLimiter:
    def __init__(self, rpm:int, tpm:int):
        self.rpm = rpm
        self.tpm = tpm
        self._lock = asyncio.Lock()
        self._win_start = time.monotonic()
        self._req_used = 0
        self._tok_used = 0
        # 自适应参数
        self._cooldown = 0.0   # 短时冷却附加
        self._scale = 1.0      # 动态调速系数（越小越慢）

    def _maybe_reset(self):
        now = time.monotonic()
        if now - self._win_start >= 60.0:
            self._win_start = now
            self._req_used = 0
            self._tok_used = 0
            # 逐步恢复速度
            self._scale = min(1.0, self._scale * 1.05)
            self._cooldown = max(0.0, self._cooldown * 0.7)

    @staticmethod
    def estimate_tokens(prompt_len:int) -> int:
        # 粗估 token：统一按 3.5 字符/Token，最低 16
        return max(16, int(prompt_len / 3.5))

    async def acquire(self, prompt_len:int):
        need_tokens = self.estimate_tokens(prompt_len)
        while True:
            async with self._lock:
                self._maybe_reset()
                rpm_cap = max(1, int(self.rpm * self._scale))
                tpm_cap = max(512, int(self.tpm * self._scale))
                can_req = (self._req_used + 1) <= rpm_cap
                can_tok = (self._tok_used + need_tokens) <= tpm_cap
                if can_req and can_tok:
                    self._req_used += 1
                    self._tok_used += need_tokens
                    cd = self._cooldown
                    self._cooldown = max(0.0, self._cooldown * 0.8)
                    # 冷却延时（命中429后会升高）
                    if cd > 0:
                        await asyncio.sleep(min(2.0, cd))
                    return
                wait = max(0.02, 60.0 - (time.monotonic() - self._win_start))
            await asyncio.sleep(min(1.0, wait))

    async def penalize(self):
        # 命中限速 -> 立刻降速 + 增加短冷却
        async with self._lock:
            self._scale = max(0.2, self._scale * 0.85)
            self._cooldown = min(2.0, self._cooldown + 0.2)

# ================== 缓存 ==================
class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = load_json(path, {})
        self._lock = asyncio.Lock()

    def _key(self, provider:str, model:str, ru:str, en:str, cn:str) -> str:
        # 简单键（足够唯一）
        raw = f"{provider}|{model}|{ru}|{en}|{cn}"
        return str(abs(hash(raw)))

    async def get(self, provider:str, model:str, ru:str, en:str, cn:str):
        k = self._key(provider, model, ru, en, cn)
        async with self._lock:
            return self.data.get(k, "")

    async def set(self, provider:str, model:str, ru:str, en:str, cn:str, value:str):
        k = self._key(provider, model, ru, en, cn)
        async with self._lock:
            if k not in self.data:
                self.data[k] = value

    async def flush(self):
        async with self._lock:
            save_json(self.path, self.data)

# ================== OpenAI Async 客户端 ==================
@asynccontextmanager
async def build_async_client(provider:str):
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
    print("请选择引擎（1=ChatGPT，2=DeepSeek）：", end="")
    choice = input().strip()
    if choice == "2" or (not choice and DEFAULT_PROVIDER == "deepseek"):
        print("🧠 使用 DEEPSEEK 模型引擎")
        return "deepseek"
    print("🧠 使用 CHATGPT 模型引擎")
    return "chatgpt"

# ================== AI 判定 & 替换 ==================
def build_ai_prompt(ru: str, en: str, cn: str) -> str:
    return f"""你是三语本地化一致性审校专家。下面是成人向视觉小说的一条三语文本（俄/英/中）：

俄文：{ru}
英文：{en}
中文：{cn}

任务：识别其中应当统一的“专有名词（人名、地名、组织名、唯一称呼）”，并给出全局统一的中文译法。

要求：
- 仅针对 1~2 个词的短语（例如：Rose、Miss Young、Sunville、Professor Richardson）
- 若某专名在中文中出现多种译法或有残留英文/俄文，请选择**最自然、本地化、统一**的中文版本
- 仅输出 JSON（键为原始英文或俄文短词，值为最终中文统一译名）
- 如无可统一项，输出 {{}}
- 严禁输出任何解释或附加文本
"""

def clean_json_only(s: str) -> Dict[str, str]:
    if not s:
        return {}
    s = s.strip().strip("`").strip()
    # 截取第一个 {...}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            # 仅保留 str->str
            clean = {}
            for k, v in obj.items():
                if isinstance(k, str) and isinstance(v, str):
                    k1 = k.strip()
                    v1 = v.strip()
                    if k1 and v1:
                        clean[k1] = v1
            return clean
    except Exception:
        return {}
    return {}

def replace_pairs_in_text(text: str, pairs: Dict[str, str]) -> str:
    # 长词优先，忽略大小写；不跨标签
    if not pairs:
        return text
    out = text
    for src, tgt in sorted(pairs.items(), key=lambda kv: -len(kv[0])):
        if not src or not tgt:
            continue
        # 只做直接子串替换（这里中文列，通常无变量冲突）
        out = re.sub(re.escape(src), tgt, out, flags=re.IGNORECASE)
    return out

# 候选短语抽取（1~2词），从俄/英各抓，再与中文进行对齐判断
def extract_short_candidates(ru: str, en: str) -> Set[str]:
    cands: Set[str] = set()
    # 英文
    toks_en = tokenize(en)
    buf = [t for t in toks_en if t]
    # 俄文
    toks_ru = tokenize(ru)
    buf_ru = [t for t in toks_ru if t]

    def add_phrases(tokens: List[str], is_ru: bool):
        # 单词
        for t in tokens:
            norm = normalize_token(t)
            if not norm or is_stopword(norm, is_ru):
                continue
            # 首字母大写优先/有专名味道：简单规则：出现大写或包含点/连字符
            if re.match(r"[A-ZА-ЯЁ]", t) or "." in t or "-" in t:
                cands.add(t.strip())
        # 两词
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i+1]
            if not t1 or not t2:
                continue
            n1, n2 = normalize_token(t1), normalize_token(t2)
            if not n1 or not n2:
                continue
            if is_stopword(n1, is_ru) or is_stopword(n2, is_ru):
                continue
            phrase = f"{t1.strip()} {t2.strip()}"
            # 任一含大写/点/连字符即可
            if (re.match(r"[A-ZА-ЯЁ]", t1) or re.match(r"[A-ZА-ЯЁ]", t2) or
                "." in phrase or "-" in phrase):
                cands.add(phrase)

    add_phrases(buf, is_ru=False)
    add_phrases(buf_ru, is_ru=True)
    # 控制长度：不超过两词
    cands_final = set()
    for p in cands:
        if 1 <= len(p.strip().split()) <= MAX_TOKENS_IN_PHRASE:
            cands_final.add(p)
    return cands_final

# ================== 主异步流程 ==================
async def main_async():
    provider = choose_provider()

    # 载数据
    data     = load_json(DATA_PATH, {})
    lang_map = load_json(LANG_MAP_PATH, {})
    name_map = load_json(NAME_MAP_PATH, {})

    total = len(data)
    # 定位列
    ru_col = pick_col_by_lang(lang_map, "Russian")
    en_col = pick_col_by_lang(lang_map, "English")
    zh_col = pick_col_by_lang(lang_map, TARGET_LANG_KEYWORD)
    if ru_col < 0 or en_col < 0 or zh_col < 0:
        print("❌ 未检测到完整三语列（需 Russian / English / Chinese）")
        return

    print(f"✅ 数据载入成功：共 {total} 条。俄文列={ru_col}，英文列={en_col}，中文列={zh_col}")

    # 筛选候选（仅中文列非空，且中英俄均有内容）
    items: List[Tuple[str, List[str]]] = []
    for key, row in data.items():
        if not isinstance(row, list):
            continue
        ensure_row_len(row, max(ru_col, en_col, zh_col))
        ru, en, cn = (row[ru_col] or "").strip(), (row[en_col] or "").strip(), (row[zh_col] or "").strip()
        if not (ru and en and cn):
            continue
        # 只对**中文列**做统一；保留名词表中已存在的优先
        items.append((key, [ru, en, cn]))

    print(f"📦 待检查候选：{len(items)} 条。")
    if not items:
        print("✅ 统一完成，共修正 0 条。")
        save_json(OUTPUT_PATH, data)
        save_json(NAME_MAP_PATH, name_map)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("")
        print(f"📘 报告保存至: {REPORT_PATH}")
        print(f"📁 输出文件: {OUTPUT_PATH}")
        print(f"🧠 更新专名映射表: {NAME_MAP_PATH}（新增 0 项）")
        return

    # 准备异步上下文
    limiter = RateLimiter(RPM, TPM)
    cache   = Cache(CACHE_PATH)
    start_ts = time.time()
    processed = 0
    modified  = 0
    new_name_pairs = {}   # 新增入 name_map 的键值
    report_lines: List[str] = []

    # 并发控制
    sem = asyncio.Semaphore(ASYNC_CONCURRENCY)

    @asynccontextmanager
    async def get_client():
        async with build_async_client(provider) as (client, model):
            yield client, model

    async def one_job(client, model, key: str, ru: str, en: str, cn: str):
        nonlocal processed, modified
        # 提取 1~2 词的候选短专名
        cands = extract_short_candidates(ru, en)
        if not cands:
            processed += 1
            return None

        # 如果所有候选都已经在中文里一致或已被 name_map 完全覆盖，也不必请求
        need_ai = False
        for c in cands:
            # 若中文里出现对应英文/俄文残留，或同词存在多种形式，才需要AI
            if re.search(re.escape(c), cn, flags=re.IGNORECASE):
                need_ai = True
                break
            # 若 name_map 有该键但中文未替换，也需要AI确认统一（兼容大小写）
            if c in name_map and name_map[c] not in cn:
                need_ai = True
                break
        if not need_ai:
            processed += 1
            return None

        # 生成 prompt（控制 tokens 估计用）
        prompt = build_ai_prompt(ru, en, cn)
        prompt_len = len(prompt)

        # 缓存查询
        cached = await cache.get(provider, model, ru, en, cn)
        if cached:
            mapping = clean_json_only(cached)
        else:
            # 限速许可
            await limiter.acquire(prompt_len)
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a multilingual localization consistency assistant. Output JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        timeout=REQUEST_TIMEOUT,
                    )
                raw = (resp.choices[0].message.content or "").strip()
                mapping = clean_json_only(raw)
                await cache.set(provider, model, ru, en, cn, raw)
            except Exception as e:
                # 命中限速 -> 降速惩罚
                if "RateLimit" in type(e).__name__ or "429" in str(e):
                    await limiter.penalize()
                mapping = {}

        if mapping:
            # 仅写中文列
            new_cn = replace_pairs_in_text(cn, mapping)
            if new_cn != cn:
                data[key][zh_col] = new_cn
                modified += 1
                # 报告写入
                report_lines.append(f"\n原: {cn}\n新: {new_cn}\n映射: {json.dumps(mapping, ensure_ascii=False)}\n")
            # 更新 name_map（不覆盖旧值）
            changed = 0
            for k0, v0 in mapping.items():
                if k0 not in name_map:
                    name_map[k0] = v0
                    new_name_pairs[k0] = v0
                    changed += 1
            if changed:
                # 不中断流程，最终统一落盘
                pass

        processed += 1
        return mapping if mapping else None

    async with get_client() as (client, model):
        tasks = []
        for key, arr in items:
            ru, en, cn = arr
            tasks.append(one_job(client, model, key, ru, en, cn))

        # 分批 gather，避免一次性创建超大任务
        BATCH = 5000
        for chunk in chunked(tasks, BATCH):
            if stop_requested:
                break
            await asyncio.gather(*chunk)
            # 实时落盘（断点保护）
            if processed % FLUSH_EVERY != 0:
                continue
            save_json(OUTPUT_PATH, data)
            save_json(NAME_MAP_PATH, name_map)
            await cache.flush()
            done_pct = processed / max(1, len(items)) * 100.0
            print(f"💾 自动保存 | 进度 {processed}/{len(items)} ({done_pct:.1f}%) | 预计剩余 {eta_str(processed, len(items), start_ts)}")

            if stop_requested:
                break

        # 最终落盘
        save_json(OUTPUT_PATH, data)
        save_json(NAME_MAP_PATH, name_map)
        await cache.flush()

    # 进度总结
    print(f"✅ 统一完成，修正 {modified} 条。报告：{REPORT_PATH}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

# ================== 入口 ==================
def main():
    print("", end="")  # 让 PowerShell 先刷新一行
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

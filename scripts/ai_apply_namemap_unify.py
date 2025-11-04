import os
import re
import json
import time
import asyncio
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ========= 环境 =========
load_dotenv()

DATA_PATH      = Path("output/language_dict_mcsurname_fixed.json")
LANG_MAP_PATH  = Path("data/language_map.json")
NAMEMAP_PATH   = Path("data/name_map.json")
OUTPUT_PATH    = Path("output/language_dict_namemap_applied.json")
REPORT_PATH    = Path("output/namemap_apply_report.txt")
CACHE_PATH     = Path("cache/ai_namemap_cache.json")

# ========= 模型 =========
DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "").strip().lower()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = os.getenv("MODEL", "gpt-4o-mini").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASEURL = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn").strip()
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1").strip()

ASYNC_CONCURRENCY = int(os.getenv("ASYNC_CONCURRENCY", "64"))
REQUEST_TIMEOUT   = int(os.getenv("REQUEST_TIMEOUT", "30"))
RPM               = int(os.getenv("RPM", "1000"))
TPM               = int(os.getenv("TPM", "100000"))
PRINT_EVERY       = 200

# ========= 辅助 =========
def load_json(p, default=None):
    if default is None: default = {}
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else default

def save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)

def clean(s: str):
    if not s: return ""
    s = s.strip().strip("`").strip()
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    return lines[0] if lines else ""

def pick_col(lang_map, label):
    for k,v in lang_map.items():
        if label.lower() in v.lower(): return int(k)
    return -1

# ========= 速率控制 =========
class RateLimiter:
    def __init__(self, rpm, tpm):
        self.rpm, self.tpm = rpm, tpm
        self._window = time.monotonic()
        self._r = 0
        self._t = 0
        self._lock = asyncio.Lock()
    def _reset(self):
        if time.monotonic()-self._window>=60:
            self._window=time.monotonic()
            self._r=self._t=0
    async def acquire(self, text_len:int):
        est = max(10, text_len//3)
        while True:
            async with self._lock:
                self._reset()
                if self._r+1<=self.rpm and self._t+est<=self.tpm:
                    self._r+=1;self._t+=est;return
                wait=max(0,60-(time.monotonic()-self._window))
            await asyncio.sleep(min(wait,1))

# ========= 异步客户端 =========
async def build_client():
    print("请选择引擎（1=ChatGPT，2=DeepSeek）：", end="")
    ch = input().strip()
    if ch=="2" or (not ch and DEFAULT_PROVIDER=="deepseek"):
        print("🧠 使用 DeepSeek 模型引擎")
        cli = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASEURL)
        return cli, DEEPSEEK_MODEL
    print("🧠 使用 ChatGPT 模型引擎")
    cli = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return cli, OPENAI_MODEL

# ========= Prompt =========
def build_prompt(text, name_map):
    return f"""
你是成人向视觉小说的中文本地化校对专家。
下列中文翻译中可能存在前后不一致或音译错误的人名/地名/称呼。
请根据给定的专名映射表，检查是否需要替换为更统一的说法。

要求：
1. 输出仅为修正后的完整句子。
2. 不要添加解释、标点、符号。
3. 严格保持原句风格与语气。

专名映射表（部分示例）：
{name_map}

中文文本：
{text}
""".strip()

# ========= 主流程 =========
async def main():
    client, model = await build_client()
    limiter = RateLimiter(RPM, TPM)

    data = load_json(DATA_PATH)
    lang_map = load_json(LANG_MAP_PATH)
    name_map = load_json(NAMEMAP_PATH)
    zh_col = pick_col(lang_map, "Chinese")
    en_col = pick_col(lang_map, "English")
    ru_col = pick_col(lang_map, "Russian")

    total = len(data)
    print(f"✅ 载入成功：共 {total} 条，中文列={zh_col}")

    # 构建反查（多对一）映射
    inv_map = {}
    for k,v in name_map.items():
        if not k or not v: continue
        inv_map.setdefault(v,set()).add(k)

    limiter = RateLimiter(RPM, TPM)
    sem = asyncio.Semaphore(ASYNC_CONCURRENCY)
    modified, last = 0, time.monotonic()

    async def process_one(idx, key, text):
        nonlocal modified
        if not text.strip(): return text
        await limiter.acquire(len(text))
        rep = text
        # 先本地替换
        for src,tgt in sorted(name_map.items(),key=lambda kv:-len(kv[0])):
            if src in rep: rep = rep.replace(src,tgt)
        # 若仍混杂中英俄，调用AI微调
        if re.search(r"[A-Za-zА-Яа-яЁё]", rep):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content":build_prompt(rep,list(name_map.items())[:40])}],
                    temperature=0.4,
                    timeout=REQUEST_TIMEOUT)
                new = clean(resp.choices[0].message.content)
                if new and new!=rep:
                    rep=new; modified+=1
            except Exception as e:
                print(f"⚠️ {type(e).__name__}")
        return rep

    tasks=[]
    for i,(key,row) in enumerate(data.items(),1):
        if len(row)<=zh_col: continue
        text=row[zh_col]
        tasks.append((i,key,text))
    total=len(tasks)
    print(f"📦 待优化 {total} 条中文翻译")

    async def worker(i,key,text):
        async with sem:
            new=await process_one(i,key,text)
            data[key][zh_col]=new
            if i%PRINT_EVERY==0:
                print(f"🔄 {i}/{total} 完成")
            if i%1000==0 or time.monotonic()-last>30:
                save_json(OUTPUT_PATH,data)

    await asyncio.gather(*[worker(i,k,t) for i,k,t in tasks])
    save_json(OUTPUT_PATH,data)
    print(f"🎉 完成，修正 {modified} 条；结果已保存至 {OUTPUT_PATH}")

if __name__=="__main__":
    asyncio.run(main())

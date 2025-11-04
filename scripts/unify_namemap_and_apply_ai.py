# scripts/unify_namemap_and_apply_ai.py
import os, json, re, time, asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ====== 环境配置 ======
load_dotenv()
DATA_PATH = Path("output/language_dict_mcname_fixed.json")
LANG_MAP_PATH = Path("data/language_map.json")
NAME_MAP_PATH = Path("data/name_map.json")
OUTPUT_PATH = Path("output/language_dict_name_final.json")
REPORT_PATH = Path("output/name_unify_report.txt")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASEURL = os.getenv("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-R1")

# ====== 选择引擎 ======
def choose_provider():
    print("\n请选择AI引擎：")
    print("  1) ChatGPT")
    print("  2) DeepSeek")
    choice = input("👉 输入 1 或 2 (默认1)：").strip()
    if choice == "2":
        print("🧠 使用 DeepSeek 模型引擎")
        return AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASEURL), DEEPSEEK_MODEL
    else:
        print("🧠 使用 ChatGPT 模型引擎")
        return AsyncOpenAI(api_key=OPENAI_API_KEY), MODEL

# ====== 工具函数 ======
def load_json(p, default=None):
    if default is None: default = {}
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else default

def save_json(p, data):
    p.parent.mkdir(exist_ok=True, parents=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_out(txt):
    if not txt: return ""
    txt = txt.strip().strip("`").strip()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    return lines[0] if lines else ""

# ====== AI语义合并 ======
async def ai_merge_names(client, model, candidates):
    """让AI判断这些名字是否为同一个实体，并给出统一中文译名"""
    if len(candidates) < 2:
        return None
    joined = " / ".join(candidates)
    prompt = f"""
你是一位本地化校对专家。
这些词来自俄语、英语、中文混合文本中，表示可能的同一人物或地名。
请判断它们是否语义上属于同一个实体，并给出最终应统一的中文译名。

示例：
输入：["Jacob","Джейкоб","雅各布"]
输出：{{"is_same": true, "final_name": "雅各布"}}

输入：["Sun","Sunville","阳光谷"]
输出：{{"is_same": true, "final_name": "阳光谷"}}

输入：["History","История"]
输出：{{"is_same": false}}

请严格输出 JSON：
{{
  "is_same": true/false,
  "final_name": "..."
}}

待判断词组：{joined}
    """
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            timeout=40,
        )
        out = clean_out(r.choices[0].message.content)
        try:
            js = json.loads(out)
            if js.get("is_same"):
                return js.get("final_name", "")
        except Exception:
            return None
    except Exception as e:
        print(f"⚠️ AI合并失败: {e}")
    return None

# ====== 替换函数 ======
def replace_names_in_text(text, name_map):
    for k,v in sorted(name_map.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text

# ====== 主逻辑 ======
async def main():
    client, model = choose_provider()
    data = load_json(DATA_PATH, {})
    lang_map = load_json(LANG_MAP_PATH, {})
    name_map = load_json(NAME_MAP_PATH, {})

    zh_col = next((int(k) for k,v in lang_map.items() if "Chinese" in v), None)
    ru_col = next((int(k) for k,v in lang_map.items() if "Russian" in v), None)
    en_col = next((int(k) for k,v in lang_map.items() if "English" in v), None)
    if zh_col is None or ru_col is None or en_col is None:
        print("❌ 找不到完整列配置，请检查 language_map.json")
        return

    print(f"✅ 数据载入成功：共 {len(data)} 条。俄={ru_col}, 英={en_col}, 中={zh_col}")

    # 统计所有候选词（仅限1-2词，重复出现多次的）
    freq = {}
    for arr in data.values():
        if len(arr) <= zh_col: continue
        txt = arr[zh_col]
        if not isinstance(txt,str): continue
        for token in re.findall(r"[A-Za-zА-Яа-яЁё一-龥]+", txt):
            if 1 <= len(token) <= 10:
                freq[token] = freq.get(token,0)+1

    candidates = [w for w,c in freq.items() if c>=3]  # 至少出现3次才视为候选
    print(f"📦 待检查候选：{len(candidates)} 条。")

    unified = {}
    report_lines = []
    tasks = []

    async def process_group(w):
        # 查找英文/俄文中对应形态
        related = [w]
        for alt in name_map.keys():
            if alt.lower() == w.lower():
                related.append(alt)
        related = list(set(related))
        if len(related)>1:
            final = await ai_merge_names(client, model, related)
            if final:
                for r in related:
                    unified[r] = final
                report_lines.append(f"🧩 合并 {related} → {final}")
                print(f"🧩 合并 {related} → {final}")

    sem = asyncio.Semaphore(10)
    async def limited_run(w):
        async with sem:
            await process_group(w)
            await asyncio.sleep(0.5)

    for w in candidates:
        tasks.append(asyncio.create_task(limited_run(w)))
    await asyncio.gather(*tasks)

    # 应用统一映射
    merged_map = {**name_map, **unified}
    modified = 0
    for key, arr in data.items():
        if len(arr) <= zh_col: continue
        old = arr[zh_col]
        new = replace_names_in_text(old, merged_map)
        if new != old:
            arr[zh_col] = new
            modified += 1

    save_json(NAME_MAP_PATH, merged_map)
    save_json(OUTPUT_PATH, data)
    with open(REPORT_PATH,"w",encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ 统一完成，修正 {modified} 条。")
    print(f"📘 报告: {REPORT_PATH}")
    print(f"📁 输出: {OUTPUT_PATH}")
    print(f"🧠 专名映射表更新: {NAME_MAP_PATH}（新增 {len(unified)} 项）")

if __name__ == "__main__":
    asyncio.run(main())

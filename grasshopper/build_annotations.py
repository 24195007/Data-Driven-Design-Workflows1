import os, json, base64
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from PIL import Image
from openai import OpenAI, OpenAIError   # ★ 新增这一行


IMG_DIR = Path("images")
CSV_OUT = "annotations.csv"
MODEL   = "gpt-4o"           

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("❌ 未在 .env 找到 OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

PROMPT_SYS = (
  "You are an architect-psychologist.\n"
  "Given a residential photo, score six feelings—Safety, Belonging, "
  "Naturalness, Shared, Privacy, Convenience—each between 0 and 1 (one decimal). "
  "Respond ONLY in minified JSON like: "
  "{\"Safety\":0.7,\"Belonging\":0.5,...}"
)

def encode_image(p):
    img = Image.open(p).convert("RGB")
    img.thumbnail((768,768))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

jpgs = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))
print("图片数量 =", len(jpgs))

rows = []
for p in tqdm(jpgs, desc="Processing"):
    try:
        img64 = encode_image(p)
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[
              {"role":"system","content":PROMPT_SYS},
              {"role":"user","content":[
                 {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img64}"}}]}],
            max_tokens=100)
        raw = rsp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")     # 去除 markdown 代码围栏
        js  = json.loads(raw)
    except Exception as e:
        print(f"FAIL -> {p.name}: {e}")
        continue

    rows.append({
        "filename":p.name,
        "concept_text":"",
        **js
    })

if not rows:
    sys.exit("❌ 所有图片失败，终止。先解决 FAIL 原因。")

pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
print(f"✅ 写出 {CSV_OUT} ({len(rows)} 行)")

import json, os, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from huggingface_hub import HfApi

STATE = Path("state.json")
WEBHOOK = os.environ["DISCORD_WEBHOOK"]
MAX_AGE = timedelta(days=30)
MIN_LIKES, MIN_DOWNLOADS = 5, 50
PIPELINES = ("text-to-image", "image-to-image")
QUANT_WORDS = ("gguf", "gptq", "awq", "mlx", "bnb", "nf4", "fp8", "int4", "int8",
               "exl2", "quantized", "4bit", "8bit", "q4", "q5", "q6", "q8")
QUANT_TAGS = {"gguf", "gptq", "awq", "mlx", "bnb", "exl2", "4-bit", "8-bit",
              "fp8", "int4", "int8", "nf4", "quantized"}
SKIP_TAGS = {"lora"}  # to include LoRAs, change to: SKIP_TAGS = set()

api = HfApi()
seen = set(json.loads(STATE.read_text())) if STATE.exists() else set()

def is_quantized(m):
    name = m.id.lower()
    tags = {t.lower() for t in (m.tags or [])}
    rel = getattr(m.card_data, "base_model_relation", None)
    return (rel == "quantized" or tags & QUANT_TAGS
            or any(w in name for w in QUANT_WORDS))

def is_derivative(m):
    return any(t.lower().startswith("base_model:") for t in (m.tags or []))

def candidates():
    for p in PIPELINES:
        yield from api.list_models(pipeline_tag=p, sort="created_at",
                                   limit=300, cardData=True)
        yield from api.list_models(pipeline_tag=p, sort="trending_score",
                                   limit=100, cardData=True)

hits, done = [], set()
now = datetime.now(timezone.utc)
for m in candidates():
    if m.id in done or m.id in seen:
        continue
    done.add(m.id)
    if now - m.created_at > MAX_AGE:
        continue
    tags = {t.lower() for t in (m.tags or [])}
    if is_quantized(m) or is_derivative(m) or tags & SKIP_TAGS:
        continue
    if (m.likes or 0) >= MIN_LIKES or (m.downloads or 0) >= MIN_DOWNLOADS:
        hits.append(f"**{m.id}** [{m.pipeline_tag}] ({m.likes} likes, {m.downloads} downloads) https://huggingface.co/{m.id}")
        seen.add(m.id)

for i in range(0, len(hits), 10):
    requests.post(WEBHOOK, json={"content": "\n".join(hits[i:i+10])[:1990]})
STATE.write_text(json.dumps(sorted(seen)))

import json, os, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from huggingface_hub import HfApi

STATE = Path("state.json")
WEBHOOK = os.environ["DISCORD_WEBHOOK"]
MAX_AGE = timedelta(days=30)
SKIP_TAGS = {"lora"}  # to include LoRAs, change to: SKIP_TAGS = set()

api = HfApi()
seen = set(json.loads(STATE.read_text())) if STATE.exists() else set()

def is_derivative(m):
    return any(t.lower().startswith("base_model:") for t in (m.tags or []))

def candidates():
    yield from api.list_models(pipeline_tag="text-to-image", sort="created_at",
                               limit=500, cardData=True)

hits, done = [], set()
stats = {"checked": 0, "too_old": 0, "filtered": 0, "sent": 0}
now = datetime.now(timezone.utc)
for m in candidates():
    if m.id in done or m.id in seen:
        continue
    done.add(m.id)
    stats["checked"] += 1
    if now - m.created_at > MAX_AGE:
        stats["too_old"] += 1
        continue
    tags = {t.lower() for t in (m.tags or [])}
    if is_derivative(m) or tags & SKIP_TAGS:
        stats["filtered"] += 1
        continue
    hits.append(f"**{m.id}** [{m.pipeline_tag}] ({m.likes} likes, {m.downloads} downloads) <https://huggingface.co/{m.id}>")
    seen.add(m.id)
    stats["sent"] += 1
print(stats)

for i in range(0, len(hits), 10):
    requests.post(WEBHOOK, json={"content": "\n".join(hits[i:i+10])[:1990]})
STATE.write_text(json.dumps(sorted(seen)))

import json, os, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from huggingface_hub import HfApi

STATE = Path("state.json")
WEBHOOK = os.environ["DISCORD_WEBHOOK"]
MAX_AGE = timedelta(days=30)
PIPELINES = ("text-to-image", "image-to-image")
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")
PAINT_WORDS = ("paint", "oil", "watercolor", "watercolour", "gouache",
               "impasto", "acrylic", "canvas", "brushwork")
NSFW_WORDS = ("nsfw", "uncensored", "abliterated", "porn", "hentai", "nude",
              "naked", "lewd", "erotic", "explicit", "xxx", "not-for-all-audiences")

api = HfApi()
seen = set(json.loads(STATE.read_text())) if STATE.exists() else set()

def is_derivative(m):
    tags = [t.lower() for t in (m.tags or [])]
    if "lora" in tags or "lora" in m.id.lower():
        text = (m.id + " " + " ".join(tags)).lower()
        return not any(w in text for w in PAINT_WORDS)
    return any(t.startswith("base_model:") for t in tags)

def is_nsfw(m):
    text = (m.id + " " + " ".join(m.tags or [])).lower()
    return any(w in text for w in NSFW_WORDS)

def has_example_image(m):
    files = [s.rfilename.lower() for s in (m.siblings or [])]
    if any(f.endswith(IMG_EXT) for f in files):
        return True
    widget = getattr(m.card_data, "widget", None) if m.card_data else None
    for w in (widget or []):
        out = w.get("output") if isinstance(w, dict) else None
        if isinstance(out, dict) and out.get("url"):
            return True
    return False

def candidates():
    for p in PIPELINES:
        yield from api.list_models(pipeline_tag=p, sort="created_at",
                                   limit=500, cardData=True, full=True)

hits, done = [], set()
stats = {"checked": 0, "too_old": 0, "filtered": 0, "no_image": 0, "sent": 0}
now = datetime.now(timezone.utc)
for m in candidates():
    if m.id in done or m.id in seen:
        continue
    done.add(m.id)
    stats["checked"] += 1
    if now - m.created_at > MAX_AGE:
        stats["too_old"] += 1
        continue
    if is_derivative(m) or is_nsfw(m):
        stats["filtered"] += 1
        continue
    if not has_example_image(m):
        stats["no_image"] += 1
        continue
    hits.append(f"**{m.id}** [{m.pipeline_tag}] ({m.likes} likes, {m.downloads} downloads) <https://huggingface.co/{m.id}>")
    seen.add(m.id)
    stats["sent"] += 1
print(stats)

for i in range(0, len(hits), 10):
    requests.post(WEBHOOK, json={"content": "\n".join(hits[i:i+10])[:1990]})
STATE.write_text(json.dumps(sorted(seen)))

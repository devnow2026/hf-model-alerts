import json, os, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from huggingface_hub import HfApi

STATE = Path("state.json")
WEBHOOK = os.environ["DISCORD_WEBHOOK"]
MAX_AGE = timedelta(days=30)
MIN_LIKES, MIN_DOWNLOADS = 10, 500
CODE_NAME = ("code", "coder", "coding", "codestral", "devstral", "swe")
CODE_BENCH = ("humaneval", "mbpp", "livecodebench", "swe-bench", "swe_bench", "bigcodebench")

api = HfApi()
seen = set(json.loads(STATE.read_text())) if STATE.exists() else set()

def candidates():
    for term in ("code", "coder", "coding"):
        yield from api.list_models(search=term, pipeline_tag="text-generation",
                                   sort="created_at", direction=-1, limit=300, cardData=True)
    yield from api.list_models(pipeline_tag="text-generation", sort="trending_score",
                               direction=-1, limit=200, cardData=True)

def code_evals(m):
    evals = getattr(m.card_data, "eval_results", None) or []
    return [e for e in evals if any(b in f"{e.dataset_name} {e.dataset_type}".lower() for b in CODE_BENCH)]

hits, done = [], set()
now = datetime.now(timezone.utc)
for m in candidates():
    if m.id in done or m.id in seen:
        continue
    done.add(m.id)
    if now - m.created_at > MAX_AGE:
        continue
    evals = code_evals(m)
    codey = evals or any(k in m.id.lower() for k in CODE_NAME) or "code" in (m.tags or [])
    proven = evals or (m.likes or 0) >= MIN_LIKES or (m.downloads or 0) >= MIN_DOWNLOADS
    if codey and proven:
        why = "reports code benchmark" if evals else f"{m.likes} likes, {m.downloads} downloads"
        hits.append(f"**{m.id}** ({why}) https://huggingface.co/{m.id}")
        seen.add(m.id)

for i in range(0, len(hits), 10):
    requests.post(WEBHOOK, json={"content": "\n".join(hits[i:i+10])[:1990]})
STATE.write_text(json.dumps(sorted(seen)))

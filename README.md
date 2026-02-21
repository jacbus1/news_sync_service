# news_sync_service Clean Edition

Public-safe research subset of `news_sync_service`.

## Scope
- Focus: data collection, normalization, analytics, ticker resolution, and offline/local research workflows.
- Excluded: Telegram/bot runtime flows, cloud deployment docs, and secret-bearing files.
- This repository does not include AWS deployment configuration.

## Quick Start (localhost-style)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build watchlists (optional FMP key via CLI):
```bash
python src/watchlists_builder.py --out-dir ./data/watchlists
```

Generate Yahoo impact JSON from local SQLite data:
```bash
python src/build_yahoo_impact_json.py \
  --db ./data/news_sync.db \
  --out ./data/yahoo_impact.json \
  --since-hours 120 \
  --max-items 600
```

## Localhost Defaults
- Paths use relative defaults (for example `./data/news_sync.db`).
- LLM-related modules default to local endpoint patterns where applicable (`http://127.0.0.1:11434/v1`).
- No `.env` file is required by the included research entry scripts.

## Data Policy
- `data/` is intentionally kept empty in the public template.
- Add your own local data files as needed.

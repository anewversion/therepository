# Data Enhancement (Web discovery + contact extraction)

This folder contains a practical, **business-focused** data collection + enrichment workflow.

It’s designed for cases where you need to:
- discover businesses for a category in one or more US states,
- collect canonical fields (name, address, lat/lon, website, phone, etc.),
- then (optionally) crawl **the business’s own website** to find **publicly posted** contact channels and leadership/team signals.

It’s intentionally built as “field notes from building”: clean enough to run, flexible enough to adapt.

---

## What’s included

### 1) `src/Unified_Web_Scraper_vNew2.py` — discovery
A “smart scrape” discovery script that can pull business listings from:
- **OpenStreetMap (Overpass API)** (enabled by default)
- **Overture Maps** (optional, requires DuckDB)

Key behaviors:
- keyword → tag/category mapping (with fallback matching)
- quality scoring + filters (keep only “good enough” rows)
- caching + retry/backoff against public endpoints
- output can merge or split results by state

Outputs land in a `runs_smart_scrape/` folder created next to the script.

### 2) `src/Executive_Finder_Unified.py` — website contact extraction
Given a CSV of businesses (or just websites), this script visits each site and tries to extract:
- **emails** that are publicly present on the site,
- **contact page signals** (about/team/contact),
- lightweight “executive / leadership” hints where present.

It’s built around *site-first* discovery (home page + sitemap + likely high-value pages) with rate limits and safety checks.

---

## Quick start

### 0) Create a virtual environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Run discovery (business list)
Open `src/Unified_Web_Scraper_vNew2.py` and edit:
- `States_To_Search` (e.g. `FL, GA` or `ALL`)
- `Bussinesses_To_Scrape` (comma-separated keywords)

Then run:
```bash
python src/Unified_Web_Scraper_vNew2.py
```

You’ll get CSV outputs under `runs_smart_scrape/`.

### 2) Run website contact extraction (optional)
Create a CSV that includes either a `Website` column or an `Email` column.
A common pattern is to take the discovery CSV output and keep the rows with websites.

Then run:
```bash
python src/Executive_Finder_Unified.py path/to/your_input.csv
```

The script writes `*_results.csv` next to your input file.

---

## Sample data

- `sample_data/Example_clean_scraped_data.csv` shows a small example of a discovery output schema.
- `docs/images/web_data_enhancement_run.jpg` is a real run screenshot for context.

---

## Responsible use / compliance

This code is powerful. Use it like a professional:

- Only collect what you have a lawful basis to collect.
- Respect site Terms of Service and applicable laws.
- Prefer **business contact data** published by the business itself (official websites, directories, public registries).
- Rate-limit requests; avoid aggressive crawling; don’t bypass access controls.
- Do not use this to target individuals’ sensitive personal information.

If you’re using this commercially: consult counsel for your exact use case and jurisdiction.

---

## Notes

- Overture support is optional and requires DuckDB + httpfs/spatial extensions.
- Outputs can be large depending on state/keywords; start narrow, validate, then scale.


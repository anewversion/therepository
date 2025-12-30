# Changelog (high level)

This folder represents a working snapshot of a broader set of experiments.

## Included scripts
- Unified_Web_Scraper_vNew2.py
  - multi-source discovery (OSM; optional Overture)
  - keyword→tag mapping + fallback matching
  - quality scoring + filters
  - caching/retry/backoff; split/merge outputs

- Executive_Finder_Unified.py
  - website-first discovery (home + sitemap + likely pages)
  - extraction for publicly posted emails + contact signals
  - rate limits and safety checks


# Changelog (Combauto v131)

This changelog is seeded from the “AIChatAndChanges” notes included in the repo.
It’s intentionally short and human-readable.

## v131 — Open-source reference release
- Fixed a Decimal/float type mismatch in ML training-data prep (`prepare_training_data`) that caused a `TypeError` during profit-target calculations.
- Fixed a `KeyError: 'symbol'` in `trade_management_loop` by ensuring new `active_positions` entries include a `symbol` field.
- General cleanup around the ML pipeline so training can start reliably during bot startup.

## Notes
- This is a representative “works right now” release. It may not include every newer experimental idea (e.g., the latest SeqMag nuance),
  but it captures the real architecture: **statistical + ML hybrid**, star scoring, stateful order/position management, and live execution.

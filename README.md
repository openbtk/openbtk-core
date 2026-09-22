# metrics-data

Generated data, not source. Do not send pull requests here.

`.github/workflows/traffic-stats.yml` (on `main`) writes to this branch daily via
`scripts/update_traffic_stats.py`, folding GitHub's rolling 14-day traffic window
into `history.json` and regenerating the `badges/*.json` files the README's
shields.io endpoint badges read from.

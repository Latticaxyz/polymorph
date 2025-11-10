# Polymorph

**Polymorph** is a modular data pipeline for fetching, processing, and analyzing Polymarket data. It provides command-line tools for market data ingestion, transformation, processing, and analytics.

## Installation

```bash
pipx install polymorph
# or
pip install polymorph
```

## Usage

Run via CLI after installation:

```bash
polymorph fetch --months 3
polymorph process
polymorph mc run --market-id <token_id> --trials 5000
polymorph tune --study my_study --n-trials 50
```

## Requirements

Python 3.11+

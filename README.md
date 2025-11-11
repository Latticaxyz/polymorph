# Polymorph

**Polymorph** is a modular data pipeline for fetching, processing, and analyzing Polymarket data.

It provides command-line tools for market data ingestion, transformation, processing, and analytics.

## Installation

```bash
pipx install polymorph
# or
pip install polymorph
```

## Usage

Fetch market and trade data for the past 3 months and store results in `./data`:

```bash
polymorph fetch --months 3
```

Fetch only markets or trades

```bash
polymorph fetch --no-trades   # Skip trade data
polymorph fetch --no-gamma    # Skip market metadata
```

Change output directory

```bash
polymorph fetch --out ./custom_dir
```

Control concurrency and timeout

```bash
polymorph fetch --max-concurrency 16 --http-timeout 60
```

Process fetched data

After fetching, generate daily returns and aggregated trade statistics:

```bash
polymorph process
```

## Requirements

Python 3.11+

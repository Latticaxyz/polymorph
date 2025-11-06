# Polymorph Refactoring: Modular Class-Based Architecture

This document describes the major refactoring of the Polymorph codebase from a functional architecture to a modular, class-based pipeline framework.

## Overview

Polymorph is a **Polymarket data pipeline** that focuses on:
1. **Fetching** - Pulling in Polymarket data from multiple sources
2. **Processing** - Transforming data, selecting features, computing statistics
3. **Analyzing** - Running Monte Carlo simulations and parameter optimization

The refactoring introduces a clean separation of concerns with well-defined abstractions, making the codebase more maintainable, testable, and extensible.

## New Architecture

### Core Abstractions (`polymorph/core/`)

#### 1. **PipelineContext**
Shared context passed between all pipeline components containing:
- Settings configuration
- Run timestamp
- Data directory
- Metadata dictionary

```python
from polymorph.core import PipelineContext
from polymorph.config import settings
from datetime import datetime, timezone
from pathlib import Path

context = PipelineContext(
    settings=settings,
    run_timestamp=datetime.now(timezone.utc),
    data_dir=Path("data"),
)
```

#### 2. **DataSource** (Abstract Base Class)
Base class for all data sources. Each source implements:
- `name` property - unique identifier
- `fetch(**kwargs)` method - async data fetching
- `validate(data)` method - optional validation

```python
from polymorph.core import DataSource, PipelineContext
import polars as pl

class MySource(DataSource[pl.DataFrame]):
    @property
    def name(self) -> str:
        return "my_source"

    async def fetch(self, **kwargs) -> pl.DataFrame:
        # Fetch and return data
        return pl.DataFrame(...)
```

#### 3. **PipelineStage** (Abstract Base Class)
Base class for pipeline stages. Each stage implements:
- `name` property - stage identifier
- `execute(input_data)` method - async execution logic
- Optional `validate_input()` and `validate_output()` methods

```python
from polymorph.core import PipelineStage, PipelineContext

class MyStage(PipelineStage[InputType, OutputType]):
    @property
    def name(self) -> str:
        return "my_stage"

    async def execute(self, input_data: InputType) -> OutputType:
        # Process and return results
        return output
```

#### 4. **StorageBackend** (Abstract Base Class)
Abstract storage interface with Parquet implementation:
- `write(data, path)` - Write DataFrame
- `read(path)` - Read DataFrame
- `scan(path)` - Lazy scan DataFrame
- `exists(path)` - Check if file exists

```python
from polymorph.core import ParquetStorage

storage = ParquetStorage(base_dir="data")
storage.write(df, "raw/markets.parquet")
df = storage.read("raw/markets.parquet")
```

#### 5. **Retry Utilities**
Shared retry logic using Tenacity:

```python
from polymorph.core.retry import with_retry

@with_retry(max_attempts=5, min_wait=1.0, max_wait=10.0)
async def fetch_data(url: str):
    # Your API call here
    pass
```

### Data Sources (`polymorph/sources/`)

#### **GammaSource**
Fetches market metadata from Gamma API.

```python
from polymorph.sources import GammaSource

async with GammaSource(context) as source:
    markets_df = await source.fetch(active_only=True)
```

#### **CLOBSource**
Fetches price history and trades from CLOB/Data API.

```python
from polymorph.sources import CLOBSource

async with CLOBSource(context) as source:
    # Fetch price history
    prices = await source.fetch_prices_history(
        token_id="abc123",
        start_ts=start,
        end_ts=end,
        fidelity=60
    )

    # Fetch trades
    trades = await source.fetch_trades(
        market_ids=["market1", "market2"],
        since_ts=start
    )
```

#### **SubgraphSource**
GraphQL query interface to Goldsky API.

```python
from polymorph.sources import SubgraphSource

async with SubgraphSource(context) as source:
    result = await source.query(
        query="{ markets { id question } }",
        variables={}
    )
```

### Pipeline Stages (`polymorph/pipeline/`)

#### **FetchStage**
Orchestrates data ingestion from all sources.

```python
from polymorph.pipeline import FetchStage

stage = FetchStage(
    context=context,
    n_months=3,
    include_gamma=True,
    include_prices=True,
    include_trades=True,
    max_concurrency=8
)

result = await stage.execute()
# Returns FetchResult with paths and counts
```

#### **ProcessStage**
Transforms raw data into analysis-ready features.

```python
from polymorph.pipeline import ProcessStage

stage = ProcessStage(
    context=context,
    raw_dir="data/raw",
    processed_dir="data/processed"
)

result = await stage.execute()
# Returns ProcessResult with processed data paths
```

### Analysis Modules (`polymorph/sims/`)

#### **MonteCarloSimulator**
Monte Carlo simulation for price path forecasting.

```python
from polymorph.sims import MonteCarloSimulator

simulator = MonteCarloSimulator(
    context=context,
    clip_min=-0.99,
    clip_max=1.0
)

result = simulator.simulate(
    token_id="token123",
    trials=10000,
    horizon_days=7
)

print(f"Median return: {result.median_return}")
print(f"P(negative): {result.prob_negative}")
```

#### **ParameterSearcher**
Optuna-based parameter optimization.

```python
from polymorph.sims import ParameterSearcher

searcher = ParameterSearcher(context=context)

result = searcher.optimize(
    study_name="my_strategy",
    n_trials=100,
    direction="maximize"
)

print(f"Best params: {result.best_params}")
print(f"Best value: {result.best_value}")
```

### Data Models (`polymorph/models/`)

Pydantic models for type safety and validation:

#### API Models (`models/api.py`)
- `Market` - Market metadata
- `Token` - Token information
- `PricePoint` - Price data point
- `Trade` - Trade record

#### Pipeline Models (`models/pipeline.py`)
- `FetchResult` - Output from fetch stage
- `ProcessResult` - Output from process stage
- `AnalysisResult` - Output from analysis stage

#### Analysis Models (`models/analysis.py`)
- `SimulationResult` - Monte Carlo results
- `OptimizationResult` - Parameter search results

## Migration Guide

### For Users

The CLI remains **fully backward compatible**. All existing commands work as before:

```bash
# Fetch data (now uses FetchStage internally)
polymorph fetch --months 3

# Process data (now uses ProcessStage internally)
polymorph process

# Run Monte Carlo (now uses MonteCarloSimulator)
polymorph mc run --market-id token123 --trials 10000

# Parameter search (now uses ParameterSearcher)
polymorph tune --study my_study --n-trials 50
```

### For Developers

The old functional API is still available for backward compatibility:

```python
# Old (still works)
from polymorph.pipeline import fetch, process
from polymorph.sims import monte_carlo, param_search

# New (recommended)
from polymorph.pipeline import FetchStage, ProcessStage
from polymorph.sims import MonteCarloSimulator, ParameterSearcher
```

## Benefits of Refactoring

### 1. **Modularity**
- Clear separation between data sources, pipeline stages, and analysis
- Each component has a single, well-defined responsibility
- Easy to add new sources or stages without modifying existing code

### 2. **Testability**
- Each class can be unit tested independently
- Dependency injection makes mocking straightforward
- 22 new tests covering core abstractions and models

### 3. **Type Safety**
- Pydantic models provide runtime validation
- Generic types ensure correct data flow between stages
- Better IDE support and autocomplete

### 4. **Extensibility**
- Add new data sources by extending `DataSource`
- Add new pipeline stages by extending `PipelineStage`
- Add new storage backends by extending `StorageBackend`
- Custom objective functions for parameter search

### 5. **Maintainability**
- Clear interfaces and contracts
- Comprehensive logging throughout
- Shared retry logic reduces duplication
- Better error handling and validation

### 6. **Flexibility**
- Mix and match components as needed
- Run stages independently or as a pipeline
- Easy to create custom workflows

## File Structure

```
polymorph/
├── core/                    # NEW: Core abstractions
│   ├── __init__.py
│   ├── base.py             # DataSource, PipelineStage, PipelineContext
│   ├── storage.py          # Storage backends
│   └── retry.py            # Retry utilities
│
├── models/                  # NEW: Data models
│   ├── __init__.py
│   ├── api.py              # API response models
│   ├── pipeline.py         # Pipeline stage I/O models
│   └── analysis.py         # Analysis result models
│
├── sources/
│   ├── __init__.py         # UPDATED: Exports both old and new
│   ├── gamma.py            # OLD: Functional API (kept for compatibility)
│   ├── gamma_source.py     # NEW: GammaSource class
│   ├── clob.py             # OLD: Functional API (kept for compatibility)
│   ├── clob_source.py      # NEW: CLOBSource class
│   ├── subgraph.py         # OLD: Functional API (kept for compatibility)
│   └── subgraph_source.py  # NEW: SubgraphSource class
│
├── pipeline/
│   ├── __init__.py         # UPDATED: Exports both old and new
│   ├── fetch.py            # OLD: Functional API (kept for compatibility)
│   ├── fetch_stage.py      # NEW: FetchStage class
│   ├── process.py          # OLD: Functional API (kept for compatibility)
│   └── process_stage.py    # NEW: ProcessStage class
│
├── sims/
│   ├── __init__.py         # UPDATED: Exports both old and new
│   ├── monte_carlo.py      # OLD: Functional API (kept for compatibility)
│   ├── monte_carlo_simulator.py  # NEW: MonteCarloSimulator class
│   ├── param_search.py     # OLD: Functional API (kept for compatibility)
│   └── parameter_searcher.py     # NEW: ParameterSearcher class
│
├── utils/
│   └── logging.py          # UPDATED: Added get_logger()
│
└── cli.py                  # UPDATED: Uses new class-based API
```

## Testing

New test suite covering core functionality:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_core.py
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

Test coverage:
- ✅ Core abstractions (PipelineContext, DataSource, PipelineStage)
- ✅ Storage backend (ParquetStorage)
- ✅ Data models (API, Pipeline, Analysis)
- ✅ Import tests for all new modules

## Future Enhancements

With the new architecture, the following enhancements are now easier:

1. **Pipeline Orchestration**
   - Chain stages together with automatic data flow
   - Parallel execution of independent stages
   - Caching and checkpointing

2. **Additional Data Sources**
   - Social media sentiment
   - News aggregators
   - Alternative prediction markets

3. **Advanced Processing**
   - Feature engineering modules
   - Data quality checks
   - Automated data validation

4. **Analysis Extensions**
   - Additional simulation strategies
   - Risk metrics and VaR calculations
   - Backtesting framework

5. **Monitoring & Observability**
   - Metrics collection
   - Performance tracking
   - Data lineage

## Summary

This refactoring transforms Polymorph from a collection of scripts into a **robust, modular pipeline framework** while maintaining full backward compatibility. The new architecture provides a solid foundation for future development and makes the codebase significantly more maintainable and extensible.

**Key Points:**
- ✅ All existing functionality preserved
- ✅ CLI remains fully compatible
- ✅ 22 new tests, all passing
- ✅ Clean abstractions and interfaces
- ✅ Type-safe with Pydantic models
- ✅ Comprehensive logging
- ✅ Ready for future enhancements

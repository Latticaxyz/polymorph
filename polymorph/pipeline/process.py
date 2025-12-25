from pathlib import Path

import polars as pl

from polymorph.core.base import PipelineContext, PipelineStage
from polymorph.models.pipeline import FetchResult, ProcessInputConfig, ProcessResult
from polymorph.utils.constants import MS_PER_DAY
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)


class ProcessStage(PipelineStage[FetchResult | None, ProcessResult]):
    def __init__(
        self,
        context: PipelineContext,
        raw_dir: str | Path | None = None,
        processed_dir: str | Path | None = None,
        input_config: ProcessInputConfig | None = None,
    ):
        super().__init__(context)

        self.storage = context.storage

        self.raw_dir = Path(raw_dir) if raw_dir else Path("raw")
        self.processed_dir = Path(processed_dir) if processed_dir else Path("processed")
        self.input_config = input_config or ProcessInputConfig(raw_dir=self.raw_dir)

    def _stamp(self) -> str:
        return self.context.run_timestamp.strftime("%Y%m%dT%H%M%SZ")

    def _output_dir(self) -> Path:
        return self.processed_dir / self._stamp()

    @property
    def name(self) -> str:
        return "process"

    def _load_markets(self) -> pl.LazyFrame | None:
        mode = self.input_config.mode

        if mode == "explicit_files":
            if self.input_config.markets_file is None:
                return None
            try:
                return pl.scan_parquet(str(self.input_config.markets_file))
            except Exception as e:
                logger.warning(f"Could not load markets file: {e}")
                return None

        if mode == "run_dir":
            if self.input_config.run_dir is None:
                return None
            markets_path = self.input_config.run_dir / "markets.parquet"
            resolved = self.storage._resolve_path(markets_path)
            if not resolved.exists():
                logger.warning(f"Markets file not found: {resolved}")
                return None
            try:
                return pl.scan_parquet(str(resolved))
            except Exception as e:
                logger.warning(f"Could not load markets from run_dir: {e}")
                return None

        if not self.storage._resolve_path(self.raw_dir).exists():
            logger.warning(f"Raw directory does not exist: {self.raw_dir}")
            return None

        markets_pattern = self.raw_dir / "*" / "markets.parquet"
        try:
            return self.storage.scan(markets_pattern)
        except Exception as e:
            logger.warning(f"Could not scan markets: {e}")
            return None

    def _load_prices(self) -> pl.LazyFrame | None:
        mode = self.input_config.mode

        if mode == "explicit_files":
            if self.input_config.prices_file is None:
                return None
            try:
                return pl.scan_parquet(str(self.input_config.prices_file))
            except Exception as e:
                logger.warning(f"Could not load prices file: {e}")
                return None

        if mode == "run_dir":
            if self.input_config.run_dir is None:
                return None
            prices_path = self.input_config.run_dir / "prices.parquet"
            resolved = self.storage._resolve_path(prices_path)
            if resolved.exists():
                try:
                    return pl.scan_parquet(str(resolved))
                except Exception as e:
                    logger.warning(f"Could not load consolidated prices: {e}")

            resolved_dir = self.storage._resolve_path(self.input_config.run_dir)
            part_files = list(resolved_dir.glob("prices_part*.parquet"))
            if part_files:
                try:
                    return pl.scan_parquet([str(f) for f in sorted(part_files)])
                except Exception as e:
                    logger.warning(f"Could not load prices part files: {e}")
                    return None

            logger.warning(f"No prices files found in run_dir: {resolved_dir}")
            return None

        if not self.storage._resolve_path(self.raw_dir).exists():
            logger.warning(f"Raw directory does not exist: {self.raw_dir}")
            return None

        prices_pattern = self.raw_dir / "*" / "prices*.parquet"
        try:
            return self.storage.scan(prices_pattern)
        except Exception as e:
            logger.warning(f"Could not scan prices: {e}")
            return None

    def _load_trades(self) -> pl.LazyFrame | None:
        mode = self.input_config.mode

        if mode == "explicit_files":
            if self.input_config.trades_file is None:
                return None
            try:
                return pl.scan_parquet(str(self.input_config.trades_file))
            except Exception as e:
                logger.warning(f"Could not load trades file: {e}")
                return None

        if mode == "run_dir":
            if self.input_config.run_dir is None:
                return None
            trades_path = self.input_config.run_dir / "trades.parquet"
            resolved = self.storage._resolve_path(trades_path)
            if not resolved.exists():
                logger.warning(f"Trades file not found: {resolved}")
                return None
            try:
                return pl.scan_parquet(str(resolved))
            except Exception as e:
                logger.warning(f"Could not load trades from run_dir: {e}")
                return None

        if not self.storage._resolve_path(self.raw_dir).exists():
            logger.warning(f"Raw directory does not exist: {self.raw_dir}")
            return None

        trades_pattern = self.raw_dir / "*" / "trades.parquet"
        try:
            return self.storage.scan(trades_pattern)
        except Exception as e:
            logger.warning(f"Could not scan trades: {e}")
            return None

    def _build_token_market_map(self) -> pl.LazyFrame | None:
        lf = self._load_markets()
        if lf is None:
            return None

        return (
            lf.with_row_index("_row")
            .select(
                [
                    "_row",
                    pl.col("id").alias("market_id"),
                    "question",
                    "condition_id",
                    "token_ids",
                    "outcomes",
                    "resolved",
                    "resolution_outcome",
                    "category",
                ]
            )
            .explode("token_ids")
            .with_columns(pl.col("token_ids").cum_count().over("_row").alias("_outcome_idx"))
            .with_columns(pl.col("outcomes").list.get(pl.col("_outcome_idx") - 1).alias("outcome_name"))
            .drop(["_row", "_outcome_idx", "outcomes"])
            .rename({"token_ids": "token_id"})
        )

    def build_enriched_prices(self) -> ProcessResult:
        logger.info("Building enriched prices")

        result = ProcessResult(run_timestamp=self.context.run_timestamp)

        token_map = self._build_token_market_map()
        if token_map is None:
            logger.warning("Could not build token-market mapping")
            return result

        prices_lf = self._load_prices()
        if prices_lf is None:
            logger.warning("Could not load prices data")
            return result

        schema = prices_lf.collect_schema()
        timestamp_col = "t" if "t" in schema.names() else "timestamp"
        price_col = "p" if "p" in schema.names() else "price"

        enriched = (
            prices_lf.select(["token_id", timestamp_col, price_col])
            .rename({timestamp_col: "t", price_col: "p"})
            .join(token_map, on="token_id", how="inner")
            .collect()
        )

        if enriched.height == 0:
            logger.warning("No enriched prices after join")
            return result

        output_path = self._output_dir() / "prices_enriched.parquet"
        self.storage.write(enriched, output_path)
        result.prices_enriched_path = self.storage._resolve_path(output_path)
        result.enriched_count = enriched.height

        logger.info(f"Enriched prices built: {result.enriched_count} rows -> {output_path}")
        return result

    def build_daily_returns(self) -> ProcessResult:
        logger.info("Building daily returns")

        result = ProcessResult(
            run_timestamp=self.context.run_timestamp,
        )

        lf = self._load_prices()
        if lf is None:
            logger.warning("Could not load prices data")
            return result

        schema = lf.collect_schema()
        required_cols = {"t", "p", "token_id"}

        if not required_cols.issubset(schema.names()):
            alt_cols = {"timestamp", "price", "token_id"}
            if not alt_cols.issubset(schema.names()):
                logger.warning(
                    f"Price data missing required columns. "
                    f"Found: {schema.names()}, Need: {required_cols} or {alt_cols}"
                )
                return result

            timestamp_col = "timestamp"
            price_col = "price"
        else:
            timestamp_col = "t"
            price_col = "p"

        token_map = self._build_token_market_map()

        daily_prices = (
            lf.with_columns((pl.col(timestamp_col).cast(pl.Int64) // MS_PER_DAY * MS_PER_DAY).alias("day_ts"))
            .group_by(["token_id", "day_ts"])
            .agg(pl.col(price_col).mean().alias("price_day"))
            .sort(["token_id", "day_ts"])
            .with_columns(pl.col("price_day").pct_change().over("token_id").alias("ret"))
        )

        if token_map is not None:
            daily_returns = daily_prices.join(token_map, on="token_id", how="left").collect()
        else:
            daily_returns = daily_prices.collect()

        output_path = self._output_dir() / "daily_returns.parquet"
        self.storage.write(daily_returns, output_path)
        result.daily_returns_path = self.storage._resolve_path(output_path)
        result.returns_count = daily_returns.height

        logger.info(f"Daily returns built: {result.returns_count} rows -> {output_path}")

        return result

    def build_price_panel(self) -> ProcessResult:
        logger.info("Building price panel (wide format)")

        result = ProcessResult(run_timestamp=self.context.run_timestamp)

        lf = self._load_prices()
        if lf is None:
            logger.warning("Could not load prices data")
            return result

        schema = lf.collect_schema()
        timestamp_col = "t" if "t" in schema.names() else "timestamp"
        price_col = "p" if "p" in schema.names() else "price"

        daily_prices = (
            lf.with_columns((pl.col(timestamp_col).cast(pl.Int64) // MS_PER_DAY * MS_PER_DAY).alias("day_ts"))
            .group_by(["token_id", "day_ts"])
            .agg(pl.col(price_col).mean().alias("price"))
            .collect()
        )

        if daily_prices.height == 0:
            logger.warning("No daily prices to pivot")
            return result

        panel = daily_prices.pivot(on="token_id", index="day_ts", values="price").sort("day_ts")

        output_path = self._output_dir() / "price_panel.parquet"
        self.storage.write(panel, output_path)
        result.price_panel_path = self.storage._resolve_path(output_path)
        result.panel_days = panel.height
        result.panel_tokens = panel.width - 1

        logger.info(f"Price panel built: {result.panel_days} days x {result.panel_tokens} tokens -> {output_path}")
        return result

    def build_trade_aggregates(self) -> ProcessResult:
        logger.info("Building trade aggregates")

        result = ProcessResult(
            run_timestamp=self.context.run_timestamp,
        )

        lf = self._load_trades()
        if lf is None:
            logger.warning("Could not load trades data")
            return result

        schema = lf.collect_schema()
        required_cols = {"timestamp", "size", "price", "conditionId"}

        if not required_cols.issubset(schema.names()):
            logger.warning(f"Trade data missing required columns. " f"Found: {schema.names()}, Need: {required_cols}")
            return result

        trade_agg = (
            lf.with_columns(
                [
                    (pl.col("timestamp").cast(pl.Int64) // MS_PER_DAY * MS_PER_DAY).alias("day_ts"),
                    (pl.col("size") * pl.col("price")).alias("notional"),
                ]
            )
            .group_by(["conditionId", "day_ts"])
            .agg(
                [
                    pl.len().alias("trades"),
                    pl.col("size").sum().alias("size_sum"),
                    pl.col("notional").sum().alias("notional_sum"),
                ]
            )
            .collect()
        )

        output_path = self._output_dir() / "trades_daily_agg.parquet"
        self.storage.write(trade_agg, output_path)
        result.trades_daily_agg_path = self.storage._resolve_path(output_path)
        result.trade_agg_count = trade_agg.height

        logger.info(f"Trade aggregates built: {result.trade_agg_count} rows -> {output_path}")

        return result

    async def execute(self, _input_data: FetchResult | None = None) -> ProcessResult:
        _ = _input_data

        logger.info("Starting process stage")

        enriched_result = self.build_enriched_prices()
        returns_result = self.build_daily_returns()
        panel_result = self.build_price_panel()
        trades_result = self.build_trade_aggregates()

        result = ProcessResult(
            run_timestamp=self.context.run_timestamp,
            prices_enriched_path=enriched_result.prices_enriched_path,
            enriched_count=enriched_result.enriched_count,
            daily_returns_path=returns_result.daily_returns_path,
            returns_count=returns_result.returns_count,
            price_panel_path=panel_result.price_panel_path,
            panel_days=panel_result.panel_days,
            panel_tokens=panel_result.panel_tokens,
            trades_daily_agg_path=trades_result.trades_daily_agg_path,
            trade_agg_count=trades_result.trade_agg_count,
        )

        logger.info(
            f"Process stage complete: {result.enriched_count} enriched prices, "
            f"{result.returns_count} returns, {result.panel_days}x{result.panel_tokens} panel, "
            f"{result.trade_agg_count} trade aggregates"
        )

        return result

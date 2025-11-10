from pathlib import Path

import polars as pl

from polymorph.core.base import PipelineStage, PipelineContext
from polymorph.core.storage import ParquetStorage
from polymorph.models.pipeline import FetchResult, ProcessResult
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)


class ProcessStage(PipelineStage[FetchResult | None, ProcessResult]):
    def __init__(
        self,
        context: PipelineContext,
        raw_dir: str | Path | None = None,
        processed_dir: str | Path | None = None,
    ):
        super().__init__(context)

        self.storage = ParquetStorage(context.data_dir)

        self.raw_dir = Path(raw_dir) if raw_dir else context.data_dir / "raw"
        self.processed_dir = (
            Path(processed_dir) if processed_dir else context.data_dir / "processed"
        )

    @property
    def name(self) -> str:
        return "process"

    def build_daily_returns(self) -> ProcessResult:
        logger.info("Building daily returns")

        result = ProcessResult(
            run_timestamp=self.context.run_timestamp,
        )

        prices_dir = self.raw_dir / "clob"
        prices_pattern = prices_dir / "*_prices.parquet"

        if not prices_dir.exists():
            logger.warning(f"Prices directory does not exist: {prices_dir}")
            return result

        try:
            lf = self.storage.scan(prices_pattern)
        except Exception as e:
            logger.warning(f"Could not scan prices: {e}")
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

        daily_returns = (
            lf.with_columns(
                (pl.col(timestamp_col).cast(pl.Int64) // 86_400 * 86_400).alias(
                    "day_ts"
                )
            )
            .group_by(["token_id", "day_ts"])
            .agg(pl.col(price_col).mean().alias("price_day"))
            .sort(["token_id", "day_ts"])
            .with_columns(
                pl.col("price_day").pct_change().over("token_id").alias("ret")
            )
            .collect()
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / "daily_returns.parquet"

        self.storage.write(daily_returns, output_path)
        result.daily_returns_path = output_path
        result.returns_count = daily_returns.height

        logger.info(
            f"Daily returns built: {result.returns_count} rows -> {output_path}"
        )

        return result

    def build_trade_aggregates(self) -> ProcessResult:
        logger.info("Building trade aggregates")

        result = ProcessResult(
            run_timestamp=self.context.run_timestamp,
        )

        trades_dir = self.raw_dir / "clob"
        trades_pattern = trades_dir / "*_trades.parquet"

        if not trades_dir.exists():
            trades_dir = self.raw_dir / "data_api"
            trades_pattern = trades_dir / "trades.parquet"

        if not trades_dir.exists():
            logger.warning(f"Trades directory does not exist: {trades_dir}")
            return result

        try:
            lf = self.storage.scan(trades_pattern)
        except Exception as e:
            logger.warning(f"Could not scan trades: {e}")
            return result

        schema = lf.collect_schema()
        required_cols = {"timestamp", "size", "price", "conditionId"}

        if not required_cols.issubset(schema.names()):
            logger.warning(
                f"Trade data missing required columns. "
                f"Found: {schema.names()}, Need: {required_cols}"
            )
            return result

        trade_agg = (
            lf.with_columns(
                [
                    (pl.col("timestamp").cast(pl.Int64) // 86_400 * 86_400).alias(
                        "day_ts"
                    ),
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

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / "trades_daily_agg.parquet"

        self.storage.write(trade_agg, output_path)
        result.trades_daily_agg_path = output_path
        result.trade_agg_count = trade_agg.height

        logger.info(
            f"Trade aggregates built: {result.trade_agg_count} rows -> {output_path}"
        )

        return result

    async def execute(self, input_data: FetchResult | None = None) -> ProcessResult:
        logger.info("Starting process stage")

        returns_result = self.build_daily_returns()
        trades_result = self.build_trade_aggregates()

        result = ProcessResult(
            run_timestamp=self.context.run_timestamp,
            daily_returns_path=returns_result.daily_returns_path,
            trades_daily_agg_path=trades_result.trades_daily_agg_path,
            returns_count=returns_result.returns_count,
            trade_agg_count=trades_result.trade_agg_count,
        )

        logger.info(
            f"Process stage complete: {result.returns_count} returns, "
            f"{result.trade_agg_count} trade aggregates"
        )

        return result

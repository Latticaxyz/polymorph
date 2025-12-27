from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from polymorph.core.base import PipelineContext
from polymorph.models.pipeline import FilterConfig, FilterResult
from polymorph.utils.constants import MS_PER_DAY
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)


def _date_to_timestamp_ms(d: date) -> int:
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class FilterStage:
    def __init__(
        self,
        context: PipelineContext,
        filter_config: FilterConfig,
        input_path: Path,
        output_path: Path,
    ):
        self.context = context
        self.filter_config = filter_config
        self.input_path = input_path
        self.output_path = output_path
        self.storage = context.storage

    def _build_filter_expressions(self, schema: pl.Schema) -> tuple[list[pl.Expr], list[str]]:
        expressions: list[pl.Expr] = []
        applied: list[str] = []
        config = self.filter_config
        columns = set(schema.names())

        if config.start_date is not None and "t" in columns:
            start_ms = _date_to_timestamp_ms(config.start_date)
            expressions.append(pl.col("t") >= start_ms)
            applied.append(f"start_date>={config.start_date}")

        if config.end_date is not None and "t" in columns:
            end_ms = _date_to_timestamp_ms(config.end_date) + MS_PER_DAY - 1
            expressions.append(pl.col("t") <= end_ms)
            applied.append(f"end_date<={config.end_date}")

        if config.resolved_only and "resolved" in columns:
            expressions.append(pl.col("resolved") == True)  # noqa: E712
            applied.append("resolved_only")

        if config.unresolved_only and "resolved" in columns:
            expressions.append(pl.col("resolved") == False)  # noqa: E712
            applied.append("unresolved_only")

        if config.min_age_days is not None and "created_at" in columns:
            cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
                days=config.min_age_days
            )
            cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")
            expressions.append(pl.col("created_at") <= cutoff_str)
            applied.append(f"min_age_days>={config.min_age_days}")

        if config.max_age_days is not None and "created_at" in columns:
            cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
                days=config.max_age_days
            )
            cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")
            expressions.append(pl.col("created_at") >= cutoff_str)
            applied.append(f"max_age_days<={config.max_age_days}")

        if config.categories and "category" in columns:
            expressions.append(pl.col("category").is_in(config.categories))
            applied.append(f"categories={config.categories}")

        if config.exclude_categories and "category" in columns:
            expressions.append(~pl.col("category").is_in(config.exclude_categories))
            applied.append(f"exclude_categories={config.exclude_categories}")

        if config.market_ids and "market_id" in columns:
            expressions.append(pl.col("market_id").is_in(config.market_ids))
            applied.append(f"market_ids={len(config.market_ids)} ids")

        if config.exclude_market_ids and "market_id" in columns:
            expressions.append(~pl.col("market_id").is_in(config.exclude_market_ids))
            applied.append(f"exclude_market_ids={len(config.exclude_market_ids)} ids")

        return expressions, applied

    def _needs_jump_computation(self) -> bool:
        config = self.filter_config
        return (
            config.compute_jumps
            or config.min_jump_pct is not None
            or config.max_jump_pct is not None
            or config.min_jump_abs is not None
            or config.max_jump_abs is not None
        )

    def _compute_jumps(self, df: pl.DataFrame) -> pl.DataFrame:
        if "token_id" not in df.columns or "t" not in df.columns or "p" not in df.columns:
            logger.warning("Cannot compute jumps: missing token_id, t, or p columns")
            return df

        return df.sort(["token_id", "t"]).with_columns(
            [
                (pl.col("p") - pl.col("p").shift(1).over("token_id")).alias("jump_abs"),
                ((pl.col("p") - pl.col("p").shift(1).over("token_id")) / pl.col("p").shift(1).over("token_id")).alias(
                    "jump_pct"
                ),
            ]
        )

    def _build_jump_filter_expressions(self) -> tuple[list[pl.Expr], list[str]]:
        expressions: list[pl.Expr] = []
        applied: list[str] = []
        config = self.filter_config

        if config.min_jump_pct is not None:
            expressions.append(pl.col("jump_pct").abs() >= config.min_jump_pct)
            applied.append(f"min_jump_pct>={config.min_jump_pct}")

        if config.max_jump_pct is not None:
            expressions.append(pl.col("jump_pct").abs() <= config.max_jump_pct)
            applied.append(f"max_jump_pct<={config.max_jump_pct}")

        if config.min_jump_abs is not None:
            expressions.append(pl.col("jump_abs").abs() >= config.min_jump_abs)
            applied.append(f"min_jump_abs>={config.min_jump_abs}")

        if config.max_jump_abs is not None:
            expressions.append(pl.col("jump_abs").abs() <= config.max_jump_abs)
            applied.append(f"max_jump_abs<={config.max_jump_abs}")

        return expressions, applied

    async def execute(self) -> FilterResult:
        logger.info(f"Starting filter stage: {self.input_path}")

        result = FilterResult(input_path=self.input_path)

        if not self.input_path.exists():
            logger.warning(f"Input file not found: {self.input_path}")
            return result

        lf = pl.scan_parquet(str(self.input_path))
        schema = lf.collect_schema()

        input_count = lf.select(pl.len()).collect().item()
        result.input_count = input_count

        expressions, applied = self._build_filter_expressions(schema)

        if not expressions:
            filtered = lf.collect()
        else:
            combined_filter = expressions[0]
            for expr in expressions[1:]:
                combined_filter = combined_filter & expr
            filtered = lf.filter(combined_filter).collect()

        if self._needs_jump_computation():
            filtered = self._compute_jumps(filtered)
            jump_exprs, jump_applied = self._build_jump_filter_expressions()
            applied.extend(jump_applied)

            if jump_exprs:
                combined_jump = jump_exprs[0]
                for expr in jump_exprs[1:]:
                    combined_jump = combined_jump & expr
                filtered = filtered.filter(combined_jump)

            if not self.filter_config.compute_jumps:
                filtered = filtered.drop(["jump_abs", "jump_pct"], strict=False)

        result.filters_applied = applied
        result.output_count = filtered.height

        filtered.write_parquet(self.output_path)
        result.output_path = self.output_path

        logger.info(
            f"Filter stage complete: {result.input_count} -> {result.output_count} rows "
            f"({len(applied)} filters applied)"
        )

        return result

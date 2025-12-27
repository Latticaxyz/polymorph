from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator

from polymorph.utils.parse import parse_timestamp_ms


class Market(BaseModel):
    id: str
    question: str | None = None
    description: str | None = None
    market_slug: str | None = Field(default=None, alias="marketSlug")
    condition_id: str | None = Field(default=None, alias="conditionId")
    clob_token_ids: list[str] = Field(default_factory=list, alias="clobTokenIds")
    outcomes: list[str] | None = None
    active: bool | None = None
    closed: bool | None = None
    archived: bool | None = None
    created_at: str | None = Field(default=None, alias="createdAt")
    end_date: str | None = Field(default=None, alias="endDate")
    resolved: bool | None = None
    resolution_date: str | None = Field(default=None, alias="resolutionDate")
    resolution_outcome: str | None = Field(default=None, alias="resolutionOutcome")
    tags: list[str] = Field(default_factory=list)
    category: str | None = None
    rewards: dict[str, float] | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("outcomes", mode="before")
    @classmethod
    def validate_outcomes(cls, v: object) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return None


class OrderBookLevel(BaseModel):
    price: float
    size: float

    @field_validator("price", "size", mode="before")
    @classmethod
    def validate_decimal(cls, v: object) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            return float(v)
        raise ValueError(f"Cannot convert {type(v).__name__} to float")


class OrderBook(BaseModel):
    token_id: str
    timestamp: int  # Unix milliseconds
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    mid_price: float | None = None
    spread: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None

    model_config = {"extra": "ignore"}

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: object) -> int:
        return parse_timestamp_ms(v)

    def calculate_spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    def calculate_mid_price(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    def get_depth_at_distance(self, distance: float, side: str = "both") -> float:
        if self.mid_price is None:
            return 0.0

        depth = 0.0

        if side in ("bid", "both"):
            for level in self.bids:
                if level.price >= self.mid_price - distance:
                    depth += level.size

        if side in ("ask", "both"):
            for level in self.asks:
                if level.price <= self.mid_price + distance:
                    depth += level.size

        return depth

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from polymorph.utils.parse import parse_decimal_string, parse_timestamp_ms


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

    model_config = {"populate_by_name": True, "extra": "forbid"}


class Token(BaseModel):
    token_id: str = Field(..., alias="tokenId")
    outcome: str | None = None
    market_id: str | None = Field(default=None, alias="marketId")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class PricePoint(BaseModel):
    t: int  # Unix milliseconds
    p: str  # Decimal string
    token_id: str | None = Field(default=None, alias="tokenId")

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @field_validator("t", mode="before")
    @classmethod
    def validate_timestamp(cls, v: object) -> int:
        return parse_timestamp_ms(v)

    @field_validator("p", mode="before")
    @classmethod
    def validate_price(cls, v: object) -> str:
        return parse_decimal_string(v)


class Trade(BaseModel):
    id: str
    market: str  # condition_id
    asset_id: str = Field(..., alias="assetId")  # token_id
    condition_id: str | None = Field(default=None, alias="conditionId")
    side: str  # "BUY" or "SELL"
    size: str  # Decimal string
    price: str  # Decimal string
    fee_rate_bps: int | None = Field(default=None, alias="feeRateBps")
    status: str | None = None
    timestamp: int  # Unix milliseconds
    maker_address: str | None = Field(default=None, alias="makerAddress")
    match_time: str | None = Field(default=None, alias="matchTime")

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: object) -> int:
        return parse_timestamp_ms(v)

    @field_validator("price", "size", mode="before")
    @classmethod
    def validate_decimal(cls, v: object) -> str:
        return parse_decimal_string(v)


class OrderBookLevel(BaseModel):
    price: str  # Decimal string
    size: str  # Decimal string

    @field_validator("price", "size", mode="before")
    @classmethod
    def validate_decimal(cls, v: object) -> str:
        return parse_decimal_string(v)


class OrderBook(BaseModel):
    token_id: str
    timestamp: int  # Unix milliseconds
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    mid_price: float | None = None
    spread: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None

    model_config = {"extra": "forbid"}

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
                # Convert string price to float for comparison
                level_price = float(level.price)
                if level_price >= self.mid_price - distance:
                    depth += float(level.size)

        if side in ("ask", "both"):
            for level in self.asks:
                # Convert string price to float for comparison
                level_price = float(level.price)
                if level_price <= self.mid_price + distance:
                    depth += float(level.size)

        return depth


class MarketResolution(BaseModel):
    market_id: str
    condition_id: str | None = None
    outcome: str
    resolution_timestamp: int | None = None  # Unix milliseconds
    resolution_date: str | None = None
    winning_outcome_price: float | None = None

    model_config = {"extra": "forbid"}

    @field_validator("resolution_timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: object) -> int | None:
        if v is None:
            return None
        return parse_timestamp_ms(v)

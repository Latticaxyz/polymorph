from pydantic import BaseModel, Field, ValidationInfo, field_validator


class Market(BaseModel):
    id: str | None = None
    question: str | None = None
    description: str | None = None
    market_slug: str | None = Field(None, alias="marketSlug")
    condition_id: str | None = Field(None, alias="conditionId")
    clob_token_ids: list[str] | None = Field(None, alias="clobTokenIds")
    outcomes: list[str] | None = None
    active: bool | None = None
    closed: bool | None = None
    archived: bool | None = None
    created_at: str | None = Field(None, alias="createdAt")
    end_date: str | None = Field(None, alias="endDate")
    resolved: bool | None = None
    resolution_date: str | None = Field(None, alias="resolutionDate")
    resolution_outcome: str | None = Field(None, alias="resolutionOutcome")
    tags: list[str] | None = None
    category: str | None = None
    rewards: dict[str, float] | None = None

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("clob_token_ids", mode="before")
    @classmethod
    def normalize_token_ids(_cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str):
            import json

            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        return [str(x) for x in arr if x is not None]
                except Exception:
                    return [s]
            if "," in s:
                return [t.strip() for t in s.split(",") if t.strip()]
            return [s]
        return [str(v)]


class Token(BaseModel):
    token_id: str = Field(..., alias="tokenId")
    outcome: str | None = None
    market_id: str | None = Field(None, alias="marketId")

    model_config = {"populate_by_name": True}


class PricePoint(BaseModel):
    t: int  # timestamp
    p: float  # price
    token_id: str | None = Field(None, alias="tokenId")

    model_config = {"populate_by_name": True}


class Trade(BaseModel):
    id: str | None = None
    market: str | None = None
    asset_id: str | None = Field(None, alias="assetId")
    condition_id: str | None = Field(None, alias="conditionId")
    side: str | None = None
    size: float | None = None
    price: float | None = None
    fee_rate_bps: int | None = Field(None, alias="feeRateBps")
    status: str | None = None
    created_at: str | None = Field(None, alias="createdAt")
    timestamp: int | None = None
    maker_address: str | None = Field(None, alias="makerAddress")
    match_time: str | None = Field(None, alias="matchTime")

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(_cls, v: object, info: ValidationInfo) -> int | None:
        if v is not None and isinstance(v, int):
            return v

        # Try to get from created_at
        created_at = info.data.get("created_at")
        if created_at and isinstance(created_at, str):
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                return int(dt.timestamp())
            except (ValueError, OSError):
                # Invalid datetime format or out of range - return None
                return None
        return None


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    token_id: str
    timestamp: int
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    mid_price: float | None = None
    spread: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None

    model_config = {"arbitrary_types_allowed": True}

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


class MarketResolution(BaseModel):
    market_id: str
    condition_id: str | None = None
    outcome: str  # YES, NO, INVALID
    resolution_timestamp: int | None = None
    resolution_date: str | None = None
    winning_outcome_price: float | None = None

    model_config = {"arbitrary_types_allowed": True}

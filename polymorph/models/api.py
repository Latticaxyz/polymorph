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

class PolymarketAPIError(Exception):
    """Base exception for API-related errors."""

    pass


class StrictParsingError(PolymarketAPIError):
    """Raised when strict parsing fails due to unexpected format."""

    pass


class InvalidTimestampError(PolymarketAPIError):
    """Raised when timestamp is invalid or out of expected range."""

    pass


class InvalidResponseFormatError(PolymarketAPIError):
    """Raised when API response doesn't match expected format."""

    pass

from typing import Any

from polymorph.utils.constants import MAX_VALID_TIMESTAMP_MS, MIN_VALID_TIMESTAMP_MS


def parse_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected str, got {type(value).__name__}: {value}")
    return value


def parse_decimal_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Price/size must be string, got {type(value).__name__}: {value}")

    # Validate it's a valid decimal
    try:
        float(value)
    except ValueError:
        raise ValueError(f"Invalid decimal string: {value}")

    return value


def parse_decimal_flexible(value: Any) -> str:
    if isinstance(value, str):
        # Validate it's a valid decimal string
        try:
            float(value)
        except ValueError:
            raise ValueError(f"Invalid decimal string: {value}")
        return value
    elif isinstance(value, (int, float)):
        # API returned number - convert to string
        return str(value)
    else:
        raise ValueError(f"Expected number or string, got {type(value).__name__}: {value}")


def parse_timestamp_ms(value: Any) -> int:
    if isinstance(value, str):
        try:
            ts = int(value)
        except ValueError:
            raise ValueError(f"Timestamp string not parseable: {value}")
    elif isinstance(value, int):
        ts = value
    else:
        raise ValueError(f"Timestamp must be str or int, got {type(value).__name__}: {value}")

    # Allow 0 as a special value for unknown/invalid timestamps
    if ts == 0:
        return 0

    if ts < MIN_VALID_TIMESTAMP_MS or ts > MAX_VALID_TIMESTAMP_MS:
        raise ValueError(
            f"Timestamp {ts} out of valid range (must be milliseconds between "
            f"{MIN_VALID_TIMESTAMP_MS} and {MAX_VALID_TIMESTAMP_MS})"
        )

    return ts

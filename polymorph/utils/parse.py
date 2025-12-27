from polymorph.utils.constants import MAX_VALID_TIMESTAMP_MS, MIN_VALID_TIMESTAMP_MS


def parse_timestamp_ms(value: object) -> int:
    if isinstance(value, str):
        try:
            ts = int(value)
        except ValueError:
            raise ValueError(f"Timestamp string not parseable: {value}")
    elif isinstance(value, int):
        ts = value
    else:
        raise ValueError(f"Timestamp must be str or int, got {type(value).__name__}: {value}")

    if ts == 0:
        return 0

    if ts < MIN_VALID_TIMESTAMP_MS or ts > MAX_VALID_TIMESTAMP_MS:
        raise ValueError(
            f"Timestamp {ts} out of valid range (must be milliseconds between "
            f"{MIN_VALID_TIMESTAMP_MS} and {MAX_VALID_TIMESTAMP_MS})"
        )

    return ts

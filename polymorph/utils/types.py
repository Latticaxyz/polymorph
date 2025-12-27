from __future__ import annotations

JsonValue = bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]

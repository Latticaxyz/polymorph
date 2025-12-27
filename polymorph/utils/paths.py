from pathlib import Path


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    if len(stem) >= 4 and stem[-4] == "_" and stem[-3:].isdigit():
        stem = stem[:-4]

    counter = 1
    while counter <= 999:
        new_path = parent / f"{stem}_{counter:03d}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1
    raise ValueError(f"Too many versions of {path}")

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: str | Path = ROOT / '.env') -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_value(value: Any) -> Any:
    """Resolve ${ENV_VAR} placeholders while preserving ints and normal strings."""
    load_dotenv()
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def repl(match):
        key = match.group(1)
        return os.environ.get(key, match.group(0))

    resolved = pattern.sub(repl, value)
    if resolved.isdigit():
        return int(resolved)
    return resolved


def safe_source_display(source: Any) -> str:
    text = str(source)
    if '@' in text and '://' in text:
        return text.split('://', 1)[0] + '://***@' + text.split('@', 1)[1]
    return text

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def json_safe(obj):
    """Convert numpy scalars/arrays and Path objects into JSON-safe values."""
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


class EventLogger:
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: Dict[str, Any]) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(json_safe(event), ensure_ascii=False) + '\n')

"""A small in-memory cache with a time limit and a size limit."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Hashable


class TTLCache:
    def __init__(self, max_entries: int, ttl: float) -> None:
        self.max_entries = max_entries
        self.ttl = ttl
        self._items: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()

    def get(self, key: Hashable) -> Any | None:
        item = self._items.get(key)
        if item is None:
            return None
        stored_at, value = item
        if time.monotonic() - stored_at >= self.ttl:
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: Hashable, value: Any) -> None:
        self._items[key] = (time.monotonic(), value)
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)

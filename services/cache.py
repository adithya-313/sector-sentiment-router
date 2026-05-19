"""
Semantic deduplication and idempotency cache.

Idempotency means that processing the same input twice produces the same
result without side-effects.  We achieve this by hashing (text + tenant_id)
into a composite key and storing the extracted result.  If the same review
is submitted again, we return the cached result instead of calling the LLM.

This is critical for:
  - Avoiding duplicate LLM costs.
  - Ensuring consistent results on retries.
  - Protecting downstream systems from duplicate writes.
"""

import hashlib
import json
from typing import Any, Optional


class LocalSemanticCache:
    """In-memory cache that uses SHA-256 hashing for idempotent lookups.

    The cache is a plain dict — suitable for single-process development.
    For production, swap this for Redis or DuckDB.
    """

    def __init__(self) -> None:
        # Underlying storage: mapping from hash key -> JSON value.
        self._store: dict[str, str] = {}

    def _build_key(self, text: str, tenant_id: str) -> str:
        """Create a deterministic hash from the review text and tenant ID.

        Why hash instead of using the raw text as a key?
          - Reviews can be very long (wastes memory as a dict key).
          - A fixed-length hex digest is faster to compare.
          - Hashing normalises tiny differences (trailing whitespace etc.).
        """
        # Concatenate the two inputs with a separator that is unlikely
        # to appear naturally in either piece of data.
        composite = f"{text}|||{tenant_id}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    def get(self, text: str, tenant_id: str) -> Optional[Any]:
        """Retrieve a cached result for (text, tenant_id).

        Returns the deserialised Python object, or None on a cache miss.
        """
        key = self._build_key(text, tenant_id)
        raw = self._store.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, text: str, tenant_id: str, value: Any) -> None:
        """Store a result for (text, tenant_id) so future calls are idempotent.

        The value is serialised to JSON so the cache only stores plain strings.
        """
        key = self._build_key(text, tenant_id)
        self._store[key] = json.dumps(value, default=str)

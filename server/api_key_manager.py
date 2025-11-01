"""API key resolution with allowlist, platform lookup, and in-memory caching."""

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from fastapi import HTTPException

import httpx


ApiKeySource = Literal["allowlist", "platform"]


class APIKeyIdentity(TypedDict, total=False):
    """Normalized identity returned for a validated API key."""

    kind: Literal["organization"]
    id: str
    name: str
    domain: str | None


@dataclass(frozen=True)
class ResolvedAPIKey:
    """Resolved API key details returned to request handlers."""

    api_key_hash: str
    api_key_hash_short: str
    identity: APIKeyIdentity
    source: ApiKeySource


class APIKeyCache:
    """Small in-memory cache with TTL-based eviction."""

    def __init__(self, max_size: int, ttl_seconds: int) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, APIKeyIdentity]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str, now: float) -> APIKeyIdentity | None:
        async with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            expires_at, identity = entry
            if expires_at <= now:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return identity

    async def set(self, key: str, identity: APIKeyIdentity, now: float) -> None:
        async with self._lock:
            if key in self._entries:
                self._entries.pop(key, None)
            while len(self._entries) >= self._max_size:
                self._entries.popitem(last=False)
            expires_at = now + self._ttl_seconds
            self._entries[key] = (expires_at, identity)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._entries.pop(key, None)


class APIKeyManager:
    """Handles API key validation with allowlist, cache, and platform lookup."""

    def __init__(
        self,
        *,
        allowlist: dict[str, Any],
        cache_max_size: int,
        valid_ttl_seconds: int,
        platform_base_url: str,
        platform_lookup_path: str,
        lookup_timeout_seconds: float,
    ) -> None:
        self._allowlist = allowlist
        self._cache = APIKeyCache(cache_max_size, valid_ttl_seconds)
        self._platform_base_url = platform_base_url
        self._platform_lookup_path = platform_lookup_path
        self._lookup_timeout_seconds = lookup_timeout_seconds
        self._client = httpx.AsyncClient()

    async def close(self) -> None:
        await self._client.aclose()

    def set_allowlist(self, allowlist: dict[str, Any]) -> None:
        self._allowlist = allowlist

    def add_allowlist_entry(self, key: str, value: Any) -> None:
        self._allowlist[key] = value

    def remove_allowlist_entry(self, key: str) -> None:
        self._allowlist.pop(key, None)

    async def resolve(self, api_key: str) -> ResolvedAPIKey:
        if not api_key:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        now = time.time()
        api_key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        api_key_hash_short = api_key_hash[:8]

        cached_identity = await self._cache.get(api_key_hash, now)
        if cached_identity:
            return ResolvedAPIKey(
                api_key_hash=api_key_hash,
                api_key_hash_short=api_key_hash_short,
                identity=cached_identity,
                source="platform",
            )

        allowlist_identity = self._lookup_allowlist(api_key)
        if allowlist_identity:
            return ResolvedAPIKey(
                api_key_hash=api_key_hash,
                api_key_hash_short=api_key_hash_short,
                identity=allowlist_identity,
                source="allowlist",
            )

        platform_identity = await self._lookup_platform(api_key, api_key_hash, now)
        await self._cache.set(api_key_hash, platform_identity, now)
        return ResolvedAPIKey(
            api_key_hash=api_key_hash,
            api_key_hash_short=api_key_hash_short,
            identity=platform_identity,
            source="platform",
        )

    def _lookup_allowlist(self, api_key: str) -> APIKeyIdentity | None:
        raw_entry = self._allowlist.get(api_key)
        if raw_entry is None:
            return None
        if isinstance(raw_entry, dict):
            identity: APIKeyIdentity = {
                "kind": "organization",
                "name": str(raw_entry.get("name")) if raw_entry.get("name") else str(raw_entry.get("id", "unknown")),
            }
            if raw_entry.get("id"):
                identity["id"] = str(raw_entry["id"])
            if "domain" in raw_entry:
                identity["domain"] = raw_entry.get("domain")
            return identity
        # fallback to treating the stored value as organization name/slug
        return {
            "kind": "organization",
            "name": str(raw_entry),
        }

    async def _lookup_platform(
        self, api_key: str, api_key_hash: str, now: float
    ) -> APIKeyIdentity:
        url = httpx.URL(self._platform_base_url).join(self._platform_lookup_path)
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            response = await self._client.get(url, headers=headers, timeout=self._lookup_timeout_seconds)
        except httpx.TimeoutException:
            raise HTTPException(status_code=503, detail="API key validation timed out") from None
        except httpx.HTTPError:
            raise HTTPException(status_code=503, detail="API key validation unavailable") from None

        if response.status_code == 200:
            data = response.json()
            identity = self._normalize_platform_identity(data)
            if identity is None:
                raise HTTPException(status_code=503, detail="API key validation unavailable")
            return identity

        if response.status_code in (401, 404):
            await self._cache.delete(api_key_hash)
            raise HTTPException(status_code=401, detail="Invalid API Key")

        if 500 <= response.status_code < 600:
            raise HTTPException(status_code=503, detail="API key validation unavailable")

        raise HTTPException(status_code=401, detail="Invalid API Key")

    @staticmethod
    def _normalize_platform_identity(data: Any) -> APIKeyIdentity | None:
        if not isinstance(data, dict):
            return None
        kind = data.get("kind")
        if kind != "organization":
            return None
        identity: APIKeyIdentity = {"kind": "organization"}
        if data.get("id"):
            identity["id"] = str(data["id"])
        if data.get("name"):
            identity["name"] = str(data["name"])
        if "domain" in data:
            identity["domain"] = data.get("domain")
        return identity if identity.get("name") else None

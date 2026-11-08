"""Bounded, release-local cache of finished schedule responses."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from time import monotonic

from starlette.responses import Response


class ScheduleResponseCache:
    """Reuse serialized data and calculations without extending live-data freshness.

    One instance belongs to one immutable serving process. Thirty-second expiry
    matches observation refresh; no response survives a release/process change.
    """

    def __init__(self, *, ttl=30.0, max_bytes=16 * 1024 * 1024, clock=monotonic):
        self.ttl = ttl
        self.max_bytes = max_bytes
        self.clock = clock
        self.entries = OrderedDict()
        self.size = 0
        self.lock = asyncio.Lock()

    async def respond(self, key, build):
        # Coalesce simultaneous misses so expensive calculations run only once.
        async with self.lock:
            now = self.clock()
            for old_key, (expires, body, _) in list(self.entries.items()):
                if expires <= now:
                    self.size -= len(body)
                    del self.entries[old_key]
            if key in self.entries:
                self.entries.move_to_end(key)
                _, body, headers = self.entries[key]
                return Response(body, headers=headers)

            response = await build()
            if response.status_code != 200:
                return response
            if hasattr(response, 'body_iterator'):
                body = b''.join([chunk async for chunk in response.body_iterator])
            else:
                body = response.body
            headers = dict(response.headers)
            if len(body) <= self.max_bytes:
                while self.entries and self.size + len(body) > self.max_bytes:
                    _, (_, old_body, _) = self.entries.popitem(last=False)
                    self.size -= len(old_body)
                self.entries[key] = (self.clock() + self.ttl, body, headers)
                self.size += len(body)
            return Response(body, headers=headers, background=response.background)

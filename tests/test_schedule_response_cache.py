"""Cache behavior checks use a fake clock, never wall-clock measurements."""
import asyncio

from starlette.responses import JSONResponse, StreamingResponse

from nba_lineup_model.web_api.schedule_response_cache import ScheduleResponseCache


def test_dates_expiry_release_isolation_and_headers():
    async def check():
        now = [0]
        cache = ScheduleResponseCache(clock=lambda: now[0])
        calls = []

        async def build():
            calls.append(True)
            return JSONResponse({'calculation': len(calls)}, headers={
                'X-Gestalt-Release': 'release-a', 'Cache-Control': 'no-store'})

        first = await cache.respond(('release-a', 'date-a'), build)
        await cache.respond(('release-a', 'date-b'), build)
        repeated = await cache.respond(('release-a', 'date-a'), build)
        assert repeated.body == first.body
        assert repeated.headers == first.headers
        assert len(calls) == 2
        now[0] = 30
        refreshed = await cache.respond(('release-a', 'date-a'), build)
        assert refreshed.body != first.body
        assert len(calls) == 3
        await cache.respond(('release-b', 'date-a'), build)
        assert len(calls) == 4
        # A restarted process has no inherited cache.
        await ScheduleResponseCache().respond(('release-a', 'date-a'), build)
        assert len(calls) == 5
    asyncio.run(check())


def test_errors_are_not_cached_and_bytes_are_bounded():
    async def check():
        cache = ScheduleResponseCache(max_bytes=20)
        calls = []

        async def error():
            calls.append(True)
            return JSONResponse({'error': 'unavailable'}, status_code=503)

        assert (await cache.respond('error', error)).status_code == 503
        await cache.respond('error', error)
        assert len(calls) == 2
        assert not cache.entries

        async def small():
            return JSONResponse('1234567890')

        await cache.respond('a', small)
        await cache.respond('b', small)
        assert list(cache.entries) == ['b']
        assert cache.size <= 20

        async def large():
            return JSONResponse('x' * 100)

        assert len((await cache.respond('large', large)).body) > 20
        assert 'large' not in cache.entries
    asyncio.run(check())


def test_simultaneous_requests_reuse_finished_stream():
    async def check():
        cache = ScheduleResponseCache()
        calls = []

        async def build():
            calls.append(True)
            async def chunks():
                yield b'{"calculation":'
                yield b'42}'
            return StreamingResponse(chunks(), media_type='application/json')

        responses = await asyncio.gather(*(cache.respond('date', build) for _ in range(3)))
        assert len(calls) == 1
        assert all(response.body == b'{"calculation":42}' for response in responses)
    asyncio.run(check())

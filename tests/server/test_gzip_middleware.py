from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Request

from omnigent.server.routes._gzip_route import StateAwareGZipMiddleware, skip_gzip


@pytest.mark.asyncio
async def test_large_api_json_is_gzipped(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
    assert "accept-encoding" in response.headers.get("vary", "").lower()


@pytest.mark.asyncio
async def test_range_request_is_not_gzipped(client: httpx.AsyncClient) -> None:
    response = await client.get(
        "/openapi.json",
        headers={"Accept-Encoding": "gzip", "Range": "bytes=0-1023"},
    )

    assert response.status_code == 200
    assert "content-encoding" not in response.headers


@pytest.mark.asyncio
async def test_handler_can_skip_application_gzip() -> None:
    app = FastAPI()
    app.add_middleware(StateAwareGZipMiddleware, minimum_size=500)

    @app.get("/_test/gzip-skip")
    async def gzip_skip(request: Request) -> dict[str, str]:
        skip_gzip(request)
        return {"data": "x" * 2048}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/_test/gzip-skip", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert "content-encoding" not in response.headers

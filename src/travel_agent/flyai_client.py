from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .flyai_errors import (
    FlyAICommandNotFoundError,
    FlyAIInvalidJsonError,
    FlyAIResponseError,
    FlyAITimeoutError,
    FlyAIUnavailableError,
)
from .providers import AttractionSearchRequest, HotelSearchRequest, TransportMode, TransportSearchRequest

Runner = Callable[..., subprocess.CompletedProcess[str]]


class FlyAIClient:
    def __init__(self, command: str | Path | None = None, *, timeout_seconds: float | None = None, runner: Runner | None = None):
        self.command = str(command or os.getenv("TRAVEL_AGENT_FLYAI_COMMAND", "flyai"))
        self.timeout_seconds = float(timeout_seconds if timeout_seconds is not None else os.getenv("TRAVEL_AGENT_FLYAI_TIMEOUT_SECONDS", "30"))
        self._runner = runner

    def build_hotel_command(self, request: HotelSearchRequest, *, key_words: str | None = None, poi_name: str | None = None, hotel_types: str | None = None, sort: str | None = None, hotel_stars: str | None = None, hotel_bed_types: str | None = None, max_price: float | None = None) -> list[str]:
        command = [self.command, "search-hotel", "--dest-name", request.city]
        values = (("--key-words", key_words), ("--poi-name", poi_name), ("--hotel-types", hotel_types), ("--sort", sort), ("--check-in-date", request.check_in.isoformat()), ("--check-out-date", request.check_out.isoformat()), ("--hotel-stars", hotel_stars), ("--hotel-bed-types", hotel_bed_types), ("--max-price", str(max_price) if max_price is not None else None))
        for flag, value in values:
            if value not in (None, ""):
                command.extend([flag, value])
        return command

    async def search_hotel(self, request: HotelSearchRequest, **filters: Any) -> dict[str, Any]:
        return await self._run_json(self.build_hotel_command(request, **filters))

    def _build_transport_command(self, subcommand: str, request: TransportSearchRequest, **filters: Any) -> list[str]:
        command = [self.command, subcommand, "--origin", request.origin]
        values = (
            ("--destination", request.destination),
            ("--dep-date", request.departure_date.isoformat()),
            ("--dep-date-start", filters.get("departure_date_start")),
            ("--dep-date-end", filters.get("departure_date_end")),
            ("--back-date", filters.get("return_date")),
            ("--journey-type", filters.get("journey_type")),
            ("--seat-class-name", filters.get("seat_class_name")),
            ("--transport-no", filters.get("transport_no")),
            ("--transfer-city", filters.get("transfer_city")),
            ("--dep-hour-start", filters.get("departure_hour_start")),
            ("--dep-hour-end", filters.get("departure_hour_end")),
            ("--arr-hour-start", filters.get("arrival_hour_start")),
            ("--arr-hour-end", filters.get("arrival_hour_end")),
            ("--total-duration-hour", filters.get("total_duration_hour")),
            ("--max-price", filters.get("max_price")),
            ("--sort-type", filters.get("sort_type")),
        )
        for flag, value in values:
            if value not in (None, ""):
                command.extend([flag, str(value)])
        return command

    def build_train_command(self, request: TransportSearchRequest, **filters: Any) -> list[str]:
        return self._build_transport_command("search-train", request, **filters)

    def build_flight_command(self, request: TransportSearchRequest, **filters: Any) -> list[str]:
        return self._build_transport_command("search-flight", request, **filters)

    async def search_train(self, request: TransportSearchRequest, **filters: Any) -> dict[str, Any]:
        return await self._run_json(self.build_train_command(request, **filters))

    async def search_flight(self, request: TransportSearchRequest, **filters: Any) -> dict[str, Any]:
        return await self._run_json(self.build_flight_command(request, **filters))

    def build_poi_command(
        self,
        request: AttractionSearchRequest,
        *,
        poi_level: int | None = None,
        keyword: str | None = None,
        category: str | None = None,
    ) -> list[str]:
        command = [self.command, "search-poi", "--city-name", request.city]
        values = (
            ("--poi-level", poi_level),
            ("--keyword", keyword),
            ("--category", category),
        )
        for flag, value in values:
            if value not in (None, ""):
                command.extend([flag, str(value)])
        return command

    async def search_poi(self, request: AttractionSearchRequest, **filters: Any) -> dict[str, Any]:
        return await self._run_json(self.build_poi_command(request, **filters))

    async def _run_json(self, command: list[str]) -> dict[str, Any]:
        try:
            if self._runner is not None:
                completed = self._runner(command, capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
            else:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), self.timeout_seconds)
                except asyncio.TimeoutError as error:
                    process.kill()
                    await process.wait()
                    raise FlyAITimeoutError("FlyAI 查询超时") from error
                completed = subprocess.CompletedProcess(command, process.returncode, stdout.decode(), stderr.decode())
        except FileNotFoundError as error:
            raise FlyAICommandNotFoundError(f"找不到 FlyAI 命令: {self.command}") from error
        except subprocess.TimeoutExpired as error:
            raise FlyAITimeoutError("FlyAI 查询超时") from error
        except OSError as error:
            raise FlyAIUnavailableError(f"FlyAI 命令不可用: {error}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise FlyAIUnavailableError(f"FlyAI 查询失败（exit={completed.returncode}）: {detail[:500]}")
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise FlyAIInvalidJsonError("FlyAI 返回内容不是有效 JSON") from error
        if not isinstance(payload, dict):
            raise FlyAIInvalidJsonError("FlyAI 返回 JSON 顶层必须是对象")
        if payload.get("status") != 0:
            raise FlyAIResponseError(f"FlyAI 返回失败状态: {payload.get('message', 'unknown error')}")
        return payload

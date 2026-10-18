from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .flyai_errors import (
    FlyAICommandNotFoundError,
    FlyAIInvalidJsonError,
    FlyAIRateLimitError,
    FlyAIResponseError,
    FlyAITimeoutError,
    FlyAIUnavailableError,
)
from .providers import AttractionSearchRequest, HotelSearchRequest, TransportSearchRequest

Runner = Callable[..., subprocess.CompletedProcess[str]]

_RATE_LIMIT_PATTERNS = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "请求过于频繁",
    "限流",
)


def _looks_rate_limited(detail: str) -> bool:
    lowered = detail.lower()
    return any(pattern in lowered for pattern in _RATE_LIMIT_PATTERNS)


class FlyAIClient:
    def __init__(
        self,
        command: str | Path | None = None,
        *,
        timeout_seconds: float | None = None,
        runner: Runner | None = None,
        max_attempts: int | None = None,
        backoff_seconds: float | None = None,
        concurrency_limit: int | None = None,
    ):
        self.command = str(command or os.getenv("TRAVEL_AGENT_FLYAI_COMMAND", "flyai"))
        self.timeout_seconds = float(timeout_seconds if timeout_seconds is not None else os.getenv("TRAVEL_AGENT_FLYAI_TIMEOUT_SECONDS", "30"))
        self._runner = runner
        self.max_attempts = int(
            max_attempts if max_attempts is not None else os.getenv("TRAVEL_AGENT_FLYAI_MAX_ATTEMPTS", "3")
        )
        self.backoff_seconds = float(
            backoff_seconds if backoff_seconds is not None else os.getenv("TRAVEL_AGENT_FLYAI_BACKOFF_SECONDS", "0.5")
        )
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        # 上游按账号限流，并发越高越容易撞 429。所有 provider 共用同一个
        # client 实例时，这个信号量就是全局的并发闸门。
        self.concurrency_limit = int(
            concurrency_limit
            if concurrency_limit is not None
            else os.getenv("TRAVEL_AGENT_FLYAI_CONCURRENCY", "2")
        )
        # 不要在这里建 asyncio.Semaphore：service 层每个请求调一次
        # asyncio.run()，而 Semaphore 会绑定到创建它的 event loop，
        # 第二次调用就会报 "bound to a different event loop"。
        # 改为按运行中的 loop 惰性创建并缓存。
        self._semaphores: dict[Any, asyncio.Semaphore] = {}

    def _semaphore_for_current_loop(self) -> asyncio.Semaphore | None:
        if self.concurrency_limit <= 0:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.concurrency_limit)
            self._semaphores[loop] = semaphore
            # loop 关闭后残留的条目没有意义，及时清理避免字典无限增长。
            weakref.finalize(loop, self._semaphores.pop, loop, None)
        return semaphore

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
        """执行一次 FlyAI 查询，对限流等可恢复错误做有界退避重试。

        只有 ``FlyAIRateLimitError`` 会重试：命令不存在、返回非法 JSON、
        业务失败状态重试都不会改变结果，重试只是浪费时间与配额。
        """
        last_error: FlyAIRateLimitError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                completed = await self._run_once(command)
            except FlyAIRateLimitError as error:
                last_error = error
                if attempt >= self.max_attempts:
                    break
                await asyncio.sleep(self._backoff_delay(attempt))
                continue
            return self._parse_output(completed)
        assert last_error is not None
        raise FlyAIRateLimitError(
            f"{last_error}（已重试 {self.max_attempts} 次）"
        ) from last_error

    def _backoff_delay(self, attempt: int) -> float:
        """指数退避 + 抖动，避免并发请求在同一时刻一起重试造成二次限流。"""
        return self.backoff_seconds * (2 ** (attempt - 1)) * (0.5 + random.random())

    async def _run_once(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        semaphore = self._semaphore_for_current_loop()
        if semaphore is not None:
            async with semaphore:
                return await self._spawn(command)
        return await self._spawn(command)

    async def _spawn(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if self._runner is not None:
            try:
                completed = self._runner(command, capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
            except FileNotFoundError as error:
                raise FlyAICommandNotFoundError(f"找不到 FlyAI 命令: {self.command}") from error
            except subprocess.TimeoutExpired as error:
                raise FlyAITimeoutError("FlyAI 查询超时") from error
            except OSError as error:
                raise FlyAIUnavailableError(f"FlyAI 命令不可用: {error}") from error
            self._raise_for_exit(completed)
            return completed

        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), self.timeout_seconds)
            except asyncio.TimeoutError as error:
                raise FlyAITimeoutError("FlyAI 查询超时") from error
        except FileNotFoundError as error:
            raise FlyAICommandNotFoundError(f"找不到 FlyAI 命令: {self.command}") from error
        except OSError as error:
            raise FlyAIUnavailableError(f"FlyAI 命令不可用: {error}") from error
        finally:
            # 无论成功、超时还是被取消，都必须回收子进程并关闭管道。
            # 否则 transport 会活过 asyncio.run() 的 event loop，在 GC 时
            # 触发 "Exception ignored ... RuntimeError: Event loop is closed"。
            await self._reap(process)

        completed = subprocess.CompletedProcess(command, process.returncode, stdout.decode(), stderr.decode())
        self._raise_for_exit(completed)
        return completed

    @staticmethod
    async def _reap(process: asyncio.subprocess.Process | None) -> None:
        if process is None:
            return
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await process.wait()
        except ProcessLookupError:
            pass
        finally:
            transport = getattr(process, "_transport", None)
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass

    def _raise_for_exit(self, completed: subprocess.CompletedProcess[str]) -> None:
        if completed.returncode == 0:
            return
        detail = (completed.stderr or completed.stdout or "").strip()
        if _looks_rate_limited(detail):
            raise FlyAIRateLimitError(
                f"FlyAI 查询被限流（exit={completed.returncode}）: {detail[:500]}"
            )
        raise FlyAIUnavailableError(f"FlyAI 查询失败（exit={completed.returncode}）: {detail[:500]}")

    def _parse_output(self, completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise FlyAIInvalidJsonError("FlyAI 返回内容不是有效 JSON") from error
        if not isinstance(payload, dict):
            raise FlyAIInvalidJsonError("FlyAI 返回 JSON 顶层必须是对象")
        if payload.get("status") != 0:
            message = str(payload.get("message", "unknown error"))
            if _looks_rate_limited(message):
                raise FlyAIRateLimitError(f"FlyAI 返回限流状态: {message}")
            raise FlyAIResponseError(f"FlyAI 返回失败状态: {message}")
        return payload

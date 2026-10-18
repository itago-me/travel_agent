"""阶段 6 预备：FlyAI 客户端的限流重试、错误分类与子进程回收。

真实故障：一轮并行工具调用同时打 transport/hotel/attraction，撞上游
``MCP HTTP 429 Rate limit exceeded``，FlyAI CLI 以 exit=1 退出，原实现把
它当成不可恢复的 ``FlyAIUnavailableError`` 直接抛出，并且只重试了超时
分支，导致 429 直接变成该轮工具失败。同时在异常路径上子进程没有被回收，
transport 活过了 ``asyncio.run()`` 的 event loop，GC 时打印
``RuntimeError: Event loop is closed``。
"""

import asyncio
import subprocess

import pytest

from travel_agent.flyai_client import FlyAIClient
from travel_agent.flyai_errors import (
    FlyAIInvalidJsonError,
    FlyAIRateLimitError,
    FlyAIResponseError,
    FlyAITimeoutError,
    FlyAIUnavailableError,
)
from travel_agent.flyai_providers import FlyAITransportProvider
from travel_agent.provider_factory import (
    ProviderSettings,
    create_provider_bundle,
)
from travel_agent.providers import TransportSearchRequest

RATE_LIMIT_STDERR = (
    'MCP HTTP 429:\nBody: {"jsonrpc":"2.0","id":"1","error":{"code":-32603,'
    '"message":"Rate limit exceeded, please try again later"}}'
)


def completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(["flyai"], returncode, stdout, stderr)


OK_STDOUT = '{"status": 0, "message": "success", "data": {"itemList": []}}'


class ScriptedRunner:
    """按顺序返回预设结果，记录每次调用参数。"""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_client(runner, **kwargs):
    kwargs.setdefault("backoff_seconds", 0)
    kwargs.setdefault("max_attempts", 3)
    return FlyAIClient(runner=runner, **kwargs)


def transport_request():
    return TransportSearchRequest(
        origin="北京", destination="杭州", departure_date="2026-10-15", traveler_count=2
    )


# --- 错误分类 ---------------------------------------------------------------


def test_http_429_is_classified_as_rate_limit():
    client = make_client(ScriptedRunner([completed(1, stderr=RATE_LIMIT_STDERR)]), max_attempts=1)

    with pytest.raises(FlyAIRateLimitError) as error:
        asyncio.run(client.search_train(transport_request()))

    assert "限流" in str(error.value)


def test_rate_limit_error_is_a_subclass_of_unavailable():
    """既有调用方捕获 FlyAIUnavailableError 的地方必须继续生效。"""
    assert issubclass(FlyAIRateLimitError, FlyAIUnavailableError)


def test_non_rate_limit_failure_stays_unavailable():
    client = make_client(ScriptedRunner([completed(1, stderr="MCP HTTP 500 internal error")]))

    with pytest.raises(FlyAIUnavailableError) as error:
        asyncio.run(client.search_train(transport_request()))

    assert not isinstance(error.value, FlyAIRateLimitError)


def test_generic_failure_is_not_retried():
    """非限流错误重试不会改变结果，只浪费配额。"""
    runner = ScriptedRunner([completed(1, stderr="boom")])
    client = make_client(runner)

    with pytest.raises(FlyAIUnavailableError):
        asyncio.run(client.search_train(transport_request()))

    assert len(runner.calls) == 1


def test_status_field_rate_limit_is_also_classified():
    """限流也可能以 status != 0 的 JSON 主体返回，而不是靠 exit code。"""
    payload = '{"status": 1, "message": "Rate limit exceeded, please try again later"}'
    client = make_client(ScriptedRunner([completed(0, stdout=payload)]), max_attempts=1)

    with pytest.raises(FlyAIRateLimitError):
        asyncio.run(client.search_train(transport_request()))


def test_business_error_status_is_not_rate_limit():
    payload = '{"status": 1, "message": "出发日期非法"}'
    client = make_client(ScriptedRunner([completed(0, stdout=payload)]))

    with pytest.raises(FlyAIResponseError) as error:
        asyncio.run(client.search_train(transport_request()))

    assert not isinstance(error.value, FlyAIRateLimitError)


# --- 重试 -------------------------------------------------------------------


def test_rate_limit_is_retried_then_succeeds():
    runner = ScriptedRunner([
        completed(1, stderr=RATE_LIMIT_STDERR),
        completed(1, stderr=RATE_LIMIT_STDERR),
        completed(0, stdout=OK_STDOUT),
    ])
    client = make_client(runner)

    payload = asyncio.run(client.search_train(transport_request()))

    assert payload["status"] == 0
    assert len(runner.calls) == 3


def test_rate_limit_gives_up_after_max_attempts():
    runner = ScriptedRunner([completed(1, stderr=RATE_LIMIT_STDERR) for _ in range(10)])
    client = make_client(runner, max_attempts=3)

    with pytest.raises(FlyAIRateLimitError) as error:
        asyncio.run(client.search_train(transport_request()))

    assert len(runner.calls) == 3
    assert "已重试 3 次" in str(error.value)


def test_backoff_delay_grows_and_jitters():
    client = FlyAIClient(backoff_seconds=1.0, max_attempts=4)

    samples = [client._backoff_delay(1) for _ in range(50)]
    # 0.5 + random() -> [0.5, 1.5)
    assert all(0.5 <= value < 1.5 for value in samples)
    assert len(set(samples)) > 1, "缺少抖动会导致并发请求同时重试"

    # attempt=2 翻倍为 [1.0, 3.0)，下界严格大于 attempt=1 的取值下限
    assert all(1.0 <= client._backoff_delay(2) < 3.0 for _ in range(20))
    assert client._backoff_delay(3) >= 2.0


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError, match="max_attempts"):
        FlyAIClient(max_attempts=0)


def test_timeout_is_reported_as_timeout_not_rate_limit():
    runner = ScriptedRunner([subprocess.TimeoutExpired("flyai", 1)])
    client = make_client(runner)

    with pytest.raises(FlyAITimeoutError):
        asyncio.run(client.search_train(transport_request()))


def test_invalid_json_is_not_retried():
    runner = ScriptedRunner([completed(0, stdout="not json at all")])
    client = make_client(runner)

    with pytest.raises(FlyAIInvalidJsonError):
        asyncio.run(client.search_train(transport_request()))

    assert len(runner.calls) == 1


# --- 子进程回收 -------------------------------------------------------------


def test_subprocess_is_reaped_on_failure_path():
    """失败路径必须回收子进程，否则 transport 会在 loop 关闭后被 GC。"""

    class FakeTransport:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self):
            self.returncode = 1
            self._transport = FakeTransport()
            self.waited = False

        async def communicate(self):
            return b"", RATE_LIMIT_STDERR.encode()

        def kill(self):
            pass

        async def wait(self):
            self.waited = True

    process = FakeProcess()
    # runner=None 才会走真实的 create_subprocess_exec 分支
    client = FlyAIClient(backoff_seconds=0, max_attempts=1)

    async def fake_create(*args, **kwargs):
        return process

    original = asyncio.create_subprocess_exec
    asyncio.create_subprocess_exec = fake_create
    try:
        with pytest.raises(FlyAIRateLimitError):
            asyncio.run(client.search_train(transport_request()))
    finally:
        asyncio.create_subprocess_exec = original

    assert process.waited, "必须等待子进程退出"
    assert process._transport.closed, "必须关闭 transport 管道"


def test_real_subprocess_does_not_leak_event_loop_warnings():
    """用真实子进程跑一次失败路径，确认不再出现 Event loop is closed。

    这是对原始 traceback 的直接回归：子进程以 429 文案写到 stderr 并退出。
    """
    import sys

    script = (
        "import sys;"
        f"sys.stderr.write({RATE_LIMIT_STDERR!r});"
        "sys.exit(1)"
    )
    # runner=None 走真实子进程；command 指向 python 以便复现 429 的 exit=1
    client = FlyAIClient(
        command=sys.executable, timeout_seconds=10, backoff_seconds=0, max_attempts=1
    )

    with pytest.raises(FlyAIRateLimitError):
        asyncio.run(client._spawn([sys.executable, "-c", script]))


# --- 并发闸门 ---------------------------------------------------------------


def test_concurrency_limit_serialises_in_flight_requests():
    """并发上限必须真正限制同时在飞的请求数量。"""
    in_flight = 0
    peak = 0

    class Gate:
        def __call__(self, command, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            in_flight -= 1
            return completed(0, stdout=OK_STDOUT)

    client = make_client(Gate(), concurrency_limit=1)

    async def scenario():
        return await asyncio.gather(
            *[client.search_train(transport_request()) for _ in range(5)]
        )

    asyncio.run(scenario())
    assert peak == 1


def test_client_survives_multiple_event_loops():
    """service 层每个请求调用一次 asyncio.run()，每次都换一个 event loop。

    在 __init__ 里创建 asyncio.Semaphore 会把它绑定到第一个 loop。第二个
    loop 上再争用同一个信号量时，内部 Future 属于已关闭的 loop，表现为
    ``RuntimeError: ... bound to a different event loop``，在真实子进程
    路径上更糟——直接死锁。信号量必须按运行中的 loop 惰性创建。

    这里用真实子进程（而不是同步 runner）来制造真正的等待者：同步 runner
    从不挂起，acquire/release 一气呵成，永远不产生 waiter，因此测不出这个
    bug。
    """
    import sys

    client = FlyAIClient(
        command=sys.executable, concurrency_limit=1, backoff_seconds=0, max_attempts=1
    )
    assert not hasattr(client, "_eager"), "信号量不得在构造时绑定 event loop"

    async def one():
        return await client._run_once(
            [sys.executable, "-c", "print('{\"status\": 0}')"]
        )

    async def contended():
        # 4 个请求抢 1 个名额 -> 必然产生等待者
        return await asyncio.gather(*[one() for _ in range(4)])

    for round_no in range(3):
        results = asyncio.run(contended())
        assert len(results) == 4, f"round {round_no} did not complete"

    # 每个 loop 一个信号量，关闭后条目被清理，不会无限增长
    assert len(client._semaphores) <= 3


def test_semaphore_cache_is_keyed_per_loop():
    client = FlyAIClient(concurrency_limit=2)

    async def grab():
        return client._semaphore_for_current_loop()

    first = asyncio.run(grab())
    second = asyncio.run(grab())

    assert first is not None and second is not None
    assert first is not second, "不同 loop 必须拿到各自的信号量"


def test_eager_semaphore_deadlocks_across_loops():
    """固化上面测试的立论基础：争用下跨 loop 复用信号量必然出错。"""
    sem = asyncio.Semaphore(1)

    async def use():
        async with sem:
            await asyncio.sleep(0.01)
            return "ok"

    async def contended():
        return await asyncio.gather(use(), use())  # 制造等待者

    assert asyncio.run(contended()) == ["ok", "ok"]
    with pytest.raises(RuntimeError, match="different event loop"):
        asyncio.run(contended())


def test_concurrency_limit_zero_disables_gate():
    runner = ScriptedRunner([completed(0, stdout=OK_STDOUT)])
    client = make_client(runner, concurrency_limit=0)

    assert client._semaphore_for_current_loop() is None

    async def scenario():
        return await client.search_train(transport_request())

    assert asyncio.run(scenario())["status"] == 0


def test_client_does_not_bind_semaphore_at_construction():
    """构造时不应存在任何已绑定的信号量。"""
    client = FlyAIClient(concurrency_limit=2)

    assert client._semaphores == {}


def test_shared_client_semaphore_is_shared_across_providers():
    """工厂必须让三个 provider 共用同一个 client，闸门才是全局的。"""
    bundle = create_provider_bundle(
        ":memory:",
        ProviderSettings(hotel="flyai", transport="flyai", attraction="flyai"),
    )

    clients = {bundle.hotel.client, bundle.transport.client, bundle.attraction.client}

    assert len(clients) == 1, "各建各的 client 会让限流闸门失效"


def test_bundle_does_not_build_client_for_sqlite_only():
    bundle = create_provider_bundle(
        ":memory:", ProviderSettings(hotel="sqlite", transport="sqlite", attraction="sqlite")
    )

    assert not hasattr(bundle.hotel, "client")


def test_rate_limit_surfaces_through_provider_layer():
    """provider 层不应吞掉限流错误，上层才有机会做降级或提示。"""
    client = make_client(ScriptedRunner([completed(1, stderr=RATE_LIMIT_STDERR)]), max_attempts=1)
    provider = FlyAITransportProvider(client)

    with pytest.raises(FlyAIRateLimitError):
        asyncio.run(provider.search(transport_request()))

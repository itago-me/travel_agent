"""阶段 9 A2：仓储接口与实现的结构一致性。

``ConsultationService`` 现在依赖 ``RepositoryProtocol`` 而不是
``SQLiteRepository``。这一层只在类型检查期起作用——运行时没有任何检查会
发现「接口声明了 29 个方法，实现只写了 28 个」。

所以用测试来守住它：**Protocol 里的每个方法，实现里必须存在，且参数名顺序
一致**。将来往接口加方法却忘了实现（或反过来），这里会立刻失败。

这正是 A2 的价值边界：它不改变行为，只把「实现必须满足的契约」变成可执行
的断言。B3 写 ``PostgresRepository`` 时，同一组断言可以复用它。
"""

import inspect

import pytest

from travel_agent.repository import SQLiteRepository
from travel_agent.repository_protocol import RepositoryProtocol


def _public_methods(cls) -> set[str]:
    """类上的公开方法名（排除下划线开头的内部方法）。"""
    return {
        name
        for name, member in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def test_protocol_covers_every_public_repository_method():
    """实现里的公开方法不能多于接口声明——多的那个就是没人依赖的死代码，
    或者接口漏声明了。两种都要处理。"""
    extra = _public_methods(SQLiteRepository) - _public_methods(RepositoryProtocol)

    assert extra == set(), f"以下方法未在接口中声明：{sorted(extra)}"


def test_repository_implements_every_protocol_method():
    missing = _public_methods(RepositoryProtocol) - _public_methods(SQLiteRepository)

    assert missing == set(), f"接口要求但实现缺失：{sorted(missing)}"


@pytest.mark.parametrize("name", sorted(_public_methods(RepositoryProtocol)))
def test_parameter_names_match(name):
    """参数名必须一致：调用方全是按关键字传参的，改名即破坏调用。

    （``inspect.signature`` 的参数顺序也一并比较——顺序不同意味着
    位置参数调用会错位。）
    """
    declared = list(inspect.signature(getattr(RepositoryProtocol, name)).parameters)
    implemented = list(inspect.signature(getattr(SQLiteRepository, name)).parameters)

    assert declared == implemented


def test_protocol_declares_database_path_as_known_coupling():
    """``database_path`` 是 SQLite 特有的，刻意留在接口上。

    它被三处真实依赖：checkpoint 路径派生、``create_agent_app`` 装配、
    工具按路径自建仓储。B3 阶段必须先消除这个依赖，Postgres 实现才接得进来。
    这个断言的作用是——**别在重构时不小心把它删掉**，那会让 service 静默
    失去类型保护。
    """
    assert "database_path" in RepositoryProtocol.__annotations__

"""Alembic 运行环境。

## 与默认模板的三处差别

1. **连接串从环境变量读**（``TRAVEL_AGENT_DATABASE_URL``），不写在
   ``alembic.ini`` 里——密码进版本库是最常见的凭据泄露方式。
2. **``target_metadata`` 保持 ``None``**：本项目用裸 SQL，没有 ORM 模型也没有
   ``sa.Table`` 声明式结构，因此 ``--autogenerate`` 不可用，迁移一律手写。
   （autogenerate 比对的其实是 ``MetaData``；ORM 只是产生它的一种方式。）
3. **只跑 Postgres**：SQLite 仍走 ``repository._initialize()`` 的手写迁移，
   两套并行，互不影响。若连不上或没配 URL，直接报错而不是静默跳过——
   「以为跑了迁移但没跑」比「明确失败」危险得多。
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# .env 里配置的 TRAVEL_AGENT_DATABASE_URL 必须能被迁移看见。
# 早先这里没有加载，导致「.env 写了连接串、alembic upgrade 却说没有设置」——
# 而 application.py 也不加载，两边都读不到，排查起来很费时间。
# override=False：export 的真实环境变量优先。
load_dotenv(override=False)

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False 不能省：默认 True 会把进程里所有已经存在的
    # logger 置为 disabled——包括 travel_agent 自己的。后果是「在同一进程里跑过
    # 迁移之后，应用日志彻底消失」（实测：alembic 跑完再断言日志内容会拿到空）。
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# 裸 SQL 项目：没有声明式 schema 可比对，迁移必须手写。
target_metadata = None


def _database_url() -> str:
    """Read the migration target from the environment.

    Precedence: an explicitly configured ``sqlalchemy.url`` (set
    programmatically, e.g. by a test fixture that needs migrations confined to
    an isolated schema) wins over the environment. Named separately from the
    usual ``sqlalchemy.url`` so a stray value in ``alembic.ini`` can never point
    migrations at the wrong database.
    """
    explicit = (config.get_main_option("sqlalchemy.url") or "").strip()
    if explicit:
        return explicit

    url = os.getenv("TRAVEL_AGENT_DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "需要设置 TRAVEL_AGENT_DATABASE_URL 才能运行迁移，例如：\n"
            "  export TRAVEL_AGENT_DATABASE_URL="
            "'postgresql+psycopg://user:pass@127.0.0.1:5432/travel_agent'"
        )
    return url


def run_migrations_offline() -> None:
    """生成 SQL 而不连库（``alembic upgrade head --sql``）。

    用来在没装驱动、或不允许直连生产的环境里审查将执行的语句。
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """连上数据库执行迁移。"""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    # NullPool：迁移是一次性短命连接，用连接池没有意义，
    # 而且连接池会让「脚本结束后连接没释放」的问题变得隐蔽。
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # 默认事务行为：整个迁移在一个事务里，失败则全部回滚。
            transaction_per_migration=False,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

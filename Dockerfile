# syntax=docker/dockerfile:1
#
# travel-agent 应用镜像

FROM python:3.12-slim

# PYTHONUNBUFFERED：不加的话 stdout 是块缓冲，docker logs 看不到实时日志
# PIP_NO_CACHE_DIR：pip 缓存在镜像里没有复用价值，只会撑大体积
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


COPY pyproject.toml ./
RUN python -c 'import tomllib, pathlib; deps = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["dependencies"]; pathlib.Path("requirements.txt").write_text("\n".join(deps) + "\n")' \
  && pip install --upgrade pip \
  && pip install -r requirements.txt


COPY src ./src
RUN pip install --no-deps .

# 迁移脚本与配置：migrate 服务要在**同一个镜像**里执行 `alembic upgrade head`。
# 只装包不带这些，容器里就没有 alembic.ini 和 versions/，迁移无从谈起。
COPY alembic.ini ./
COPY migrations ./migrations


RUN mkdir -p /app/data \
  && useradd --create-home --uid 1000 appuser \
  && chown -R appuser:appuser /app
USER appuser

EXPOSE 8077

# 用镜像自带的 python 探活，不额外装 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8077/health', timeout=3).status == 200 else 1)"


CMD ["uvicorn", "travel_agent.api:app", "--host", "0.0.0.0", "--port", "8077", "--log-level", "warning"]

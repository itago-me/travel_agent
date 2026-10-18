# Travel Agent

一个面向人工旅行顾问的 **Copilot**，用于在学习 Agent 工程的同时构建接近真实业务的工作流。

顾问是业务责任人，Agent 是协作工具。Agent 可以理解需求、调用工具、查询产品、生成方案，但**不能绕过审批自行完成高风险操作**——预订类工具在执行前必须经过人工中断。

---

## 为什么做这个项目

大多数 Agent 示例停在"能对话、能调工具"。真实业务里更难的部分是：

- 模型连续对话后**忘记**自己在处理哪一版方案；
- 高风险操作需要人**拦一下**，而且拦完之后任务要能继续；
- 供应商限流、子进程泄漏、脏数据把整条链路**卡死**；
- 上下文无限增长，什么时候该裁剪、什么时候该摘要。

本项目把这些逐一实现并留下可复现的验证，而不是只描述"应该怎么做"。

---

## 当前能力

### 1. 真实 Agent 对话闭环

使用 LangChain 官方 `create_agent`，由**模型自主决定**是否调用工具、调用哪个、调用几次——不是 CLI 写死的脚本流程。

```text
顾问自然语言输入
    -> Agent 判断下一步
    -> 调用真实 StructuredTool
    -> 工具校验参数、调用数据适配器
    -> 返回结构化观察
    -> Agent 继续规划或回答
```

### 2. 五个工具，两类风险

| 工具 | 类型 | 权限 |
|---|---|---|
| `search_transport_options` | 查询 | 只读 |
| `search_hotel_options` | 查询 | 只读 |
| `search_attractions` | 查询 | 只读 |
| `create_mock_booking` | 写 | **SENSITIVE** |
| `cancel_mock_booking` | 写 | **SENSITIVE** |

敏感工具通过 metadata 标记 `permission=SENSITIVE`、`requires_human_approval=true`，并由 `HumanInTheLoopMiddleware` 在执行前强制中断。

### 3. 双数据源，逐类切换

三个查询工具内部走 Provider 工厂，同一套 Schema 下可独立切换数据源：

```text
search_hotel_options     -> SQLiteHotelProvider     / FlyAIHotelProvider
search_transport_options -> SQLiteTransportProvider / FlyAITransportProvider
search_attractions       -> SQLiteAttractionProvider/ FlyAIAttractionProvider
```

`sqlite` 用于离线测试与本地开发；`flyai` 通过外部 `flyai` 子命令接入真实旅行数据（含限流重试与错误分级）。

### 4. 方案版本从真实工具结果派生

`chat` 结束时读取本轮 `search_*` 工具的 `ToolMessage`，把供应商**真正返回**的选项落成一个版本。因此版本不可能描述供应商从未提供过的选项。

预算按真实 `total_price` / `price_per_night` / `ticket_price` 重算，门票乘以 `traveler_count`。**信息不足时宁可不给预算字段，也不编造总额**——因为版本会被用于核对审批金额。

### 5. HITL：暂停、决定、恢复

```text
Agent 生成敏感 tool call
    -> HumanInTheLoopMiddleware.after_model
    -> LangGraph interrupt + checkpoint 落盘
    -> 审批单写入 approval_requests（PENDING）
    -> 咨询状态 WAITING_FOR_APPROVAL
    -> 顾问 approve / edit / reject
    -> Command(resume={"decisions": [...]}) 恢复原线程
    -> 批准建单 / 拒绝后模型继续规划
```

三个设计要点：

- **拒绝不会中断任务**。拒绝原因作为错误 `ToolMessage` 回传给模型，它可以解释或换方案重试。
- **`--edit-json` 是补丁语义**。官方中间件会用 `edited_action.args` 整体替换参数，只提交改动的字段会丢掉 `consultation_id` 等必需项。因此只允许覆盖已存在的键，合并结果再过一遍工具 Schema 校验。
- **一次 interrupt 可能含多个动作**。中间件要求决定数量与动作数量严格一致，所以服务层在仍有 PENDING 动作时不恢复，全部决定后按 `action_index` 顺序一次性恢复。

### 6. 记忆与上下文治理

三层 Middleware 分工明确：

| Middleware | 职责 |
|---|---|
| `TravelMemoryMiddleware` | 客户偏好、当前需求、结构化摘要、自然语言摘要、方案版本注入 |
| `ContextWindowMiddleware` | 消息窗口裁剪（默认 12 条 / 约 4000 估算 token） |
| `HumanInTheLoopMiddleware` | 敏感工具执行前中断 |

**当前需求优先于长期偏好**：本轮明确设置了 `budget`，就把它从历史偏好默认值中移除。代码负责确定性字段冲突，模型负责"赶时间""重视舒适度"这类语义判断。

窗口裁剪不丢数据——业务 SQLite 和 checkpoint 都保留完整原始消息；被移出窗口的旧消息由 `before_model` 增量摘要成 `structured_summary`（代码生成）和 `conversation_summary`（模型生成）。摘要模型调用失败时保留旧摘要，不阻断主流程。

### 7. 双轨持久化

```text
业务 SQLite (data/travel_agent.db)           顾问可查、可审计的业务事实
LangGraph SQLite checkpoint                  图执行位置、消息、工具结果、中断
```

两者通过 `consultation_id` / `thread_id` / `interrupt_id` 关联，但**不混为一张表**。判断"图是否真的停在中途"只能查 checkpoint；业务库看不到这个信息。

---

## 快速开始

### 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
cp .env.example .env      # 填入你自己的模型配置
```

> 真实模型调用前需要 `unset all_proxy ALL_PROXY`，否则 httpx 可能因缺少 `socksio` 而失败。

### 完整演示

```bash
export PYTHONPATH=src

DB=data/travel_agent.db
CK=data/travel_agent_checkpoints.sqlite
ta() { python -m travel_agent.cli --database $DB --checkpoint-database $CK "$@"; }

# 1. 建单（同时提取日期/人数/预算等结构化需求）
ta start --consultant-id c1 --customer-id u1 --customer-name 张先生 \
  --message "我想去杭州，2人，2026-10-01到2026-10-03，从北京出发，预算8000元"

# 2. 让 Agent 查询并生成方案（版本 1 自动落库）
ta chat <CID> --env-file .env --message "请查询北京到杭州的交通和杭州2晚酒店"

# 3. 选定一个版本（预订的前提）
ta itineraries <CID>
ta select-itinerary <CID> <VID>

# 4. 让 Agent 发起模拟预订 -> 图在此暂停
ta chat <CID> --env-file .env --message "已选定版本，请创建模拟预订"

# 5. 审批（approve 会自动恢复线程）
ta pending <CID>
ta approve <AID> --decided-by consultant-1 --reason "同意"

# 6. 查看结果
ta status <CID>
```

第 4 步返回 `status: WAITING_FOR_APPROVAL`，此时 `mock_bookings` 仍为 0 行——**这是 HITL 生效的证明**。第 5 步之后才写入订单。

---

## CLI 一览

| 命令 | 作用 |
|---|---|
| `start` | 创建咨询并提取结构化需求 |
| `status` | 查看咨询、消息、待审批列表 |
| `chat` | 与 Agent 对话（唯一会跑模型和工具的命令） |
| `itineraries` | 列出方案版本 |
| `select-itinerary` | 选定版本（预订前提） |
| `pending` | 查看未决审批 |
| `approve` | 批准，支持 `--edit-json` 改参、`--no-resume` 延迟恢复 |
| `reject` | 拒绝，必须给出 `--reason` |
| `resume-approval` | 仅恢复（用于 `--no-resume` 之后） |
| `sensitive-tools` | 查看敏感工具的 Schema 与权限标记 |
| `preferences` / `set-preference` / `delete-preference` | 客户长期偏好维护 |

所有命令输出 JSON，不打印 Python 堆栈；未知咨询、未知审批单、重复决定、非法 JSON 都作为参数错误返回。

### `--edit-json` 的边界

```bash
# 只写要改的字段（补丁）
ta approve <AID> --decided-by c1 --edit-json '{"total_amount": 2400}'

# 以下都会被拒绝
--edit-json '{"name": "cancel_mock_booking"}'   # 不允许改工具名
--edit-json '{"discount": 100}'                 # 不允许注入新键
--edit-json '{"total_amount": -5}'              # 违反工具 Schema
```

---

## 配置

```env
# 模型（OpenAI 兼容）
TRAVEL_AGENT_MODEL_MODE=openai_compatible
TRAVEL_AGENT_MODEL_NAME=...
TRAVEL_AGENT_API_KEY=...
TRAVEL_AGENT_BASE_URL=...

# 数据源：每类独立切换 sqlite | flyai
TRAVEL_AGENT_HOTEL_PROVIDER=sqlite
TRAVEL_AGENT_TRANSPORT_PROVIDER=sqlite
TRAVEL_AGENT_ATTRACTION_PROVIDER=sqlite
TRAVEL_AGENT_FLYAI_COMMAND=flyai
TRAVEL_AGENT_FLYAI_TIMEOUT_SECONDS=30
TRAVEL_AGENT_FLYAI_MAX_ATTEMPTS=3          # 限流重试上限
TRAVEL_AGENT_FLYAI_BACKOFF_SECONDS=0.5     # 指数退避基准
TRAVEL_AGENT_FLYAI_CONCURRENCY=2           # 并发闸门，0 关闭

# 上下文窗口（4 字符约 1 token 估算）
TRAVEL_AGENT_CONTEXT_MAX_MESSAGES=12
TRAVEL_AGENT_CONTEXT_MAX_TOKENS=4000
```

`.env` 被 gitignore，仓库只提供 `.env.example`。密钥绝不入仓库。

---

## 架构

```text
CLI（argparse，13 个子命令，只做参数解析和 JSON 输出）
        |
        v
ConsultationService（service.py，唯一业务入口）
        |
        v
LangGraph create_agent（agent.py）
        |
        +--> Chat Model（OpenAI-compatible）
        +--> Middleware 栈（agent_context.py）
        |        TravelMemoryMiddleware    偏好/需求/摘要/版本注入
        |        ContextWindowMiddleware   消息窗口裁剪
        |        HumanInTheLoopMiddleware  敏感工具前中断
        |
        v
StructuredTool（langchain_tools.py）
        |
        +--> 查询工具 x3 -> Provider 工厂 -> SQLite / FlyAI
        +--> 敏感工具 x2 -> Pydantic 校验 -> 幂等写库
        |
        v
业务 SQLite（repository.py） / LangGraph checkpoint / 审计事件
```

**关键原则**：CLI 里没有业务逻辑，所有决策在 `service.py`；状态列有数据库级 CHECK 约束，脏数据写不进去。

### 数据表

```text
consultations          咨询 + 结构化需求
messages               业务侧用户/助手消息
itinerary_drafts       行程草稿（含 selected_version_id）
itinerary_versions     方案版本（含 budget）
approval_requests      审批请求（interrupt_id + action_index 幂等去重）
mock_bookings          模拟订单（创建/取消各自幂等键）
customer_preferences   跨咨询客户偏好
audit_events           全链路审计
provider_requests      供应商查询记录
```

---

## 工程细节：几个真实踩过的坑

这些不是设计说明，是运行中真实发生并修复的问题，每个都留下了回归测试。

**`interrupt()` 是可重放的。** 首次调用抛出 `GraphInterrupt`，恢复时节点会**从头重新执行**，而不是从断点继续。因此恢复逻辑必须幂等。

**`APPROVAL_RESUMED` 必须在提前返回之前写入。** 否则被消费过的 interrupt 会一直被视为"已决定但未执行"，在后续恢复时被优先命中，导致新的暂停**永远无法批准**。

**`asyncio.Semaphore` 绑定创建它的事件循环。** `service.py` 每请求一次 `asyncio.run()`，每次换新 loop。在 `__init__` 里建信号量会在第二个 loop 上报错，在真实子进程路径上更糟——**直接死锁**。改为按运行中的 loop 惰性创建并缓存。

**这个 bug 只有真实子进程能复现。** 同步 runner 从不挂起，`acquire`/`release` 一气呵成，永远没有等待者。对应的回归测试必须用真实子进程制造争用。

**错误信息是产品的一部分。** `itinerary version must be selected before booking` 这句话让排查只能靠手工翻库。现在的报错会给出请求版本、当前选定状态、可用版本列表，以及修复命令：

```text
itinerary version must be selected before booking: requested 18e9e548-...,
but no version is selected yet. Available versions: v1=18e9e548-..., v2=8ce62d22-....
Run `travel-agent select-itinerary c0ffe693-... 18e9e548-...` and retry the booking.
```

**模型不知道哪个版本是"当前的"。** 版本列表原本只存在于数据库和 CLI 输出里，模型上下文只有对话中出现过的 id，于是它倾向复用最早见过的那个。现在每轮 chat 都把最新版本列表连同 `← 已选定` 标记注入 system prompt，未选定则提前警告"直接提交会失败"。

**这个修复差点失效。** 最初把版本信息塞进 `turn_state`，但该字典只在 `history_status != "pending"` 时随输入下发，而多轮对话恰好走 pending 分支——版本信息被整个丢弃。真实模型测试暴露了它（模型说"我手里没有这两个版本的明细"）。回归测试固定了这个接入点：撤销那一行，测试立即变红。

**修好约束不等于修好历史任务。** 旧库有一行 `approval_requests.status = '1'`（不在枚举内），读取时抛 `ValueError`，导致审批单既批不了也拒不了。加了 CHECK 约束并迁移后脏值被归一为 `PENDING`，但如果对应的 checkpoint 早已不存在，恢复仍然失败——这类咨询应当放弃重走。

---

## 测试

```bash
python -m pytest -q
```

**174 passed**，29 个测试文件，**全部离线**。

| 方面 | 做法 |
|---|---|
| 只替换什么 | 仅替换 Chat Model 与 checkpoint 存储 |
| 保留什么 | 真实 CLI、Service、`create_agent`、Middleware、StructuredTool、SQLite |
| 覆盖 | HITL 三种决定、多动作暂停、拒绝后重规划、崩溃恢复、迁移与约束、版本注入、FlyAI 限流与跨 loop 信号量 |
| 真实模型验证 | 作为独立的联网验收，不算入离线测试 |

离线测试证明确定性逻辑正确，**不证明**真实模型的语义判断。两者分开记录。

---

## 当前状态与后续

### 已实现

- 真实 Agent 对话闭环、双轨持久化（阶段 3）
- 客户偏好、需求覆盖、窗口裁剪、双层摘要、统一记忆注入（阶段 4）
- HITL 审批、暂停恢复、幂等、拒绝重规划（阶段 5）
- APPROVAL/ORDER 状态约束与迁移、可操作报错、版本上下文注入、FlyAI 韧性（阶段 6）

### 未实现

| 缺口 | 说明 |
|---|---|
| 工具权限/预算网关 | 真实 `StructuredTool` 目前**绕过**任何网关直接执行 |
| 审批金额一致性校验 | `create_mock_booking` 只校验金额为正，不校验是否等于版本冻结的 `budget.total`。真实走查中模型连续拒了三次才暴露此缺口 |
| 审批有效期 | `expires_at` / `EXPIRED` 只有字段与枚举，无过期策略 |
| HTTP 接口 | `api.py` 已删除，当前只有 CLI |
| 长驻事件循环 | `service.py` 每请求一次 `asyncio.run()`，属架构隐患 |
| 评测体系 | 缺少可重复的任务级评分（阶段 7） |

### 删除过什么

项目早期有一套 mock 规划链路（`planning.py` 的规划图、`mock_model.py`、`mock_providers.py`、`tool_gateway.py` 等 9 个模块）。它和真实 Agent 无关，还固化过一个 bug——门票没有乘出行人数，三条测试甚至把错误总额写成了期望值。

删除后只保留真实链路，方案版本改由 chat 的工具结果派生。**但缺陷的教训被保留**：新的预算计算同样按 `traveler_count` 乘门票，并有专门回归测试。删除模块不等于删除教训。

---

## 项目结构

```text
src/travel_agent/
├── cli.py                    13 个命令，只做参数解析和输出
├── service.py                业务编排：对话、审批、版本、恢复
├── agent.py                  create_agent + Middleware 装配
├── agent_context.py          记忆/摘要/版本注入、窗口裁剪
├── langchain_tools.py        5 个 StructuredTool
├── repository.py             SQLite 持久化、约束、迁移
├── domain.py                 领域模型与枚举
├── requirement_extractor.py  需求提取（中文日期归一）
├── providers.py              Provider Protocol
├── sqlite_providers.py       本地数据适配器
├── flyai_client.py           FlyAI 子进程客户端（重试/回收/并发闸门）
├── flyai_providers.py        FlyAI 适配器
├── flyai_parsers.py          FlyAI 响应标准化
├── flyai_errors.py           错误分级
├── provider_factory.py       Provider 装配
├── context_summary.py        摘要生成
├── booking.py                预订请求模型
└── application.py            依赖装配
```

`schedule.md` 记录了完整的阶段规划、实现约定与真实故障复盘。

---

## 许可

学习项目，未附带许可证。

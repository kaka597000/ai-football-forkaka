# Football Prediction API — Code Wiki

> 基于 FastAPI + MySQL + XGBoost + Poisson + DeepSeek LLM 的足球比赛数据同步与智能预测系统完整文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈](#2-技术栈)
3. [项目结构](#3-项目结构)
4. [整体架构](#4-整体架构)
5. [模块详解](#5-模块详解)
   - 5.1 [入口与生命周期](#51-入口与生命周期-mainpy)
   - 5.2 [Core 配置层](#52-core-配置层)
   - 5.3 [DB 数据库层](#53-db-数据库层)
   - 5.4 [Models ORM 模型层](#54-models-orm-模型层)
   - 5.5 [Schemas Pydantic 层](#55-schemas-pydantic-层)
   - 5.6 [Services 业务逻辑层](#56-services-业务逻辑层)
   - 5.7 [API 路由层](#57-api-路由层)
   - 5.8 [Prediction 预测推理层](#58-prediction-预测推理层)
   - 5.9 [Training 模型训练层](#59-training-模型训练层)
   - 5.10 [Tools 命令行工具层](#510-tools-命令行工具层)
   - 5.11 [Alembic 数据库迁移](#511-alembic-数据库迁移)
6. [数据库表结构](#6-数据库表结构)
7. [核心数据流](#7-核心数据流)
   - 7.1 [数据同步流程](#71-数据同步流程)
   - 7.2 [单场预测流程](#72-单场预测流程)
   - 7.3 [模型训练流程](#73-模型训练流程)
8. [依赖关系图](#8-依赖关系图)
9. [模型分组与回退机制](#9-模型分组与回退机制)
10. [预测结果表结构详解](#10-预测结果表结构详解)
11. [配置说明](#11-配置说明)
12. [运行方式](#12-运行方式)
    - 12.1 [开发环境准备](#121-开发环境准备)
    - 12.2 [启动 API 服务](#122-启动-api-服务)
    - 12.3 [数据初始化](#123-数据初始化)
    - 12.4 [重新训练模型](#124-重新训练模型)
13. [调度系统详解](#13-调度系统详解)
14. [中文字段兜底机制](#14-中文字段兜底机制)
15. [附录：API 端点汇总](#15附录api-端点汇总)

---

## 1. 项目概述

本项目是一套完整的足球比赛智能预测后端系统。它通过 API-Football 数据源采集全球足球赛事数据（联赛、球队、球员、积分榜、赛程、实时赛况、比赛统计等），沉淀到 MySQL 本地数据仓库；然后通过 **XGBoost + Poisson + DeepSeek LLM** 三层组合模型输出可解释的比赛预测结果（胜平负概率、大小球、比分、让球盘、深度分析）；同时提供 FastAPI 对外服务和 CLI 工具，支持数据同步、批量预测、赛后验证回填和定时调度。

### 核心能力

| 能力 | 说明 |
|------|------|
| 数据采集闭环 | 从 API-Football 拉取全链路数据并落库 |
| 多模型组合预测 | XGBoost 输出结构化概率 → Poisson 生成比分分布 → LLM 最终决策 |
| 模型回退 | 联赛专属 → 赛事大组 → GLOBAL 全局模型 |
| 赛后复盘 | 回填实际比分，验证 LLM 预测的准确性 |
| 定时调度 | asyncio 驱动的 8 个内置任务 |
| API + CLI 双入口 | REST API 和命令行工具均可触发预测与同步 |

---

## 2. 技术栈

### 后端框架与数据

| 技术 | 版本 | 用途 |
|------|------|------|
| FastAPI | 0.115.0 | REST API 框架 |
| Uvicorn | 0.30.0 | ASGI 服务器 |
| SQLAlchemy | 2.0.35 | ORM |
| Alembic | 1.13.0 | 数据库迁移 |
| PyMySQL | 1.1.1 | MySQL 驱动 |
| Pydantic | 2.9.0 | Schema 校验 |
| Loguru | 0.7.2 | 日志管理 |
| Rich | 15.0.0 | CLI 美化输出 |
| Typer | 0.25.1 | CLI 框架 |

### 机器学习与分析

| 技术 | 版本 | 用途 |
|------|------|------|
| XGBoost | 3.2.0 | 结构化数据预测（分类 + 回归） |
| scikit-learn | 1.8.0 | 训练评估、时间切分 |
| pandas | 3.0.3 | 特征工程向量化计算 |
| numpy | 2.4.6 | 数值计算 |
| scipy | 1.17.1 | Poisson 分布概率计算 |
| joblib | 1.5.3 | 模型持久化 |
| Optuna | 4.8.0 | 超参数调优（可选） |

### LLM / API

| 技术 | 用途 |
|------|------|
| DeepSeek API | 最终预测决策与自然语言分析生成 |
| httpx / aiohttp | HTTP 请求（同步/异步） |
| API-Football | 足球数据上游 |

---

## 3. 项目结构

```
backend/
├── main.py                      # FastAPI 应用入口
├── requirements.txt             # Python 依赖
├── alembic.ini                  # Alembic 迁移配置
├── .env.example                 # 环境变量示例
│
├── alembic/                     # 数据库迁移（24 个版本）
│   ├── env.py                   # 迁移环境配置
│   ├── script.py.mako           # 迁移脚本模板
│   ├── README
│   └── versions/                # 迁移版本脚本
│
├── app/                         # 主应用包
│   ├── __init__.py              # 空文件
│   │
│   ├── core/                    # 核心配置
│   │   ├── config.py            # Pydantic Settings（.env 加载）
│   │   ├── log_config.py        # Loguru 日志配置
│   │   ├── whitelist.py         # 白名单联赛 ID
│   │   └── zh.py                # 中文字段兜底工具
│   │
│   ├── db/                      # 数据库连接
│   │   ├── base.py              # SQLAlchemy Base
│   │   └── session.py           # Engine + SessionLocal + get_db
│   │
│   ├── models/                  # ORM 模型
│   │   ├── league.py            # League + Season
│   │   ├── team.py              # Team + Venue
│   │   ├── player.py            # Player + PlayerStats
│   │   ├── standing.py          # Standing（积分榜）
│   │   └── fixture.py           # Fixture + FixtureEvent + FixtureLineup + FixtureStatistic + FixturePlayerStat
│   │
│   ├── schemas/                 # Pydantic Schema
│   │   ├── league.py            # LeagueSchema + PaginatedResponse[T]
│   │   ├── team.py              # TeamSchema + VenueSchema
│   │   ├── player.py            # PlayerSchema + PlayerStatsSchema
│   │   ├── standing.py          # StandingSchema
│   │   └── fixture.py           # FixtureSchema + 子表 Schema
│   │
│   ├── services/                # 业务服务
│   │   ├── scheduler.py         # asyncio 调度引擎（8 个默认任务）
│   │   ├── league_service.py    # 联赛/赛季同步
│   │   ├── team_service.py      # 球队/场馆同步
│   │   ├── player_service.py    # 球员/球员统计同步
│   │   ├── standing_service.py  # 积分榜同步
│   │   ├── fixture_service.py   # 赛程同步 + 子数据并发拉取
│   │   ├── auto_predict_service.py   # 赛前自动预测
│   │   └── prediction_result_service.py  # 赛后回填验证
│   │
│   └── api/                     # REST 路由
│       ├── leagues.py           # /api/leagues*
│       ├── teams.py             # /api/teams*
│       ├── players.py           # /api/players*
│       ├── standings.py         # /api/standings*
│       ├── fixtures.py          # /api/fixtures*
│       ├── scheduler.py         # /api/scheduler*
│       └── predictions.py       # /api/predictions*
│
├── prediction/                  # 预测核心包
│   ├── __init__.py
│   ├── features.py              # 特征工程（批量 + 单场）
│   ├── predict.py               # 单场预测主流程
│   ├── train.py                 # 训练入口
│   ├── features.csv             # 训练特征输出
│   ├── metrics.csv              # 模型评估报告
│   ├── models/                  # 已训练模型文件（.pkl）
│   │   ├── feature_cols.pkl     # 特征列名
│   │   ├── GLOBAL_win.pkl       # GLOBAL 模型文件
│   │   ├── PL_win.pkl           # 英超专属模型
│   │   ├── LALIGA_win.pkl       # 西甲专属模型
│   │   ├── SERIEA_win.pkl       # 意甲专属模型
│   │   ├── BUNDESLIGA_win.pkl   # 德甲专属模型
│   │   ├── LIGUE1_win.pkl       # 法甲专属模型
│   │   └── ... (每个模型组 5 个文件)
│   │
│   └── training/                # 训练核心逻辑
│       ├── __init__.py
│       ├── data.py              # 数据加载 + 分组逻辑
│       ├── model.py             # XGBoost 模型定义、训练、保存/加载
│       └── evaluate.py           # 评估指标（accuracy / AUC / RPS / logloss）
│
└── tools/                       # 命令行工具脚本
    ├── manual_predict.py        # Rich 交互式单场预测
    ├── batch_predict.py         # 批量预测指定日期
    ├── sync_fixtures.py         # 完赛子数据补完
    ├── sync_players.py          # 单联赛球员同步
    ├── async_sync_subdata.py    # asyncio + asyncmy 高并发子数据同步
    └── retry_fixtures.py        # 重试失败的联赛赛程
```

---

## 4. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户 / 前端 / Bot                           │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ HTTP / CLI
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (main.py)                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐  │
│  │ leagues │ │ teams   │ │ players │ │ fixtures│ │ predictions  │  │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──────┬───────┘  │
│       │           │           │           │             │          │
│  ┌────┴───────────┴───────────┴───────────┴─────────────┴───────┐  │
│  │                       API Layer (api/*.py)                     │  │
│  └─────────────────────────────┬────────────────────────────────┘  │
│                                │                                    │
│  ┌─────────────────────────────┴────────────────────────────────┐  │
│  │                     Service Layer (services/*.py)              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐   │  │
│  │  │ sync_*   │ │ scheduler│ │auto_pred │ │ backfill_result│   │  │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬────────┘   │  │
│  └───────┼────────────┼────────────┼───────────────┼─────────────┘  │
│          │            │            │               │                │
└──────────┼────────────┼────────────┼───────────────┼────────────────┘
           │            │            ▼               │
           │            │  ┌──────────────────┐      │
           │            │  │ Prediction Layer  │      │
           │            │  │ predict.py       │      │
           │            │  │ features.py      │      │
           │            │  │  ┌─────────────┐  │      │
           │            │  │  │ Training    │  │      │
           │            │  │  │ model.py    │  │      │
           │            │  │  │ data.py     │  │      │
           │            │  │  │ evaluate.py │  │      │
           │            │  │  └─────────────┘  │      │
           │            │  └──────────────────┘      │
           │            ▼                              │
┌──────────┼───────────────────────────────────────────┼──────────────┐
│          ▼                                           ▼              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   MySQL 数据库 (ORM Models)                    │  │
│  │  leagues / seasons / teams / venues / players / standings    │  │
│  │  fixtures + 4 张子表 + predictions + scheduler_* 日志表      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           Data Layer                                │
└─────────────────────────────────────────────────────────────────────┘
           │                                                     │
           ▼                                                     ▼
┌──────────────────┐                                   ┌────────────────┐
│  API-Football v3 │                                   │ DeepSeek API    │
│  （足球数据源）   │                                   │（LLM 分析）     │
└──────────────────┘                                   └────────────────┘
```

### 核心设计要点

| 设计点 | 说明 |
|--------|------|
| 三层预测架构 | XGBoost（结构化概率） → Poisson（比分分布） → LLM（最终决策） |
| 回退机制 | 联赛专属模型 > 赛事大组模型 > GLOBAL 全局模型 |
| 反范式存储 | fixtures 表存储 league_name/home_name 等字符串（带 _zh 兜底），避免多表 JOIN |
| 定时引擎 | asyncio.Task + 数据库持久化状态，支持启动/停止/热更新 |
| 并发拉取 | ThreadPoolExecutor 批量拉取子数据，asyncmy 高并发备用 |
| 时间切分 | 训练集/测试集按时间顺序切分（后 20% 作测试集） |

---

## 5. 模块详解

### 5.1 入口与生命周期 `main.py`

**文件**：[main.py](backend/main.py)

#### 关键函数

##### `lifespan(app: FastAPI)` — FastAPI 生命周期管理

- **触发时机**：FastAPI 启动时触发，启动完成后进入 yield，停止时触发 finally
- **主要职责**：
  1. 调用 `setup_logger()` 初始化 Loguru 日志（控制台 + 文件按日轮转，保留 30 天）
  2. 调用 `init_scheduler()` 加载 scheduler_tasks 表中 `is_enabled=1` 的任务并启动对应 asyncio loop
  3. 停止时调用 `shutdown_scheduler()` 优雅取消所有 loop

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在注册定时任务...")
    init_scheduler()
    try:
        yield
    except BaseException:
        pass
    finally:
        await shutdown_scheduler()
```

> **重要说明**：代码中禁用了 Alembic 自动迁移（autogenerate 会删除没有 SQLAlchemy 模型的表，如 predictions、scheduler_tasks、scheduler_logs 均使用 raw SQL 管理）。新增表需手动建表或手动编写迁移脚本。

##### `app = FastAPI(...)` — 应用实例

注册 7 组路由，全部以 `/api` 为前缀：

| Router | 模块 | 标签 |
|--------|------|------|
| leagues_router | app.api.leagues | Leagues |
| teams_router | app.api.teams | Teams |
| players_router | app.api.players | Players |
| standings_router | app.api.standings | Standings |
| fixtures_router | app.api.fixtures | Fixtures |
| scheduler_router | app.api.scheduler | Scheduler |
| predictions_router | app.api.predictions | Predictions |

CORS 全开（`allow_origins=["*"]`，允许所有方法和头部）。

**Health Check**：`GET /api/health` → `{"status": "ok"}`

---

### 5.2 Core 配置层

#### 5.2.1 `config.py` — 全局配置

**文件**：[app/core/config.py](backend/app/core/config.py)

```python
class Settings(BaseSettings):
    # MySQL 数据库连接
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "football"

    # API-Football 数据源
    api_football_key: str = ""
    api_football_base_url: str = ""

    # DeepSeek LLM
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # 缓存
    cache_ttl: int = 3600
```

- 通过 `pydantic-settings` 从当前工作目录的 `.env` 文件加载
- `db_url` 是动态拼接的属性方法：`mysql+pymysql://user:password@host:port/db`
- 全局单例 `settings = Settings()` 被所有服务模块引用

#### 5.2.2 `log_config.py` — 日志

**文件**：[app/core/log_config.py](backend/app/core/log_config.py)

```python
def setup_logger():
    logger.remove()                      # 清空默认 handler
    logger.add(sys.stdout, ...)          # 控制台 DEBUG 级别（彩色）
    logger.add("logs/football_{time:YYYY-MM-DD}.log",
               rotation="00:00",          # 每天 0 点轮转
               retention="30 days",       # 保留 30 天
               level="INFO")             # 文件 INFO 级别
```

输出格式：`时间 | 级别 | 模块:函数:行号 - 消息`

#### 5.2.3 `whitelist.py` — 白名单联赛 ID

**文件**：[app/core/whitelist.py](backend/app/core/whitelist.py)

```python
WHITELIST_LEAGUE_IDS = {
    1, 2, 3, 10, 11, 13, 17, 18, 39, 40, 41, 45,
    61, 62, 71, 78, 79, 88, 89, 94, 98, 99, 103, 106,
    113, 128, 135, 140, 141, 144, 169, 188, 197, 203, 204,
    235, 244, 253, 262, 265, 274, 283, 292, 293, 296,
    307, 323, 333, 666, 848, 1097,
}
```

联赛同步时，如果 `league_raw["id"]` 在此集合内，则自动置 `enabled=True`。预测、赛程同步、球队同步等下游均依赖 `League.enabled` 过滤。

#### 5.2.4 `zh.py` — 中文字段兜底

**文件**：[app/core/zh.py](backend/app/core/zh.py)

系统设计约定：所有主表（leagues/teams/players/venues/fixture_events/fixture_statistics）都有 `*_zh` 列，翻译脚本写入后，API 返回时自动用 `_zh` 替换原列。

**核心映射 `ZH_MAPS`**：

| 类名 | 原字段 | 中文字段 |
|------|--------|----------|
| League | name | name_zh |
| League | country_name | country_name_zh |
| Team | name | name_zh |
| Team | country | country_zh |
| Player | name | name_zh |
| Venue | name | name_zh |
| Venue | city | city_zh |
| FixtureEvent | type | type_zh |
| FixtureEvent | detail | detail_zh |
| FixtureStatistic | stat_type | stat_type_zh |

**关键函数**：

| 函数 | 作用 |
|------|------|
| `zh_swap(obj)` | 单对象 in-place 替换（ORM 对象，session autoflush=False 只读） |
| `zh_swap_many(objs)` | 批量遍历调用 `zh_swap` |
| `fixtures_apply_denorm_zh(db, fixtures)` | fixtures 表反范式字段（league_name/home_name/away_name/venue_name）按 ID 批量回查 leagues/teams/venues 表的 `*_zh` 覆盖 |
| `standings_apply_denorm_zh(db, standings)` | standings.team_name 按 team_id 回查 teams.name_zh 覆盖 |

---

### 5.3 DB 数据库层

#### 5.3.1 `base.py`

**文件**：[app/db/base.py](backend/app/db/base.py)

```python
from sqlalchemy.orm import declarative_base
Base = declarative_base()
```

所有 ORM 模型类都继承自这个 Base。

#### 5.3.2 `session.py`

**文件**：[app/db/session.py](backend/app/db/session.py)

```python
engine = create_engine(
    settings.db_url,
    pool_size=10,        # 连接池大小
    max_overflow=20,     # 溢出连接
    pool_pre_ping=True,  # 连接有效性检查
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

**`get_db()`** — FastAPI 依赖注入，每次请求一个独立 session，请求结束后自动关闭：

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

### 5.4 Models ORM 模型层

#### 模型总览

| 类名 | 表名 | 主键 | 关系 |
|------|------|------|------|
| League | leagues | id (API ID) | Season (一对多) |
| Season | seasons | id (自增) | League (多对一) |
| Team | teams | id (API ID) | Venue (多对一) |
| Venue | venues | id (API ID) | Team (一对多) |
| Player | players | id (API ID) | PlayerStats (一对多) |
| PlayerStats | player_stats | id (自增) | Player (多对一) |
| Standing | standings | id (自增) | 无（独立表） |
| Fixture | fixtures | id (API ID) | FixtureEvent/Lineup/Statistic/PlayerStat (一对多) |
| FixtureEvent | fixture_events | id (自增) | Fixture (多对一) |
| FixtureLineup | fixture_lineups | id (自增) | Fixture (多对一) |
| FixtureStatistic | fixture_statistics | id (自增) | Fixture (多对一) |
| FixturePlayerStat | fixture_player_stats | id (自增) | Fixture (多对一) |

#### 各模型关键字段

##### `League` — 联赛

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer (PK) | API-Football 联赛 ID（39=英超, 140=西甲 等） |
| name | String(255) | 联赛名称 |
| name_zh | String(255) | 中文名（翻译脚本写入） |
| type | String(20) | League / Cup |
| country_name / country_name_zh | String | 国家（英/中） |
| enabled | Boolean | 是否启用（白名单自动 True） |

##### `Season` — 赛季

| 字段 | 类型 | 说明 |
|------|------|------|
| league_id | Integer (FK) | 所属联赛 |
| year | Integer | 赛季年份 |
| is_current | Boolean | 是否当前赛季 |
| coverage | JSON | API 返回的数据覆盖范围 |

##### `Team` + `Venue` — 球队与场馆

Team 核心字段：`id`, `name`, `name_zh`, `national`, `code`, `country`, `logo`, `venue_id`(FK)

Venue 核心字段：`id`, `name`, `name_zh`, `city`, `city_zh`, `capacity`, `surface`, `image`

##### `Player` + `PlayerStats` — 球员与赛季统计

Player 核心字段：`id`, `name`, `name_zh`, `age`, `nationality`, `injured`

PlayerStats 核心字段：`player_id`, `team_id`, `league_id`, `season` + 11 个 JSON 统计块（games/substitutes/shots/goals/passes/tackles/duels/dribbles/fouls/cards/penalty）

##### `Standing` — 积分榜

独立表，不依赖 ORM 关系。关键字段：`league_id`, `season`, `group_name`（小组赛/开幕/闭幕）, `rank`, `team_id`, `team_name`, `points`, `goals_diff`, `form`(WWDLW), `status`(same/up/down) + 全主客三维度的 played/win/draw/lose/goals_for/goals_against。

##### `Fixture` + 4 张子表 — 比赛详情

Fixture 核心字段：`id`, `date`(UTC), `status_short`(FT/AET/PEN/NS...), `league_id/name`, `season`, `round`, `home_id/name/logo/winner`, `away_id/name/logo/winner`, `goals_home/away`, 半全场/加时/点球比分, `sub_data_synced`(布尔标记子数据是否已拉取), `category`(分类标签如 jingzu)

**4 张子表**：

| 子表 | 关键字段 |
|------|----------|
| FixtureEvent | elapsed/minute, type(Goal/Card/Subst/Var), team/player/assist ID+name |
| FixtureLineup | team_id, formation, player_id/name/number/pos, is_substitute |
| FixtureStatistic | team_id, stat_type (Total Shots/Ball Possession 等), stat_value |
| FixturePlayerStat | player_id/name/photo + games/shots/goals/passes/tackles 等 JSON 块 |

#### ORM 导入汇总

所有模型在 [app/models/__init__.py](backend/app/models/__init__.py) 中统一导出：

```python
from app.models.league import League, Season
from app.models.team import Team, Venue
from app.models.player import Player, PlayerStats
from app.models.standing import Standing
from app.models.fixture import Fixture, FixtureEvent, FixtureLineup, FixtureStatistic, FixturePlayerStat
```

---

### 5.5 Schemas Pydantic 层

所有 Schema 继承 `BaseModel`（Pydantic v2），使用 `class Config: from_attributes = True`（替代旧版 `orm_mode = True`）。

#### 通用分页容器

[app/schemas/league.py](backend/app/schemas/league.py) 中定义了泛型分页容器，被多处复用：

```python
T = TypeVar("T")
class PaginatedResponse(BaseModel, Generic[T]):
    data: list[T]
    total: int
    page: int
    page_size: int
```

#### Schema 设计模式

每个业务实体有两层 Schema：

| Schema | 包含的关系数据 | 用途 |
|--------|----------------|------|
| `XxxSchema` | 基础字段 | 列表页展示 |
| `XxxDetailSchema` | 基础字段 + 嵌套子表 | 详情页展示 |

示例：
- `LeagueSchema` → 基础联赛字段（不含 seasons）
- `LeagueDetailSchema` → LeagueSchema + `seasons: list[SeasonSchema]`
- `TeamSchema` → 基础球队字段（不含 venue）
- `TeamDetailSchema` → TeamSchema + `venue: Optional[VenueSchema]`
- `PlayerSchema` → 基础球员字段（不含 statistics）
- `PlayerDetailSchema` → PlayerSchema + `statistics: list[PlayerStatsSchema]`
- `FixtureSchema` → 基础比赛字段（不含 4 张子表）
- `FixtureDetailSchema` → FixtureSchema + 4 张子表的列表

#### fixture.py 子表 Schema

`FixtureEventSchema`, `FixtureLineupSchema`, `FixtureStatisticSchema`, `FixturePlayerStatSchema` 均对应 fixture.py 中的 4 张子表 ORM 模型。

#### standing.py

独立的 `StandingSchema`，包含全部字段（league_id, season, group_name, rank, team_name, points, goals_diff, form + 全主客三维度统计）。

#### league.py

还定义了 `CoverageSchema`（API 返回的赛季数据覆盖范围）和 `SeasonSchema`。

---

### 5.6 Services 业务逻辑层

所有 Service 函数接收 `db: Session` 参数，内部执行 upsert/merge + commit。

#### 5.6.1 League Service — `league_service.py`

**文件**：[app/services/league_service.py](backend/app/services/league_service.py)

**`sync_leagues(db: Session)`** — 核心同步函数

1. 检查 `settings.api_football_key` 是否配置
2. GET `{base_url}/leagues` 获取全量联赛列表
3. 对每个联赛执行 upsert（`db.query(League).get(id)` 后 update 或新建）
4. 白名单自动启用：`if league_raw["id"] in WHITELIST_LEAGUE_IDS: league.enabled = True`
5. 对联赛下的每个 `seasons[]` 执行 upsert（按 `(league_id, year)` 查存）
6. `db.commit()` 提交

**辅助函数**：
- `parse_coverage(raw: dict) -> dict` — 将 API 返回的 coverage 对象展平为扁平字典存储
- `_parse_date(value: str | None) -> date | None` — ISO 日期解析，失败返回 None

#### 5.6.2 Team Service — `team_service.py`

**文件**：[app/services/team_service.py](backend/app/services/team_service.py)

**`sync_teams(db: Session)`**

1. 查所有 `League.enabled == True` 且 `Season.is_current == True` 的赛季
2. 对每个赛季 GET `{base_url}/teams?league=&season=`
3. 用 `db.merge(Venue(...))` upsert 场馆（去重 seen_venues set）
4. 用 `db.merge(Team(...))` upsert 球队（去重 seen_teams set）
5. 每个赛季处理完 commit 一次

#### 5.6.3 Player Service — `player_service.py`

**文件**：[app/services/player_service.py](backend/app/services/player_service.py)

**`sync_players(db: Session)`** — 遍历所有启用联赛当前赛季

**`_sync_league_players(db, league_id, season_year)`** — 单联赛球员同步（私有函数）

1. 分页拉取 GET `{base_url}/players?league=&season=&page=`，直到 `page >= total_pages`
2. 对每个球员 upsert Player（seen_players 去重）
3. 对球员的每个 `statistics[]` 块（可能有多个 team/league）：
   - 按 `(player_id, team_id, league_id, season)` 查 PlayerStats 是否存在
   - 存在则 update 所有 JSON 块，不存在则 insert
4. 每处理一页 commit

**`sync_players_by_league(db, league_id)`** — 按指定联赛单独同步（CLI 工具使用）

#### 5.6.4 Standing Service — `standing_service.py`

**文件**：[app/services/standing_service.py](backend/app/services/standing_service.py)

**`sync_standings(db: Session)`**

1. 遍历所有启用联赛当前赛季
2. GET `{base_url}/standings?league=&season=`
3. standings 是二维数组 `[[group1_rows], [group2_rows]]`，需要双层遍历
4. 按 `(league_id, season, group_name, team_id)` 查 Standing 是否存在
5. 存在则 `_update_standing()` 更新，不存在则 `_make_standing()` 新建
6. 每个赛季 commit

**`_make_standing()` / `_update_standing()`** — 构造/更新 Standing 对象，映射 API 的 `all[]`, `home[]`, `away[]` 三维度统计到对应字段。

#### 5.6.5 Fixture Service — `fixture_service.py`

**文件**：[app/services/fixture_service.py](backend/app/services/fixture_service.py) — 最复杂的 Service

##### 主函数

| 函数 | 说明 |
|------|------|
| `_upsert_fixture(db, row)` | upsert 一条 Fixture 主表记录（解析 API 嵌套结构） |
| `sync_fixtures(db)` | 每日同步：遍历启用联赛当前赛季，拉取昨日+今日赛程 |
| `sync_completed_sub_data(db)` | 为完赛（status_short in FT/AET/PEN/AWD/WO）且子数据未同步的比赛并发拉取 4 张子表 |
| `sync_live_fixtures(db)` | 实时同步：按今日+昨日拉取比赛状态和比分，检测"刚结束"触发异步子数据拉取 |

##### 子数据并发拉取架构

`sync_completed_sub_data` 使用三层并发：

```
┌─ BATCH_SIZE=4（外层分批） ─────────────────────┐
│                                                 │
│  ThreadPoolExecutor(max_workers=4) ──────────── │
│  ├── _sync_one_fixture_sub_data(fixture_id)     │
│  │   └── 内部：ThreadPoolExecutor(max_workers=4)│
│  │       ├── _fetch_events()                     │
│  │       ├── _fetch_lineups()                   │
│  │       ├── _fetch_statistics()                │
│  │       └── _fetch_player_stats()              │
│  ├── (第二场) ...                                │
│  └── (第三场) ...                                │
└─────────────────────────────────────────────────┘
```

**4 个子数据拉取函数**（私有）：

| 函数 | API 端点 | 子表 |
|------|----------|------|
| `_fetch_events(db, fixture_id)` | `/fixtures/events` | fixture_events |
| `_fetch_lineups(db, fixture_id)` | `/fixtures/lineups` | fixture_lineups |
| `_fetch_statistics(db, fixture_id)` | `/fixtures/statistics` | fixture_statistics |
| `_fetch_player_stats(db, fixture_id)` | `/fixtures/players` | fixture_player_stats |

**`_do_fetch(db, fixture_id)`** — 单场 4 个 API 并行拉取，已存在的子表则跳过（纯 INSERT，无 DELETE）。全部成功后置 `sub_data_synced = True`。

**重试机制**：`_sync_one_fixture_sub_data` 失败时退避重试 3 次（2s → 4s → 6s）。

##### 实时同步的刚结束检测

`sync_live_fixtures` 跟踪 `status_short` 变化：如果 `new_short in {FT, AET, PEN, AWD, WO}` 且 `old_short not in 该集合`，则记录为"刚结束"并触发 `_sync_one_fixture_sub_data(fid)`。

#### 5.6.6 Scheduler — `scheduler.py`

**文件**：[app/services/scheduler.py](backend/app/services/scheduler.py)

##### 默认任务定义

```python
_DEFAULT_TASKS = {
    "league_sync":    {"name": "联赛数据同步",   "start_hour": 2,   "fn": sync_leagues},
    "team_sync":      {"name": "球队数据同步",   "start_hour": 2.5, "fn": sync_teams},
    "player_sync":    {"name": "球员数据同步",   "start_hour": 3,   "fn": sync_players},
    "standing_sync":  {"name": "积分榜数据同步", "start_hour": 4,   "fn": sync_standings},
    "fixture_daily":  {"name": "赛程每日同步",   "start_hour": 5,   "fn": sync_fixtures},
    "fixture_live":   {"name": "赛程实时同步",   "start_hour": None, "fn": sync_live_fixtures,
                       "interval_seconds": 120},   # 每 2 分钟
    "backfill_pred":  {"name": "预测结果回填",   "start_hour": None, "fn": backfill_results,
                       "interval_seconds": 3600},  # 每 1 小时
    "auto_predict":   {"name": "赛前自动预测",   "start_hour": 12,  "fn": auto_predict},
}
```

**两种调度模式**：
1. **定时任务**（`start_hour` 有值）：每天固定时间执行，如 `start_hour=2.5` → 每天 02:30
2. **间隔任务**（`interval_seconds` 有值）：周期性循环执行

##### 核心数据结构

- `_active_loops: dict[str, asyncio.Task]` — 运行中的 asyncio loop
- `scheduler_tasks` 表（raw SQL 管理）：持久化任务配置
  - 字段：`id, name, interval_seconds, start_hour, is_enabled, last_run, next_run, is_running`
- `scheduler_logs` 表：执行日志
  - 字段：`id, task_id, task_name, status(running/success/failed), started_at, finished_at, message`

##### 核心函数

| 函数 | 说明 |
|------|------|
| `init_scheduler()` | 种子化默认任务到 scheduler_tasks 表，启动所有 is_enabled 的 loop |
| `shutdown_scheduler()` | 优雅取消所有 asyncio.Task |
| `_seed_defaults()` | `INSERT IGNORE` 写入 8 个默认任务 |
| `_start_loop(task_key)` | `asyncio.create_task(_run_loop(task_key))` |
| `_run_loop(task_key)` | 主循环：计算下一次执行时间 → `asyncio.sleep(delay)` → 执行 → 循环 |
| `run_task_with_log(task_id, task_name, sync_fn, db)` | 执行 + 记录 scheduler_logs |
| `get_scheduler_status()` | 读取 scheduler_tasks 表返回状态 |
| `start_scheduled_task / stop_scheduled_task` | 启停单个任务 |
| `update_task(task_key, start_hour, interval_seconds, is_enabled)` | 更新任务配置并重启 loop |

##### `_run_loop` 执行逻辑

```python
async def _run_loop(task_key):
    while True:
        # 1. 读 DB 取当前配置
        row = _read_db_row(task_key)
        if not row or not row["is_enabled"]: break

        # 2. 计算下一执行时间
        if sh is not None:   # 定时
            target = now 对应日期的 start_hour
            delay = (target - now).seconds
        else:                # 间隔
            delay = interval_seconds

        _update_db(task_key, next_run=target)
        await asyncio.sleep(delay)

        # 3. 执行
        try:
            await asyncio.to_thread(run_task_with_log, task_key, name, fn)
        except asyncio.CancelledError: raise
        except Exception as e: logger.error(...)
```

> **线程切换**：`sync_fn` 是同步函数（SQLAlchemy session），通过 `asyncio.to_thread()` 放入线程池执行，避免阻塞 asyncio 主循环。

#### 5.6.7 Auto Predict Service — `auto_predict_service.py`

**文件**：[app/services/auto_predict_service.py](backend/app/services/auto_predict_service.py)

**`auto_predict(db=None)`** — 赛前自动预测（每天 12:00 触发）

1. 计算"今天北京日期"对应 UTC 范围（北京时间 10:10 为日界，当天 10:10 → 次日 10:10）
2. 查询 fixtures + JOIN leagues，筛选：
   - `fixture.date >= 起始 AND < 结束`
   - `status_short = 'NS'`（未开赛）
   - `league.enabled = 1`
   - 且（无 predictions 记录 OR predictions 缺少 LLM 字段 OR lineup 赛后更新）
3. 对每场调用 `prediction.predict.predict_fixture(fixture_id, db=db)`

#### 5.6.8 Prediction Result Service — `prediction_result_service.py`

**文件**：[app/services/prediction_result_service.py](backend/app/services/prediction_result_service.py)

**`backfill_results(db=None)`** — 赛后验证回填（每 1 小时触发）

1. 查询 predictions 表中 `actual_home_goals IS NULL` 的记录
2. 对每条调用 `_backfill_one(db, fixture_id)`：
   - 先从本地 fixtures 表读状态和比分
   - 如果 `status in {FT, AET, PEN, AWD, WO}` 且比分存在 → 直接用
   - 否则 GET API-Football `/fixtures?id=` 拉取最新数据并更新 fixtures 表
3. 计算 4 项正确性指标：
   - **win_correct**：LLM 预测的胜平负是否匹配实际（主胜=gh>ga, 平局=gh==ga, 客胜=gh<ga）
   - **over25_correct**：LLM 预测的大/小球是否匹配总进球（≥3=大, <3=小）
   - **handicap_correct**：按 LLM 给出的让球数调整后判断输赢（负值=主队让，正值=客队让）
   - **score_in_top3**：实际比分是否在 LLM 给出的 Top3 比分中
4. UPDATE predictions 表

---

### 5.7 API 路由层

所有路由使用 `async def` + `await asyncio.to_thread(_sync_fn)` 模式将同步 DB 操作放入线程池，避免阻塞 ASGI 循环。

#### 5.7.1 leagues.py

| 方法 | 路径 | 参数 | 说明 |
|------|------|------|------|
| GET | `/api/leagues` | page, page_size, search, enabled | 分页联赛列表（含 seasons 嵌套） |
| GET | `/api/leagues/{league_id}` | - | 联赛详情 |
| PATCH | `/api/leagues/{league_id}/toggle` | - | 启用/停用联赛 |

**zh_swap**：对 League 和 Season 列表调用 `zh_swap_many`。

#### 5.7.2 teams.py

| 方法 | 路径 | 参数 | 说明 |
|------|------|------|------|
| GET | `/api/teams` | page, page_size, search | 分页球队列表（含 venue） |
| GET | `/api/teams/{team_id}` | - | 球队详情 |

#### 5.7.3 players.py

| 方法 | 路径 | 参数 | 说明 |
|------|------|------|------|
| GET | `/api/players` | page, page_size, search | 分页球员列表（含 statistics → team/league） |
| GET | `/api/players/{player_id}` | - | 球员详情 |

使用 `selectinload` 链式加载多级关系：`Player.statistics → PlayerStats.team` 和 `PlayerStats.league`。

#### 5.7.4 standings.py

| 方法 | 路径 | 参数 | 说明 |
|------|------|------|------|
| GET | `/api/standings` | league_id (必填), season (可选), page, page_size | 积分榜查询 |

**zh**：调用 `standings_apply_denorm_zh` 回查 teams.name_zh 覆盖反范式存储的 team_name。

#### 5.7.5 fixtures.py

| 方法 | 路径 | 参数 | 说明 |
|------|------|------|------|
| GET | `/api/fixtures` | league_id, season, team_id, date, status, page, page_size | 分页比赛列表 |
| GET | `/api/fixtures/{fixture_id}` | - | 比赛详情（含 events/lineups/statistics/player_stats） |
| PATCH | `/api/fixtures/{fixture_id}/category` | category, - | 设置分类标签 |
| (private) | `get_fixture_form(fixture_id)` | - | 取球队近 10 场战绩 + H2H |

**时间转换**：`_date_to_utc_range(date_str)` 将北京时间日期（YYYY-MM-DD）转为 UTC 范围（当天 10:10 起始 → 次日 10:10）。

#### 5.7.6 predictions.py

| 方法 | 路径 | 参数 | 说明 |
|------|------|------|------|
| POST | `/api/predict/{fixture_id}` | - | 手动触发单场预测 |
| GET | `/api/predictions` | date, category, page, page_size | 分页预测结果查询 |

##### 预测查询返回结构

通过 raw SQL 多表 JOIN（predictions → fixtures → teams → leagues）一次性获取数据，并用 `COALESCE(ht.name_zh, p.home_name)` 做中文兜底。返回数据分 4 块：

```json
{
  "basic": {
    "fixture_id": "...",
    "home_name": "...", "away_name": "...",
    "league_name": "...", "match_date": "...",
    "status_short": "...", "category": "..."
  },
  "xgb": {
    "model_group": "PL",
    "prob": {"home": 0.65, "draw": 0.22, "away": 0.13},
    "over25": {"over": 0.55, "under": 0.45},
    "lambda": {"home": 1.8, "away": 1.2},
    "top3": [{"score": "2-1", "prob": 0.12}, {"score": "1-1", "prob": 0.10}, {"score": "3-1", "prob": 0.08}],
    "handicap": "主让0.5球"
  },
  "llm": {
    "win": "主胜", "win_pct": "85%",
    "score": "2-1,1-1,3-0",
    "handicap": "主队 -1", "handicap_num": -1, "handicap_team": "主队", "handicap_pct": "75%",
    "over_under": "大", "ou_line": "2.5", "ou_type": "大", "ou_pct": "60%",
    "brief": "主队优势明显", "core_data": "...", "deep_report": "..."
  },
  "result": {
    "score": "2-1", "win_correct": 1,
    "over25_correct": 1, "handicap_correct": null, "top3_correct": 1
  }
}
```

#### 5.7.7 scheduler.py

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/scheduler/status` | 返回所有任务状态 |
| POST | `/api/scheduler/{task_id}/trigger` | 手动触发任意任务（通用接口） |
| POST | `/api/scheduler/{task_id}/start` | 启动任务 |
| POST | `/api/scheduler/{task_id}/stop` | 停止任务 |
| PATCH | `/api/scheduler/{task_id}` | 修改任务配置（start_hour/interval_seconds/is_enabled） |
| GET | `/api/scheduler/logs` | 执行日志分页 |
| POST | `/api/scheduler/leagues/sync` | 兼容旧接口 |
| POST | `/api/scheduler/teams/sync` | 兼容旧接口 |
| POST | `/api/scheduler/players/sync` | 兼容旧接口 |
| POST | `/api/scheduler/standings/sync` | 兼容旧接口 |
| POST | `/api/scheduler/fixtures/sync` | 兼容旧接口 |
| POST | `/api/scheduler/fixtures/live` | 兼容旧接口 |

**TASK_MAP** 将字符串 task_id 映射到同步函数名：

```python
TASK_MAP = {
    "league_sync":   ("联赛数据同步", sync_leagues),
    "team_sync":     ("球队数据同步", sync_teams),
    ...
}
```

---

### 5.8 Prediction 预测推理层

#### 5.8.1 `predict.py` — 单场预测主流程

**文件**：[prediction/predict.py](backend/prediction/predict.py)

##### 入口函数 `predict_fixture(fixture_id: int, db=None) -> dict | None`

执行完整预测流程并写入 predictions 表，返回预测结果字典或 None（失败）。

**6 步流程**：

```
① 读取比赛信息 (fixtures 表)
    ↓
② 特征提取 + XGBoost 推理
    extract_features_for_fixture(db, fixture) → dict
    _load_best_models(league_id) → (models_dict, model_group)
    5 个模型推理：
      win_probs = models['win'].predict_proba(X)        → [P客胜, P平, P主胜]
      over25_prob = models['over25'].predict_proba(X)  → P大2.5
      lambda_home = models['lambda_home'].predict(X)    → 主队 Poisson λ
      lambda_away = models['lambda_away'].predict(X)    → 客队 Poisson λ
      _poisson_top3(λh, λa) → [{"score": "2-1", "prob": 0.12}, ...]
    ↓
③ 拉取赔率数据 (API-Football /odds)
    取近 7 天 top 5 庄家的 Match Winner 赔率变化
    计算隐含概率（去水）
    ↓
④ 拉取全量上下文
    _fetch_team_recent_via_api(home_id) → 近 10 场战绩+逐场明细
    _fetch_team_recent_via_api(away_id)
    _fetch_standings_text(db, team_id, league_id, season) → 积分榜文本
    _fetch_lineups_text(fixture_id, home_id, away_id) → 首发阵容/阵型
    ↓
⑤ 调用 LLM 综合分析
    _build_llm_prompt(...) → 拼装 prompt
    _call_llm(prompt) → DeepSeek Chat Completions
    _parse_llm_json(content) → 提取 JSON
    _has_required_llm_fields(parsed) → 校验必填字段
    ↓
⑥ 写入 predictions 表 (ON DUPLICATE KEY UPDATE)
    _save_prediction(db, fixture, xgb_result, llm_result, odds, model_group)
```

##### 关键辅助函数

| 函数 | 作用 |
|------|------|
| `_load_feature_cols()` | 从 `models/feature_cols.pkl` 加载推理需用的特征列名 |
| `_load_best_models(league_id)` | 三级回退：`L_{league_id}` → 赛事大组 → GLOBAL |
| `_poisson_top3(lambda_home, lambda_away, max_goals=8)` | Poisson 分布计算 8×8=64 种比分的联合概率，取 Top3 |
| `_call_llm(prompt)` | httpx POST → DeepSeek `/chat/completions`（temperature=0.3, max_tokens=800, timeout=30s） |
| `_parse_llm_json(content)` | 去除 ```json``` 包裹，提取第一个合法 JSON object |
| `_has_required_llm_fields(data)` | 校验 win/win_pct/score/handicap_num/handicap_team/handicap_pct/ou_line/ou_type/ou_pct 全部非空 |
| `_fetch_odds(fixture_id)` | 近 7 天 top 5 庄家赔率变化，隐含概率去水 |
| `_competition_context(fixture)` | 识别赛事类型（世界杯/预选赛/友谊赛/杯赛/联赛）并给 LLM 提示 |
| `_build_llm_prompt(...)` | 拼装完整 prompt（比赛信息、双方数据、赔率、XGB 参考、重要规则） |
| `_save_prediction(db, fixture, xgb, llm, odds, model_group)` | raw SQL `INSERT ... ON DUPLICATE KEY UPDATE` |

##### LLM Prompt 设计要点

Prompt 中明确告知 LLM：
- 先独立判断，再参考 XGB 概率做校准
- 世界杯/国家队若无确认首发，必须考虑轮换并降低置信度
- 赔率只代表市场热度，不代表真实胜率
- 模型概率只作参考，不能盲从
- 输出严格 JSON 格式，包含 12 个必填字段

#### 5.8.2 `features.py` — 特征工程

**文件**：[prediction/features.py](backend/prediction/features.py)

模块包含两套特征提取：批量版（训练用，`main()`）和单场版（推理用，`extract_features_for_fixture()`）。

##### 特征组概览

| 组 | 窗口 | 特征名示例 | 维度 |
|----|------|------------|------|
| A. 近期状态（全场） | 近 5 场 | h_win5, h_draw5, h_loss5, h_gf5, h_ga5, h_ht_gf5, h_pts5 | 8×2=16 |
| A. 近期状态（全场） | 近 10 场 | h_win10, h_gf10, h_ga10 | 3×2=6 |
| A. 近期状态（主/客） | 近 5 场 | h_home_win5, h_home_gf5, h_home_ga5 | 3×2=6 |
| A. 差值特征 | - | win_rate_diff, gf_diff5, pts_diff5 | 3 |
| B. 结果序列 | 近 5 场 | h_result_1~5, h_momentum | 6×2=12 |
| C. 比赛统计均值 | 近 5 场 | h_xg5, h_shots5, h_shots_on5, h_possession5, h_corners5, h_passes_pct5, h_fouls5 | 7×2=14 |
| C. 差值 | - | xg_diff5 | 1 |
| D. H2H 历史交锋 | 近 5 场 | h2h_home_win, h2h_draw, h2h_avg_goals, h2h_over25, h2h_n | 5 |
| E. 积分榜（当赛季） | - | h_rank, h_points, h_goals_diff, h_home_win_rate_s, h_home_gf_avg_s, h_home_ga_avg_s | 6×2=12 |
| E. 差值 | - | rank_diff, points_diff | 2 |
| F. 赛事背景 | - | is_cup | 1 |

**总特征数约 75+**（具体看 feature_cols.pkl）

##### 训练标签

| 标签 | 含义 | 类型 |
|------|------|------|
| label_win | 0=客胜, 1=平局, 2=主胜 | 三分类 |
| label_over25 | 1=大球(≥3), 0=小球 | 二分类 |
| label_goals_range | 0=0-1球, 1=2-3球, 2=4-5球, 3=6+球 | 四分类 |
| goals_home / goals_away | 实际进球数 | 回归目标（Poisson λ 预测） |

##### 批量特征提取 `main()`

一次性加载全部数据到内存，用 pandas 向量化计算，速度比单场循环提升 100x+：

1. `_load_all_data()` → fixtures, fixture_statistics, standings 三张大表
2. `_compute_team_form(fixtures, n, home_only/away_only)` → 滚动窗口均值（shift(1).rolling(n).mean()，min_periods=3）
3. `_compute_result_sequence(fixtures, n=5)` → 逐场结果序列 + momentum（近 3 场胜率 − 近 5 场胜率）
4. `_compute_match_stats_form(fixtures, stats_pivoted, n=5)` → 比赛统计滚动均值
5. `_pivot_stats(stats)` → 宽表化（fixture_id, team_id → 列=统计值）
6. `_compute_h2h(fixtures, n=5)` → H2H 特征（pair_key 去重）
7. 组装所有特征 + 标签 → `features.csv`

**过滤条件**：`df = df[df['h_n5'] >= 3]` — 至少需要近 3 场数据

##### 单场推理 `extract_features_for_fixture(db, fixture_row)`

给定 fixture 字典（含 id/home_id/away_id/league_id/season/date），实时从 DB 查询：

| 函数 | 查询 |
|------|------|
| `_form_stats_single(db, team_id, before_date, n)` | 近 n 场全场战绩（SQL 按 date DESC LIMIT） |
| `_form_stats_single(..., home_only=True)` | 近 n 场仅主场 |
| `_form_stats_single(..., away_only=True)` | 近 n 场仅客场 |
| `_match_stats_single(db, team_id, before_date, n)` | 近 n 场比赛统计均值 |
| `_result_sequence_single(db, team_id, before_date, n)` | 近 n 场结果序列 + momentum |
| `_h2h_single(db, home_id, away_id, before_date, n)` | H2H 近 n 场 |
| `_standings_single(db, team_id, league_id, season, side)` | 积分榜主/客队视角 |

**过滤条件**：`if hf5.get("n", 0) < 3 or af5.get("n", 0) < 3: return None` — 任一队近 5 场不足 3 场数据则放弃预测。

#### 5.8.3 `train.py` — 训练入口

**文件**：[prediction/train.py](backend/prediction/train.py)

`run()` 主流程：

```python
df = load_features()                              # 读 features.csv
feature_cols = get_feature_cols(df)               # 排除非特征列
pickle.dump(feature_cols, 'feature_cols.pkl')     # 保存列名（推理时用）

groups = get_groups(df)                           # 按联赛 ID 和 GROUP_MAP 分组

for group_key, group_df in groups.items():
    df_train, df_test = train_test_split_temporal(group_df)
    
    if len(df_train) < 100: continue              # 训练集太小跳过
    
    models = train_group(group_key, df_train, feature_cols)
    save_models(group_key, models)
    results = evaluate_group(models, X_test, df_test)
    
    # 写 metrics.csv
```

---

### 5.9 Training 模型训练层

#### 5.9.1 `data.py` — 数据加载与分组

**文件**：[prediction/training/data.py](backend/prediction/training/data.py)

##### 联赛分组 `GROUP_MAP`

```python
GROUP_MAP = {
    # 五大联赛独立
    39: 'PL', 140: 'LALIGA', 135: 'SERIEA', 78: 'BUNDESLIGA', 61: 'LIGUE1',
    # 欧洲二级联赛
    40: 'EU2', 41: 'EU2', 62: 'EU2', ...
    # 欧洲杯赛
    2: 'EU_CUP', 3: 'EU_CUP', 848: 'EU_CUP', 45: 'EU_CUP',
    # 美洲
    253: 'AMERICAS', 128: 'AMERICAS', ...
    # 亚洲/大洋洲
    98: 'ASIA', 99: 'ASIA', ...
    # 其他欧洲
    244: 'EU_OTHER', 666: 'EU_OTHER', 10: 'EU_OTHER',
}
```

##### `get_groups(df)` 返回三级分组

1. **联赛专属组**：`L_{league_id}` — 需要样本数 ≥ MIN_SAMPLES(300) 才会建
2. **赛事大组**：PL/LALIGA/SERIEA/BUNDESLIGA/LIGUE1/EU2/EU_CUP/AMERICAS/ASIA/EU_OTHER
3. **GLOBAL 全局组**：所有数据合并

##### `train_test_split_temporal(df, test_ratio=0.2)`

按时间顺序切分：前 80% 训练，后 20% 测试（保证时序不泄漏）。

##### `assign_group(league_id)` — 联赛 ID → 大组名

```python
def assign_group(league_id: int) -> str:
    return GROUP_MAP.get(league_id, 'GLOBAL')
```

#### 5.9.2 `model.py` — XGBoost 模型

**文件**：[prediction/training/model.py](backend/prediction/training/model.py)

##### 4 个模型定义

| 模型 | 类型 | XGBoost objective | 标签 |
|------|------|-------------------|------|
| win | 3 分类 | multi:softprob (num_class=3) | label_win (0/1/2) |
| over25 | 2 分类 | binary:logistic | label_over25 (0/1) |
| goals_range | 4 分类 | multi:softprob (num_class=4) | label_goals_range (0-3) |
| lambda_home / lambda_away | 回归 | reg:squarederror | goals_home / goals_away |

##### 超参数（所有模型共用）

| 参数 | 默认值 |
|------|--------|
| n_estimators | 300（回归 200） |
| learning_rate | 0.05 |
| max_depth | 5（分类 4，回归 4） |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| eval_metric | mlogloss / logloss |
| random_state | 42 |
| n_jobs | -1 |

##### 核心函数

| 函数 | 说明 |
|------|------|
| `_fill_na(X)` | 数值化 + 中位数填充 NaN（`pd.to_numeric + fillna(median)`） |
| `train_win_model / train_over25_model / train_goals_range_model / train_lambda_model` | 各模型单独训练 |
| `train_group(group_key, df_train, feature_cols)` | 训练一个分组的全部 5 个模型 |
| `save_models(group_key, models)` | pickle dump 到 `models/{group_key}_{name}.pkl` |
| `load_models(group_key)` | 加载 5 个模型文件，任一缺失返回 None |
| `load_any_model(group_key, feature_cols)` | 按优先级加载：联赛专属 → 大组 → GLOBAL |

#### 5.9.3 `evaluate.py` — 评估指标

**文件**：[prediction/training/evaluate.py](backend/prediction/training/evaluate.py)

| 函数 | 指标 | 公式 |
|------|------|------|
| `accuracy_3way` | 3 分类准确率 | argmax(probs) vs y_true |
| `rps_3way` | Ranked Probability Score（越低越好） | Σ (cum_pred − cum_true)² / (2N) |
| `evaluate_win` | 胜平负综合 | accuracy_3way + rps + logloss |
| `evaluate_over25` | 大小球综合 | accuracy + AUC (roc_auc_score) |
| `evaluate_goals_range` | 进球区间 | accuracy |

---

### 5.10 Tools 命令行工具层

#### 5.10.1 `manual_predict.py` — Rich 交互式单场预测

**文件**：[tools/manual_predict.py](backend/tools/manual_predict.py)

使用 Rich 库实现交互式 CLI：
1. 提示输入日期（默认今天）
2. 展示该日期所有启用联赛比赛的表格（序号、fixture_id、时间、联赛、状态、已有预测）
3. 用户输入序号或 fixture_id 执行预测
4. 渲染预测结果：Panel（标题）+ LLM 最终预测表 + XGBoost 参考表 + 分析摘要 Panel

**运行**：在 backend 目录下 `python tools/manual_predict.py`

#### 5.10.2 `batch_predict.py` — 批量预测

**文件**：[tools/batch_predict.py](backend/tools/batch_predict.py)

硬编码 `DATES = ['2026-06-11']`，遍历这些日期查询所有启用联赛的未预测比赛，逐个调用 `predict_fixture`，成功/失败计数输出。

#### 5.10.3 `sync_fixtures.py` — 完赛子数据补完

**文件**：[tools/sync_fixtures.py](backend/tools/sync_fixtures.py)

循环检测 `status_short in 完赛集合 AND sub_data_synced=False` 的数量，调用 `sync_completed_sub_data` 补完，直到没有剩余为止。每轮打印已运行分钟数和进度。

#### 5.10.4 `sync_players.py` — 单联赛球员同步

**文件**：[tools/sync_players.py](backend/tools/sync_players.py)

极简脚本，只同步联赛 848（世界杯）的球员。

#### 5.10.5 `async_sync_subdata.py` — asyncio 高并发子数据同步

**文件**：[tools/async_sync_subdata.py](backend/tools/async_sync_subdata.py)

与同步版 `fixture_service.sync_completed_sub_data` 功能相同，但使用 **asyncmy**（异步 MySQL 驱动）+ **httpx.AsyncClient** + **asyncio.Semaphore** 实现更高并发（CONCURRENT=10）。

核心特点：
- `asyncmy.create_pool()` 连接池
- `asyncio.gather()` 并行拉取 4 个 API 端点
- `cur.executemany()` 批量 INSERT
- 失败重试 3 次（退避 1s, 2s, 3s）
- BATCH_SIZE=200 分批处理

#### 5.10.6 `retry_fixtures.py` — 重试失败联赛

**文件**：[tools/retry_fixtures.py](backend/tools/retry_fixtures.py)

硬编码 `FAILED = [(141, 2024)]`，对这些 `(league_id, year)` 重新调用 API 拉取完整赛程并 upsert。

---

### 5.11 Alembic 数据库迁移

**配置文件**：[alembic.ini](backend/alembic.ini)（`sqlalchemy.url` 在 env.py 中动态设置）

**迁移环境**：[alembic/env.py](backend/alembic/env.py)（从 `app.core.config.settings` 读取 db_url）

**迁移版本数**：24 个（截至 2026-10-15）

**重要说明**（来自 main.py 注释）：
> 禁用了 alembic 自动迁移，因为 autogenerate 会删除没有 SQLAlchemy 模型的表（predictions、scheduler_tasks、scheduler_logs 使用 raw SQL 管理）。新增表请手动建表或手动编写迁移。

**迁移涉及的表**：

| 迁移 | 操作 |
|------|------|
| 74eeae5ca0dd | init（建基础表） |
| 7964cc3d3875 | add_players |
| 5fa77c6875be | add_standings |
| 76f99095e473 | add_standings |
| ba14b44197e8 | add_fixtures |
| c8599e8c469b | add_teams_venues |
| 7aa46f446ddd | add_predictions_table |
| 112dda7fa9e6 | create_users_table |
| ce0fa08af165 | add_fixture_category |
| 03400ae7cc60 | add_team_league_name_to_player_stats |
| 1ae0f8115dae | make_team_id_nullable_in_player_stats |
| b6d2a8c9f001 | drop_users_orders |
| 09335e945514 | extend_llm_score |
| 2e1e3ff28115 | add_llm_detail_splits |
| 662b02356fed | add_llm_detail_fields |
| 16f391a4464c | add_llm_score_drop_deprecated |
| 8990f718c8b8 | add_handicap_against_model_to_ |
| e01ca8690057 | add_handicap_correct |
| f1173d35c3c1 | add_scheduler_tasks |
| 4856638256ec | add_scheduler_logs |
| 2c9c5fbebfbd | add_league_enabled |
| 1334a9cd5b21 | drop_goals_range_correct |
| d8aea5d67d80 | drop_team_league_name_from_player_stats |
| aa8539407817 | add_orders_table |

---

## 6. 数据库表结构

### 6.1 核心业务表（SQLAlchemy ORM 管理）

| 表 | 主要字段 | 引擎 |
|----|----------|------|
| leagues | id, name, name_zh, type, country_name, country_name_zh, country_code, country_flag, enabled | ORM |
| seasons | id(自增), league_id(FK), year, start_date, end_date, is_current, coverage(JSON) | ORM |
| teams | id, name, name_zh, code, country, country_zh, founded, national, logo, venue_id(FK) | ORM |
| venues | id, name, name_zh, address, city, city_zh, country, capacity, surface, image | ORM |
| players | id, name, name_zh, firstname, lastname, age, nationality, height, weight, injured, photo, birth_date, birth_place, birth_country | ORM |
| player_stats | id(自增), player_id(FK), team_id, league_id, season, games/substitutes/shots/goals/passes/tackles/duels/dribbles/fouls/cards/penalty(JSON 块) | ORM |
| standings | id(自增), league_id, season, group_name, rank, team_id, team_name, team_logo, points, goals_diff, form, status, description + 全主客三维度 played/win/draw/lose/goals_for/goals_against | ORM |
| fixtures | id, date, timestamp, timezone, referee, venue_id, venue_name, venue_city, status_short, status_long, status_elapsed, status_extra, league_id, league_name, season, round, home_id/name/logo/winner, away_id/name/logo/winner, goals_home/away, halftime/fulltime/extratime/penalty 的 home/away, sub_data_synced, category | ORM |
| fixture_events | id(自增), fixture_id(FK), elapsed, extra, type, type_zh, detail, detail_zh, comments, team_id, team_name, player_id, player_name, assist_id, assist_name | ORM |
| fixture_lineups | id(自增), fixture_id(FK), team_id, team_name, formation, player_id, player_name, player_number, player_position, player_grid, is_substitute | ORM |
| fixture_statistics | id(自增), fixture_id(FK), team_id, team_name, stat_type, stat_type_zh, stat_value | ORM |
| fixture_player_stats | id(自增), fixture_id(FK), team_id, team_name, player_id, player_name, player_photo + games/offsides/shots/goals/passes/tackles/duels/dribbles/fouls/cards/penalty(JSON 块) | ORM |

### 6.2 Raw SQL 管理的表（ORM 不建模）

| 表 | 字段 | 说明 |
|----|------|------|
| predictions | fixture_id (PK), match_date, model_group, home_name/away_name/league_name/home_logo/away_logo, win_home/win_draw/win_away, over25_prob, top3_scores(JSON 字符串), lambda_home/lambda_away, handicap, llm_win, llm_win_pct, llm_score, llm_brief, llm_core_data, llm_deep_report, llm_handicap, llm_over_under, llm_handicap_num/team/pct, llm_ou_line/type/pct, actual_home_goals/actual_away_goals, win_correct, over25_correct, handicap_correct, score_in_top3, created_at, updated_at | 预测结果主表，ON DUPLICATE KEY UPDATE |
| scheduler_tasks | id (PK), name, interval_seconds, start_hour, is_enabled, last_run, next_run, is_running | 任务配置持久化 |
| scheduler_logs | id (PK), task_id, task_name, status(running/success/failed), message, started_at, finished_at | 任务执行日志 |

---

## 7. 核心数据流

### 7.1 数据同步流程

```
API-Football ──httpx──▶ Sync Function ──▶ MySQL
                                         │
                     ┌───────────────────┤
                     ▼                   ▼
               leagues              seasons
                     │
                     ▼
                   teams            venues
                     │
                     ▼
                  players          player_stats
                                    standings
                     │
                     ▼
                   fixtures ────────────┐
                     │                  │
                     ▼                  ▼
              (赛后检测完赛)      并发拉取 4 张子表：
                     │            ├── fixture_events
                     │            ├── fixture_lineups
                     │            ├── fixture_statistics
                     │            └── fixture_player_stats
                     ▼
              sub_data_synced = True
```

### 7.2 单场预测流程

```
fixture_id
    │
    ▼
┌─────────────────────┐
│ 读取 fixtures 表    │
└─────────┬───────────┘
          ▼
┌─────────────────────────────────────────┐
│ 特征提取                                 │
│ features.extract_features_for_fixture() │
│   ├── 近 5/10 场状态                     │
│   ├── 主客场分拆状态                     │
│   ├── 结果序列 + momentum                │
│   ├── 比赛统计均值                       │
│   ├── H2H 历史交锋                      │
│   ├── 积分榜（主/客视角）                │
│   └── 赛事背景                           │
└─────────┬───────────────────────────────┘
          ▼
┌─────────────────────────────────────────┐
│ 模型加载（三级回退）                     │
│ L_{league_id} → 赛事大组 → GLOBAL        │
└─────────┬───────────────────────────────┘
          ▼
┌─────────────────────────────────────────┐
│ XGBoost 推理                             │
│   win:     predict_proba → [P客, P平, P主]│
│   over25:  predict_proba → P(大2.5)      │
│   lambda_home/away: predict → Poisson λ │
│   Poisson: Top3 比分                     │
└─────────┬───────────────────────────────┘
          ▼
┌─────────────────────────────────────────┐
│ 全量上下文采集                           │
│   ├── API-Football /odds → 近 7 天赔率   │
│   ├── API-Football /fixtures?team= → 近 10 场│
│   ├── standings 表 → 积分榜文本          │
│   └── API-Football /lineups → 首发阵容   │
└─────────┬───────────────────────────────┘
          ▼
┌─────────────────────────────────────────┐
│ LLM 调用 (DeepSeek)                     │
│   prompt = 比赛信息 + 双方全量数据       │
│          + 赔率 + XGB 参考 + 规则约束    │
│   → 返回 JSON (12 字段)                  │
└─────────┬───────────────────────────────┘
          ▼
┌─────────────────────────────────────────┐
│ 写入 predictions 表 (ON DUPLICATE UPDATE)│
└─────────────────────────────────────────┘
```

### 7.3 模型训练流程

```
所有完赛 fixtures
    │
    ▼
┌────────────────────────────────────────────┐
│ features.py::main() — 批量特征提取          │
│                                             │
│ _load_all_data()                            │
│   ├── fixtures (完赛, 启用联赛)              │
│   ├── fixture_statistics                    │
│   └── standings                             │
│                                             │
│ pandas 向量化计算：                         │
│   ├── _compute_team_form (n=5, 10, 主, 客) │
│   ├── _compute_result_sequence (n=5)        │
│   ├── _compute_match_stats_form (n=5)       │
│   ├── _compute_h2h (n=5)                    │
│   └── 积分榜 + 差值 + is_cup               │
│                                             │
│ 输出 → features.csv                         │
└─────────────────┬──────────────────────────┘
                  ▼
┌────────────────────────────────────────────┐
│ train.py::run() — 训练入口                  │
│                                             │
│ data.py:                                    │
│   load_features() → 读 features.csv        │
│   get_feature_cols() → 排除非特征列         │
│   get_groups() → 三级分组                   │
│   train_test_split_temporal(test_ratio=0.2) │
│                                             │
│ model.py (对每个 group_key):                │
│   train_win_model      → XGBClassifier(3类) │
│   train_over25_model  → XGBClassifier(2类) │
│   train_goals_range   → XGBClassifier(4类) │
│   train_lambda_model  → 2×XGBRegressor     │
│   save_models() → models/{group_key}_*.pkl  │
│                                             │
│ evaluate.py:                                │
│   accuracy_3way + rps + logloss             │
│   accuracy + AUC                            │
│   → 写 metrics.csv                          │
└────────────────────────────────────────────┘
```

---

## 8. 依赖关系图

### 8.1 内部模块依赖

```
main.py
  ├── app.core.log_config
  ├── app.api.leagues  ──▶ app.services.league_service ──▶ app.core.{config, whitelist}
  ├── app.api.teams    ──▶ app.services.team_service   ──▶ app.core.config
  ├── app.api.players  ──▶ app.services.player_service ──▶ app.core.config
  ├── app.api.standings──▶ app.services.standing_service──▶ app.core.config
  ├── app.api.fixtures ──▶ app.services.fixture_service──▶ app.core.config
  │                                         └──▶ app.models.fixture (4 张子表)
  ├── app.api.scheduler──▶ app.services.scheduler ──▶ 全部 sync_* service
  │                                         └──▶ app.services.{auto_predict, prediction_result}
  └── app.api.predictions ──▶ prediction.predict ──▶ prediction.features
                                            └──▶ prediction.training.model
                                                         └──▶ prediction.training.data

prediction.predict
  ├── app.db.session
  ├── app.core.config
  ├── prediction.features
  │   └── app.db.session
  └── prediction.training.{model, data}

app.services.* (全部)
  ├── app.core.config
  ├── app.db.session
  └── httpx (同步 HTTP)

prediction.training.*
  └── 无 app 依赖（只依赖 pandas/numpy/xgboost/sklearn/scipy）
```

### 8.2 分层依赖规则

| 层级 | 依赖 | 被依赖 |
|------|------|--------|
| API | Service, DB, Schema, zh | main.py |
| Service | Model, DB, config | API, Scheduler, prediction.predict |
| Model | DB.base | Service, features.py |
| Prediction | features, training, DB, config | Services, API.predictions |
| Training | 无 app 依赖 | prediction.predict |
| Tools | Services, prediction, DB | 外部脚本 |

**禁止反向依赖**：Models 不依赖 Services；Training 不依赖任何 app 包。

---

## 9. 模型分组与回退机制

### 三级模型分组

```
联赛专属组 (L_{league_id})     ← 优先
    │ (样本 < 300 不存在)
    ▼
赛事大组 (PL/LALIGA/SERIEA/BUNDESLIGA/LIGUE1/EU2/EU_CUP/AMERICAS/ASIA/EU_OTHER)
    │
    ▼
GLOBAL (全部数据合并)          ← 兜底
```

### `_load_best_models(league_id)` 实现

```python
def _load_best_models(league_id: int):
    # 1. 尝试联赛专属模型
    league_key = f'L_{league_id}'
    m = load_models(league_key)
    if m:
        return m, league_key

    # 2. 尝试赛事大组
    group = assign_group(league_id)
    m = load_models(group)
    if m:
        return m, group

    # 3. 兜底 GLOBAL
    m = load_models('GLOBAL')
    if m:
        return m, 'GLOBAL'

    return None, None  # 完全无模型
```

### 分组样本门槛

- **MIN_SAMPLES = 300**：联赛专属组和大组都需要至少 300 个训练样本
- 样本不足的联赛（如某些杯赛）自动归入对应大组或 GLOBAL

### 每个模型组包含 5 个模型文件

```
models/{group_key}_win.pkl          # 胜平负 3 分类
models/{group_key}_over25.pkl       # 大小球 2 分类
models/{group_key}_goals_range.pkl  # 进球区间 4 分类
models/{group_key}_lambda_home.pkl  # 主队 Poisson λ 回归
models/{group_key}_lambda_away.pkl  # 客队 Poisson λ 回归
```

推理时任一文件缺失则 `load_models()` 返回 None。

---

## 10. 预测结果表结构详解

`predictions` 表使用 raw SQL 管理，不映射到 ORM。

### 写入 SQL（`predict.py::_save_prediction`）

```sql
INSERT INTO predictions (
    fixture_id, home_name, away_name, home_logo, away_logo,
    league_name, match_date, model_group,
    win_home, win_draw, win_away, over25_prob,
    top3_scores, lambda_home, lambda_away, handicap,
    llm_win, llm_score, llm_win_pct,
    llm_brief, llm_core_data, llm_deep_report,
    llm_handicap, llm_over_under,
    llm_handicap_num, llm_handicap_team, llm_handicap_pct,
    llm_ou_line, llm_ou_type, llm_ou_pct,
    created_at, updated_at
) VALUES (...)
ON DUPLICATE KEY UPDATE
    win_home=:wh, ..., updated_at=:now
```

### 字段来源

| 字段 | 来源 | 类型 |
|------|------|------|
| fixture_id (PK) | fixtures 表 | Integer |
| home_name/away_name/league_name/home_logo/away_logo | fixtures 表 | String |
| match_date | fixtures.date | DateTime |
| model_group | `_load_best_models` 返回值 | String（PL/GLOBAL/L_39 等） |
| win_home / win_draw / win_away | XGBoost win.predict_proba | Float |
| over25_prob | XGBoost over25.predict_proba[1] | Float |
| top3_scores | `_poisson_top3()` 返回的 list → JSON 字符串 | String(JSON) |
| lambda_home / lambda_away | XGBoost lambda_home/away.predict | Float |
| handicap | 让球参考（主让/客让 + 球数） | String |
| llm_win | LLM JSON.win | String（主胜/平局/客胜） |
| llm_win_pct | LLM JSON.win_pct | String（如 "85%"） |
| llm_score | LLM JSON.score | String（如 "2-1,1-1,3-0"） |
| llm_brief | LLM JSON.brief_analysis | String |
| llm_core_data | LLM JSON.core_data | String |
| llm_deep_report | LLM JSON.deep_report | String |
| llm_handicap | LLM 格式："-1.5 主队86%" | String |
| llm_over_under | LLM 格式："大球 2.5" | String |
| llm_handicap_num | LLM JSON.handicap_num（数值） | Float |
| llm_handicap_team | LLM JSON.handicap_team | String（主队/客队） |
| llm_handicap_pct | LLM JSON.handicap_pct | String |
| llm_ou_line | LLM JSON.ou_line | String（如 "2.5"） |
| llm_ou_type | LLM JSON.ou_type | String（大/小） |
| llm_ou_pct | LLM JSON.ou_pct | String |
| actual_home_goals / actual_away_goals | 赛后回填 | Integer |
| win_correct / over25_correct / handicap_correct / score_in_top3 | 赛后回填 | Integer (0/1) |

### 赛后回填逻辑（`prediction_result_service._backfill_one`）

1. 判断 `fixtures.status_short in {FT, AET, PEN, AWD, WO}` 且比分存在
2. 计算 `actual_win`（主胜/平局/客胜）对比 LLM 预测
3. 总进球 `total = gh + ga` 判断实际大/小球
4. 按 LLM 给出的让球数调整比分（`adjusted_home = gh + handicap_num`）判断盘口
5. 实际比分是否在 LLM.score 给出的逗号分隔列表中

---

## 11. 配置说明

### backend/.env（从 .env.example 复制）

```env
# MySQL
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=football

# API-Football 数据源
API_FOOTBALL_KEY=
API_FOOTBALL_BASE_URL=

# DeepSeek LLM
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# 缓存
CACHE_TTL=3600
```

> **重要**：`.env` 必须放在 `backend/` 目录下，因为 pydantic-settings 从当前运行目录读取配置。

### 配置加载链

```
backend/.env
    │
    ▼
app.core.config.Settings (pydantic-settings.BaseSettings)
    │
    ├─ db_url 动态拼接 → SQLAlchemy engine
    ├─ api_football_key / api_football_base_url → sync service
    └─ deepseek_api_key / deepseek_base_url / deepseek_model → predict.py::_call_llm
```

---

## 12. 运行方式

### 12.1 开发环境准备

```bash
# 进入 backend 目录
cd backend

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 12.2 启动 API 服务

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动后自动：
- 初始化 Loguru 日志
- 执行 `init_scheduler()`：种子化 8 个默认任务到 scheduler_tasks 表，加载所有 is_enabled=1 的任务并启动 asyncio loop

### 12.3 数据初始化

#### 方式一：已有 SQL 数据文件

```bash
mysql -u root -p football < dump-football-api-*.sql
```

#### 方式二：从空库开始

```bash
# 创建数据库
CREATE DATABASE football CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

# 迁移（注意：只迁移 ORM 管理的表）
alembic upgrade head

# 启动服务后按顺序触发同步
curl -X POST http://127.0.0.1:8000/api/scheduler/leagues/sync
curl -X POST http://127.0.0.1:8000/api/scheduler/teams/sync
curl -X POST http://127.0.0.1:8000/api/scheduler/standings/sync
curl -X POST http://127.0.0.1:8000/api/scheduler/fixtures/sync
# 球员同步较慢，可选
curl -X POST http://127.0.0.1:8000/api/scheduler/players/sync
```

### 12.4 重新训练模型

```bash
cd backend

# 1. 确保 features.csv 已生成（需要数据库中有足够的完赛数据）
python prediction/features.py

# 2. 训练所有分组模型
python prediction/train.py
```

训练完成后：
- `prediction/models/` 目录下生成各分组的 5 个 `.pkl` 文件 + `feature_cols.pkl`
- `prediction/metrics.csv` 输出每个分组的评估指标

### 12.5 CLI 工具使用

```bash
# 交互式单场预测
python tools/manual_predict.py

# 批量预测指定日期（硬编码在脚本内 DATES 列表）
python tools/batch_predict.py

# 补完赛子数据（断线重连友好）
python tools/sync_fixtures.py

# 异步高并发子数据同步（需要 asyncmy）
python tools/async_sync_subdata.py

# 单联赛球员同步
python tools/sync_players.py
```

---

## 13. 调度系统详解

### 任务状态持久化

所有任务的配置和运行状态持久化在 `scheduler_tasks` 表中，支持热更新：

| 字段 | 说明 |
|------|------|
| id | 任务 ID（league_sync, team_sync 等） |
| name | 显示名称 |
| start_hour | 定时任务：每天几点（2=02:00, 2.5=02:30） |
| interval_seconds | 间隔任务：每多少秒执行一次 |
| is_enabled | 是否启用（启动时加载） |
| last_run | 上次实际执行时间 |
| next_run | 计划下次执行时间 |
| is_running | 当前是否正在执行 |

### Loop 生命周期

```
init_scheduler()
  │
  ├── _seed_defaults()  → INSERT IGNORE 8 个默认任务
  │
  ├── 读 scheduler_tasks WHERE is_enabled=1
  │
  └── 对每个 enabled 任务调用 _start_loop(task_key)
        │
        └── asyncio.create_task(_run_loop(task_key))
              │
              └── while True:
                    读 DB 取配置 → 计算 delay
                    asyncio.sleep(delay)
                    run_task_with_log() → asyncio.to_thread()
                    更新 last_run / is_running / next_run
```

### Loop 间通信

所有状态变更（启停、修改配置）通过数据库 + 轮询实现：

```
API PATCH /scheduler/{task_id}
  → update_task() → UPDATE scheduler_tasks SET is_enabled=0/1 ...
  → 取消旧 asyncio.Task（如果在运行）
  → 如果启用：_start_loop() 新 asyncio.Task
       │
       └── _run_loop 中每次循环开头都会读 DB 检查 is_enabled
```

### 调度日志

`scheduler_logs` 表记录每次执行：

| 字段 | 说明 |
|------|------|
| id | 自增主键 |
| task_id / task_name | 任务标识 |
| status | running / success / failed |
| started_at / finished_at | 开始/结束时间 |
| message | 失败时的异常信息 |

查询：`GET /api/scheduler/logs?task_id=&page=&page_size=`

---

## 14. 中文字段兜底机制

### 设计原则

所有**源表**（直接存原始数据的表）都有 `*_zh` 列，翻译脚本（`translate_zh.py`）离线写入中文后，API 返回时自动用 `_zh` 覆盖原英文字段。

### 两层表的处理方式

#### 第一层：ORM 直接映射的表（League/Team/Player/Venue/FixtureEvent/FixtureStatistic）

调用 `zh_swap()` 做 in-place 对象属性替换：

```python
# League.name = "Premier League" → League.name_zh = "英超"
# zh_swap 后 League.name 变成 "英超"，League.name_zh 不变
```

映射表在 `ZH_MAPS` 中定义。

#### 第二层：反范式存储的表（fixtures/standings）

fixtures 表直接冗余存了字符串（league_name, home_name, away_name, venue_name），standings 表存了 team_name。API 查询时按 ID 批量回查源表覆盖：

```python
# fixtures_apply_denorm_zh(db, fixtures)
#   按 fixtures.home_id 批量查 teams.name_zh → 覆盖 fixtures.home_name
#   按 fixtures.league_id 批量查 leagues.name_zh → 覆盖 fixtures.league_name

# standings_apply_denorm_zh(db, standings)
#   按 standings.team_id 批量查 teams.name_zh → 覆盖 standings.team_name
```

### 单场预测中的中文兜底

`predictions` 表也存了 home_name/away_name/league_name（原始英文），API 查询时用 `COALESCE(ht.name_zh, p.home_name)` 多表 JOIN 兜底：

```sql
SELECT
  COALESCE(ht.name_zh, p.home_name) AS home_name,
  COALESCE(at.name_zh, p.away_name) AS away_name,
  COALESCE(lg.name_zh, p.league_name) AS league_name,
  ...
FROM predictions p
LEFT JOIN fixtures f ON p.fixture_id = f.id
LEFT JOIN teams ht ON f.home_id = ht.id
LEFT JOIN teams at ON f.away_id = at.id
LEFT JOIN leagues lg ON f.league_id = lg.id
```

---

## 15. 附录：API 端点汇总

### 基础查询

| 方法 | 路径 | 查询参数 | 说明 |
|------|------|----------|------|
| GET | `/api/health` | - | 健康检查 |
| GET | `/api/leagues` | page, page_size, search, enabled | 分页联赛列表 |
| GET | `/api/leagues/{league_id}` | - | 联赛详情（含 seasons） |
| PATCH | `/api/leagues/{league_id}/toggle` | - | 启用/停用联赛 |
| GET | `/api/teams` | page, page_size, search | 分页球队列表 |
| GET | `/api/teams/{team_id}` | - | 球队详情（含 venue） |
| GET | `/api/players` | page, page_size, search | 分页球员列表（含 statistics） |
| GET | `/api/players/{player_id}` | - | 球员详情 |
| GET | `/api/standings` | league_id(必填), season, page, page_size | 积分榜 |
| GET | `/api/fixtures` | league_id, season, team_id, date, status, page, page_size | 分页比赛列表 |
| GET | `/api/fixtures/{fixture_id}` | - | 比赛详情（含 4 张子表） |
| PATCH | `/api/fixtures/{fixture_id}/category` | category | 设置比赛分类 |

### 预测

| 方法 | 路径 | 查询参数 | 说明 |
|------|------|----------|------|
| POST | `/api/predict/{fixture_id}` | - | 手动触发单场预测 |
| GET | `/api/predictions` | date, category, page, page_size | 分页预测结果 |

### 调度管理

| 方法 | 路径 | 查询参数 | 说明 |
|------|------|----------|------|
| GET | `/api/scheduler/status` | - | 所有任务状态 |
| POST | `/api/scheduler/{task_id}/trigger` | - | 手动触发任意任务 |
| POST | `/api/scheduler/{task_id}/start` | - | 启动任务 |
| POST | `/api/scheduler/{task_id}/stop` | - | 停止任务 |
| PATCH | `/api/scheduler/{task_id}` | start_hour, interval_seconds, is_enabled | 修改任务配置 |
| GET | `/api/scheduler/logs` | page, page_size, task_id | 执行日志 |

### 调度手动同步（兼容旧接口）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/scheduler/leagues/sync` | 联赛同步 |
| POST | `/api/scheduler/teams/sync` | 球队同步 |
| POST | `/api/scheduler/players/sync` | 球员同步 |
| POST | `/api/scheduler/standings/sync` | 积分榜同步 |
| POST | `/api/scheduler/fixtures/sync` | 赛程同步 |
| POST | `/api/scheduler/fixtures/live` | 实时赛况同步 |

---

> 文档生成时间：2026-10-15 · 基于仓库当前代码

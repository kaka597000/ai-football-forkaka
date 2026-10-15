"""
初始化数据库：建表 + 种子数据
同时创建 SQLAlchemy ORM 管理的表和 raw SQL 管理的表（predictions, scheduler_*）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import text
from app.db.session import engine, SessionLocal
from app.db.base import Base
from app.models import (
    League, Season, Team, Venue, Player, PlayerStats, Standing,
    Fixture, FixtureEvent, FixtureLineup, FixtureStatistic, FixturePlayerStat,
)

# 为 SQLite 写的建表 SQL
SQLITE_PREDICTIONS_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    fixture_id INTEGER PRIMARY KEY,
    home_name TEXT,
    away_name TEXT,
    home_logo TEXT,
    away_logo TEXT,
    league_name TEXT,
    match_date DATETIME,
    model_group TEXT,
    win_home REAL,
    win_draw REAL,
    win_away REAL,
    over25_prob REAL,
    top3_scores TEXT,
    lambda_home REAL,
    lambda_away REAL,
    handicap TEXT,
    llm_win TEXT,
    llm_score TEXT,
    llm_win_pct TEXT,
    llm_brief TEXT,
    llm_core_data TEXT,
    llm_deep_report TEXT,
    llm_handicap TEXT,
    llm_over_under TEXT,
    llm_handicap_num REAL,
    llm_handicap_team TEXT,
    llm_handicap_pct TEXT,
    llm_ou_line TEXT,
    llm_ou_type TEXT,
    llm_ou_pct TEXT,
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    win_correct INTEGER,
    over25_correct INTEGER,
    handicap_correct INTEGER,
    score_in_top3 INTEGER,
    created_at DATETIME,
    updated_at DATETIME
);
"""

SQLITE_SCHEDULER_TASKS_SQL = """
CREATE TABLE IF NOT EXISTS scheduler_tasks (
    id TEXT PRIMARY KEY,
    name TEXT,
    interval_seconds INTEGER,
    start_hour REAL,
    is_enabled INTEGER DEFAULT 1,
    last_run DATETIME,
    next_run DATETIME,
    is_running INTEGER DEFAULT 0
);
"""

SQLITE_SCHEDULER_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS scheduler_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    task_name TEXT,
    status TEXT,
    message TEXT,
    started_at DATETIME,
    finished_at DATETIME
);
"""

# MySQL 版本（生产用）
MYSQL_PREDICTIONS_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    fixture_id INT PRIMARY KEY,
    home_name VARCHAR(255), away_name VARCHAR(255),
    home_logo VARCHAR(500), away_logo VARCHAR(500),
    league_name VARCHAR(255), match_date DATETIME,
    model_group VARCHAR(50),
    win_home FLOAT, win_draw FLOAT, win_away FLOAT,
    over25_prob FLOAT,
    top3_scores TEXT,
    lambda_home FLOAT, lambda_away FLOAT,
    handicap VARCHAR(50),
    llm_win VARCHAR(20), llm_score VARCHAR(255), llm_win_pct VARCHAR(10),
    llm_brief VARCHAR(255), llm_core_data TEXT, llm_deep_report TEXT,
    llm_handicap VARCHAR(50), llm_over_under VARCHAR(20),
    llm_handicap_num FLOAT, llm_handicap_team VARCHAR(20), llm_handicap_pct VARCHAR(10),
    llm_ou_line VARCHAR(10), llm_ou_type VARCHAR(10), llm_ou_pct VARCHAR(10),
    actual_home_goals INT, actual_away_goals INT,
    win_correct TINYINT, over25_correct TINYINT, handicap_correct TINYINT, score_in_top3 TINYINT,
    created_at DATETIME, updated_at DATETIME
);
"""

MYSQL_SCHEDULER_TASKS_SQL = """
CREATE TABLE IF NOT EXISTS scheduler_tasks (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    interval_seconds INT,
    start_hour FLOAT,
    is_enabled TINYINT DEFAULT 1,
    last_run DATETIME,
    next_run DATETIME,
    is_running TINYINT DEFAULT 0
);
"""

MYSQL_SCHEDULER_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS scheduler_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(50), task_name VARCHAR(100),
    status VARCHAR(20), message TEXT,
    started_at DATETIME, finished_at DATETIME
);
"""


def init_db():
    # 1. SQLAlchemy 管理的表
    Base.metadata.create_all(bind=engine)
    print("✅ SQLAlchemy 模型表已创建")

    # 2. raw SQL 管理的表
    is_sqlite = str(engine.url).startswith("sqlite")
    if is_sqlite:
        preds_sql = SQLITE_PREDICTIONS_SQL
        tasks_sql = SQLITE_SCHEDULER_TASKS_SQL
        logs_sql = SQLITE_SCHEDULER_LOGS_SQL
        print("🗄️  后端: SQLite")
    else:
        preds_sql = MYSQL_PREDICTIONS_SQL
        tasks_sql = MYSQL_SCHEDULER_TASKS_SQL
        logs_sql = MYSQL_SCHEDULER_LOGS_SQL
        print("🗄️  后端: MySQL")

    with engine.connect() as conn:
        conn.execute(text(preds_sql))
        conn.execute(text(tasks_sql))
        conn.execute(text(logs_sql))
        conn.commit()
    print("✅ predictions / scheduler_* 表已创建")

    # 3. 简单种子数据：一条启用联赛 + 一场比赛
    db = SessionLocal()
    try:
        from datetime import date, datetime

        # 检查是否已经有数据
        if db.query(League).count() == 0:
            league = League(
                id=39,
                name="Premier League",
                name_zh="英超",
                type="League",
                country_name="England",
                country_name_zh="英格兰",
                country_code="GB",
                country_flag="",
                enabled=True,
            )
            db.add(league)

            season = Season(
                league_id=39,
                year=2025,
                start_date=date(2025, 8, 1),
                end_date=date(2026, 5, 31),
                is_current=True,
                coverage={"fixtures": True, "standings": True, "players": True},
            )
            db.add(season)

            # 球队
            team_ar = Team(id=42, name="Arsenal", name_zh="阿森纳", country="England", national=False)
            team_li = Team(id=50, name="Liverpool", name_zh="利物浦", country="England", national=False)
            db.add_all([team_ar, team_li])

            # 积分榜
            from app.models.standing import Standing
            db.add(Standing(
                league_id=39, season=2025, group_name=None, rank=1,
                team_id=42, team_name="Arsenal", team_logo=None,
                points=45, goals_diff=22, form="WWWDW",
                all_played=20, all_win=14, all_draw=3, all_lose=3,
                all_goals_for=40, all_goals_against=18,
                home_played=10, home_win=7, home_draw=2, home_lose=1,
                home_goals_for=22, home_goals_against=10,
                away_played=10, away_win=7, away_draw=1, away_lose=2,
                away_goals_for=18, away_goals_against=8,
            ))
            db.add(Standing(
                league_id=39, season=2025, group_name=None, rank=2,
                team_id=50, team_name="Liverpool", team_logo=None,
                points=44, goals_diff=20, form="WWDLW",
                all_played=20, all_win=13, all_draw=5, all_lose=2,
                all_goals_for=38, all_goals_against=18,
                home_played=10, home_win=8, home_draw=1, home_lose=1,
                home_goals_for=24, home_goals_against=8,
                away_played=10, away_win=5, away_draw=4, away_lose=1,
                away_goals_for=14, away_goals_against=10,
            ))

            # 之前的完赛记录（用于特征提取）
            import datetime as dt
            import random
            base = dt.datetime(2025, 8, 1)
            for i in range(10):
                for home_id, away_id in [(42, 50), (42, 0), (50, 0)]:
                    if home_id != 0 and away_id != 0:
                        fi = Fixture(
                            id=1000 + i * 3 + (0 if home_id == 42 else 1),
                            league_id=39, league_name="Premier League", season=2025,
                            home_id=home_id, home_name="Arsenal" if home_id == 42 else "Liverpool",
                            away_id=away_id, away_name="Liverpool" if away_id == 50 else "Chelsea",
                            home_logo=None, away_logo=None,
                            date=base + dt.timedelta(days=i * 7),
                            status_short="FT", status_long="Finished",
                            goals_home=random.randint(0, 3), goals_away=random.randint(0, 3),
                            halftime_home=random.randint(0, 1), halftime_away=random.randint(0, 1),
                        )
                        db.add(fi)

            # 一场今天的待开赛比赛
            today = datetime.now()
            db.add(Fixture(
                id=99999,
                league_id=39, league_name="Premier League", season=2025,
                home_id=42, home_name="Arsenal", home_logo=None,
                away_id=50, away_name="Liverpool", away_logo=None,
                date=today + dt.timedelta(hours=3),
                status_short="NS", status_long="Not Started",
            ))

            db.commit()
            print("✅ 种子数据已插入：联赛、2 支球队、积分榜、历史比赛、1 场待开赛比赛")
        else:
            print(f"ℹ️  已有 {db.query(League).count()} 条联赛数据，跳过种子")

    finally:
        db.close()

    print("\n🎉 数据库初始化完成！")


if __name__ == "__main__":
    init_db()

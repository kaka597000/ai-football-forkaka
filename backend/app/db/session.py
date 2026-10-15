from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.db.base import Base

# SQLite 特殊配置：需要 check_same_thread=False，连接池配置不同
if settings.db_url.startswith("sqlite"):
    engine = create_engine(
        settings.db_url,
        connect_args={"check_same_thread": False},
        echo=False,
    )
else:
    engine = create_engine(
        settings.db_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI 依赖: 获取数据库 session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings

_url = settings.database_url
if _url.startswith("postgres://"):  # some hosts still hand out the legacy scheme
    _url = _url.replace("postgres://", "postgresql://", 1)

if _url.startswith("sqlite"):
    kwargs = {"connect_args": {"check_same_thread": False}}
    if ":memory:" in _url:
        kwargs["poolclass"] = StaticPool
    engine = create_engine(_url, **kwargs)
else:
    engine = create_engine(_url, pool_pre_ping=True, pool_size=5, max_overflow=10, pool_recycle=1800)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

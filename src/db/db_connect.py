from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

load_dotenv()

DBRole = Literal["source", "datamart"]


@dataclass(frozen=True)
class DatabaseSettings:
    role: DBRole
    user: str | None
    password: str | None
    host: str | None
    port: str
    name: str | None

    @property
    def is_configured(self) -> bool:
        return all([self.user, self.host, self.name])

    @property
    def url(self) -> str:
        return URL.create(
            drivername="mysql+pymysql",
            username=self.user,
            password=self.password or None,
            host=self.host,
            port=int(self.port),
            database=self.name,
        ).render_as_string(hide_password=False)


def _env(
    primary: str,
    *,
    fallback_keys: tuple[str, ...] = (),
    default: str | None = None,
) -> str | None:
    for key in (primary, *fallback_keys):
        value = os.getenv(key)
        if value is not None:
            return value
    return default


def get_database_settings(role: DBRole = "source") -> DatabaseSettings:
    if role == "source":
        return DatabaseSettings(
            role="source",
            user=_env("SOURCE_DB_USER", fallback_keys=("DB_USER",)),
            password=_env("SOURCE_DB_PASSWORD", fallback_keys=("DB_PASSWORD",)),
            host=_env("SOURCE_DB_HOST", fallback_keys=("DB_HOST",)),
            port=_env("SOURCE_DB_PORT", fallback_keys=("DB_PORT",), default="3306") or "3306",
            name=_env("SOURCE_DB_NAME", fallback_keys=("DB_NAME",)),
        )

    source = get_database_settings("source")
    return DatabaseSettings(
        role="datamart",
        user=_env("DATAMART_DB_USER", fallback_keys=("SOURCE_DB_USER", "DB_USER"), default=source.user),
        password=_env(
            "DATAMART_DB_PASSWORD",
            fallback_keys=("SOURCE_DB_PASSWORD", "DB_PASSWORD"),
            default=source.password,
        ),
        host=_env("DATAMART_DB_HOST", fallback_keys=("SOURCE_DB_HOST", "DB_HOST"), default=source.host),
        port=_env(
            "DATAMART_DB_PORT",
            fallback_keys=("SOURCE_DB_PORT", "DB_PORT"),
            default=source.port,
        )
        or "3306",
        name=_env("DATAMART_DB_NAME", default="analytics"),
    )


@lru_cache(maxsize=2)
def get_db_engine(role: DBRole = "source") -> Engine:
    settings = get_database_settings(role)
    if not settings.is_configured:
        missing = [
            key
            for key, value in {
                "user": settings.user,
                "host": settings.host,
                "database": settings.name,
            }.items()
            if not value
        ]
        raise ValueError(
            f"{role} DB configuration is incomplete. Missing: {', '.join(missing)}"
        )
    return create_engine(settings.url, pool_pre_ping=True)


def get_source_db_engine() -> Engine:
    return get_db_engine("source")


def get_datamart_db_engine() -> Engine:
    return get_db_engine("datamart")


def get_table_samples(engine: Engine, table_name: str, sample_count: int = 10) -> list[dict]:
    try:
        with engine.connect() as connection:
            query = text(f"SELECT * FROM `{table_name}` ORDER BY RAND() LIMIT {sample_count}")
            df = pd.read_sql(query, connection)
            return df.to_dict(orient="records")
    except Exception as exc:
        print(f"⚠️ {table_name} 샘플 추출 실패: {exc}")
        return []


if __name__ == "__main__":
    for role in ("source", "datamart"):
        try:
            engine = get_db_engine(role)
            with engine.connect():
                print(f"✅ {role} DB 연결 성공: {get_database_settings(role).name}")
        except Exception as exc:
            print(f"❌ {role} DB 연결 실패: {exc}")

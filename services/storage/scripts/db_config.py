"""Database connection settings shared by storage scripts.

Values can be supplied explicitly, via CLI flags, or through environment
variables. Explicit arguments win over the environment; the environment wins
over the defaults. GitHub Actions repository variables flow in as environment
variables.
"""
from __future__ import annotations
import argparse
import os
from dataclasses import dataclass

import pymysql


@dataclass(frozen=True)
class DBConfig:
    host: str
    user: str
    password: str
    port: int
    db_name: str

    @classmethod
    def from_values(
        cls,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int | str | None = None,
        db_name: str | None = None,
    ) -> "DBConfig":
        return cls(
            host=host if host is not None else os.getenv("DB_HOST"),
            user=user if user is not None else os.getenv("DB_USER"),
            password=password if password is not None else os.getenv("DB_PASSWORD"),
            port=int(port if port is not None else os.getenv("DB_PORT")),
            db_name=db_name if db_name is not None else os.getenv("DB_NAME"),
        )

    def connect(self, include_database: bool = True, autocommit: bool = True):
        kwargs = {
            "host": self.host,
            "user": self.user,
            "password": self.password,
            "port": self.port,
            "autocommit": autocommit,
            "cursorclass": pymysql.cursors.DictCursor,
        }
        if include_database:
            kwargs["database"] = self.db_name
        return pymysql.connect(**kwargs)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--host", default=None, help="MySQL host (env: DB_HOST)")
    parser.add_argument("--user", default=None, help="MySQL user (env: DB_USER)")
    parser.add_argument("--password", default=None, help="MySQL password (env: DB_PASSWORD)")
    parser.add_argument("--port", default=None, help="MySQL port (env: DB_PORT)")
    parser.add_argument("--db-name", dest="db_name", default=None,
                        help="MySQL database (env: DB_NAME)")
    return parser


def config_from_namespace(args: argparse.Namespace) -> DBConfig:
    return DBConfig.from_values(
        host=args.host,
        user=args.user,
        password=args.password,
        port=args.port,
        db_name=args.db_name,
    )

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import tempfile

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
  id TEXT PRIMARY KEY, day TEXT UNIQUE NOT NULL, count INTEGER NOT NULL, plan TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'planned', route TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS works (
  id TEXT PRIMARY KEY, batch TEXT NOT NULL REFERENCES batches(id), slot INTEGER NOT NULL,
  data TEXT NOT NULL, sha TEXT UNIQUE NOT NULL, phash TEXT NOT NULL,
  UNIQUE(batch, slot)
);
CREATE TABLE IF NOT EXISTS schedules (
  day TEXT PRIMARY KEY, batch TEXT, mode TEXT NOT NULL, state TEXT NOT NULL,
  job TEXT, deadline TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS schedule_fallbacks (
  day TEXT PRIMARY KEY, batch TEXT NOT NULL, first_job TEXT NOT NULL,
  first_schedule TEXT NOT NULL, accounting TEXT NOT NULL,
  second_job TEXT, deadline TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deliveries (
  batch TEXT PRIMARY KEY REFERENCES batches(id), medium TEXT NOT NULL, recipient TEXT NOT NULL,
  approver TEXT NOT NULL, token TEXT NOT NULL UNIQUE, message TEXT NOT NULL UNIQUE,
  thread TEXT UNIQUE, provider_id TEXT, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
  sent_at TEXT, decision TEXT
);
CREATE TABLE IF NOT EXISTS replies (
  id TEXT PRIMARY KEY, batch TEXT NOT NULL REFERENCES batches(id), result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
  work TEXT PRIMARY KEY REFERENCES works(id), reply TEXT NOT NULL REFERENCES replies(id),
  chosen_at TEXT NOT NULL, snapshot TEXT NOT NULL
);
"""


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic(path: Path, contents: bytes, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".curator-", delete=False) as temp:
            tmp_name = temp.name
            os.fchmod(temp.fileno(), mode)
            temp.write(contents)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(tmp_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)


def atomic_json(path, value, mode=0o600):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(), mode)


class Busy(RuntimeError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.db = self.root / "curator.sqlite3"
        with self.connect() as db:
            db.executescript(SCHEMA)
        self.db.chmod(0o600)

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.db, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def lock(self, name):
        with (self.root / f"{name}.lock").open("a") as mutex:
            try:
                fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise Busy(f"{name} is already running.") from error
            yield

    def batch(self, day=None):
        with self.connect() as db:
            if day:
                return db.execute("SELECT * FROM batches WHERE day=?", (day,)).fetchone()
            return db.execute("""
                SELECT * FROM batches WHERE state IN ('ready','partial') ORDER BY day DESC LIMIT 1
            """).fetchone()

    def artworks(self, batch):
        with self.connect() as db:
            return [json.loads(row["data"]) for row in db.execute(
                "SELECT data FROM works WHERE batch=? ORDER BY slot", (batch,)
            )]

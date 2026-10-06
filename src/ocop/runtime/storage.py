import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from filelock import FileLock

from ocop.runtime.config import canonical_json
from ocop.graph import contract_hash


class StoreConflict(ValueError):
    pass


class RunHalted(StoreConflict):
    pass


class ReadStore:
    def __init__(self, path: Path):
        self.path = path
        with read_database(path) as db:
            self.tables = {name: [dict(row) for row in db.execute(f"SELECT * FROM {name} ORDER BY rowid")]
                           for name in ("run", "records", "links", "requests", "attempts")}
        self.run_id = self.tables["run"][0]["id"]
        self.archive = json.loads(self.read_blob(self.tables["run"][0]["manifest_blob"]))
        self.by_kind = {}
        self.links = {}
        for row in self.tables["records"]:
            self.by_kind.setdefault(row["kind"], {})[row["logical_key"]] = {
                "id": row["id"], "key": row["logical_key"], "payload": json.loads(self.read_blob(row["payload_blob"]))}
        for row in self.tables["links"]:
            self.links.setdefault(row["child_id"], {})[row["relation"]] = row["parent_id"]
        self.request_owners = {}
        self.request_attempts = {}
        for row in self.tables["requests"]:
            self.request_owners.setdefault(row["owner_id"], []).append(row)
        for row in self.tables["attempts"]:
            self.request_attempts.setdefault(row["request_id"], []).append(row)

    def read_blob(self, name):
        data = (self.path / "blobs" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != name:
            raise StoreConflict("Blob checksum mismatch")
        return data

    def rows(self, name):
        return self.tables[name]

    def record_items(self, kind):
        return list(self.by_kind.get(kind, {}).values())

    def get_record(self, kind, key):
        row = self.by_kind.get(kind, {}).get(key)
        return row["payload"] if row else None

    def attempts_for(self, request_id):
        return self.request_attempts.get(request_id, [])

    def requests_for_owner(self, owner_id):
        return self.request_owners.get(owner_id, [])


SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE run (
    id TEXT PRIMARY KEY, config_hash TEXT NOT NULL, manifest_blob TEXT NOT NULL,
    created REAL NOT NULL, failure_streak INTEGER NOT NULL DEFAULT 0, halt_reason TEXT
);
CREATE TABLE records (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, logical_key TEXT NOT NULL,
    payload_blob TEXT NOT NULL, created REAL NOT NULL, UNIQUE(kind, logical_key)
);
CREATE TABLE links (
    child_id TEXT NOT NULL REFERENCES records(id), relation TEXT NOT NULL,
    parent_id TEXT NOT NULL REFERENCES records(id), PRIMARY KEY(child_id, relation)
);
CREATE TABLE requests (
    id TEXT PRIMARY KEY, logical_key TEXT UNIQUE NOT NULL,
    owner_id TEXT REFERENCES records(id), spec_blob TEXT NOT NULL,
    state TEXT NOT NULL, result_json TEXT, created REAL NOT NULL
);
CREATE TABLE attempts (
    id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES requests(id),
    number INTEGER NOT NULL, state TEXT NOT NULL,
    started REAL NOT NULL, finished REAL, elapsed REAL,
    http_status INTEGER, headers_json TEXT, body_blob TEXT,
    result_json TEXT, UNIQUE(request_id, number)
);
PRAGMA user_version=1;
COMMIT;
"""


class RunStore:
    def __init__(self, root: Path, config: dict, *, run_id: str | None = None, provenance: dict | None = None):
        self.run_id = uuid.uuid4().hex if run_id is None else run_id
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", self.run_id):
            raise ValueError("Invalid run ID")
        self.path = root.resolve() / self.run_id
        self.config = {"config": config, "contract_hash": contract_hash(), "record_version": 1}
        self.provenance = provenance or {}
        self._mutex = threading.RLock()

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(self.path / "writer.lock", timeout=0)
        self._lock.acquire()
        try:
            (self.path / "blobs").mkdir(exist_ok=True)
            self.db = sqlite3.connect(self.path / "records.sqlite3", isolation_level=None, check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                self.db.executescript(SCHEMA)
            elif version != 1:
                raise StoreConflict(f"Unsupported store version: {version}")
            digest = hashlib.sha256(canonical_json(self.config)).hexdigest()
            with self.transaction():
                row = self.db.execute("SELECT * FROM run").fetchone()
                if row is None:
                    manifest = self.put_blob(canonical_json({**self.config, "provenance": self.provenance}))
                    self.db.execute("INSERT INTO run(id,config_hash,manifest_blob,created) VALUES(?,?,?,?)",
                                    (self.run_id, digest, manifest, time.time()))
                elif row["config_hash"] != digest:
                    raise StoreConflict("Run configuration or graph contract changed; use a new run ID")
                self.db.execute("UPDATE attempts SET state='uncertain', finished=? WHERE state='in_flight'", (time.time(),))
                self.db.execute("UPDATE requests SET state='pending' WHERE state='in_flight'")
            self.put_record("session", uuid.uuid4().hex, self.provenance)
            return self
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self._lock.release()
            raise

    def __exit__(self, *args):
        self.db.close()
        self._lock.release()

    @contextmanager
    def transaction(self):
        with self._mutex:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def put_blob(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        destination = self.path / "blobs" / digest
        if not destination.exists():
            temporary = destination.with_name(f".{uuid.uuid4().hex}.tmp")
            with temporary.open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return digest

    def read_blob(self, digest: str) -> bytes:
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid blob hash")
        data = (self.path / "blobs" / digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise StoreConflict("Blob checksum mismatch")
        return data

    def rows(self, table: str) -> list[dict]:
        if table not in {"run", "records", "links", "requests", "attempts"}:
            raise ValueError("Unknown table")
        with self._mutex:
            return [dict(row) for row in self.db.execute(f"SELECT * FROM {table} ORDER BY rowid")]

    def put_record(self, kind: str, key: str, payload: Any, *, parents: dict[str, str] | None = None) -> str:
        parents = parents or {}
        blob = self.put_blob(canonical_json(payload))
        with self.transaction():
            existing = self.db.execute("SELECT * FROM records WHERE kind=? AND logical_key=?", (kind, key)).fetchone()
            if existing is not None:
                links = dict(self.db.execute("SELECT relation,parent_id FROM links WHERE child_id=?", (existing["id"],)))
                if existing["payload_blob"] != blob or links != parents:
                    raise StoreConflict("An immutable record already exists with different content or parents")
                return existing["id"]
            record_id = uuid.uuid4().hex
            self.db.execute("INSERT INTO records VALUES(?,?,?,?,?)", (record_id, kind, key, blob, time.time()))
            for relation, parent in parents.items():
                self.db.execute("INSERT INTO links VALUES(?,?,?)", (record_id, relation, parent))
            return record_id

    def get_record(self, kind: str, key: str) -> Any | None:
        with self._mutex:
            row = self.db.execute("SELECT payload_blob FROM records WHERE kind=? AND logical_key=?", (kind, key)).fetchone()
            return None if row is None else json.loads(self.read_blob(row["payload_blob"]))

    def recover_circuit(self, key: str, payload: dict):
        blob = self.put_blob(canonical_json(payload))
        with self.transaction():
            run = self.db.execute("SELECT halt_reason FROM run").fetchone()
            if run["halt_reason"] != "consecutive_request_failures":
                raise StoreConflict("Circuit recovery requires a consecutive request failure halt")
            self.db.execute("INSERT INTO records VALUES(?,?,?,?,?)",
                            (uuid.uuid4().hex, "recovery", key, blob, time.time()))
            self.db.execute("UPDATE run SET halt_reason=NULL, failure_streak=0")

    def record_items(self, kind: str) -> list[dict]:
        with self._mutex:
            rows = self.db.execute("SELECT * FROM records WHERE kind=? ORDER BY rowid", (kind,)).fetchall()
            return [{"id": row["id"], "key": row["logical_key"], "payload": json.loads(self.read_blob(row["payload_blob"]))} for row in rows]

    def requests_for_owner(self, owner_id: str) -> list[dict]:
        with self._mutex:
            return [dict(row) for row in self.db.execute("SELECT * FROM requests WHERE owner_id=? ORDER BY rowid", (owner_id,))]

    def halt(self, reason: str):
        with self.transaction():
            self.db.execute("UPDATE run SET halt_reason=COALESCE(halt_reason, ?)", (reason,))

    def request(self, key: str, spec: dict, owner_id: str | None) -> dict:
        blob = self.put_blob(canonical_json(spec))
        with self.transaction():
            row = self.db.execute("SELECT * FROM requests WHERE logical_key=?", (key,)).fetchone()
            if row is not None:
                if row["spec_blob"] != blob or row["owner_id"] != owner_id:
                    raise StoreConflict("A logical request cannot change inputs or owner")
                return dict(row)
            request_id = uuid.uuid4().hex
            self.db.execute("INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
                            (request_id, key, owner_id, blob, "pending", None, time.time()))
            return dict(self.db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone())

    def attempts_for(self, request_id: str) -> list[dict]:
        with self._mutex:
            return [dict(row) for row in self.db.execute("SELECT * FROM attempts WHERE request_id=? ORDER BY number", (request_id,))]

    def start_attempt(self, request_id: str) -> dict:
        with self.transaction():
            request = self.db.execute("SELECT state FROM requests WHERE id=?", (request_id,)).fetchone()
            if request is None or request["state"] != "pending":
                raise StoreConflict("Request is not pending")
            if self.db.execute("SELECT 1 FROM attempts WHERE request_id=? AND state='response_saved'", (request_id,)).fetchone():
                raise StoreConflict("The saved response must be finalized before another attempt")
            number = self.db.execute("SELECT COUNT(*) FROM attempts WHERE request_id=?", (request_id,)).fetchone()[0] + 1
            attempt_id = uuid.uuid4().hex
            self.db.execute("INSERT INTO attempts(id,request_id,number,state,started) VALUES(?,?,?,'in_flight',?)",
                            (attempt_id, request_id, number, time.time()))
            self.db.execute("UPDATE requests SET state='in_flight' WHERE id=?", (request_id,))
            return dict(self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone())

    def save_response(self, attempt_id: str, *, body: bytes, status: int, headers: dict, elapsed: float, redacted: bool = False):
        blob = self.put_blob(body)
        with self.transaction():
            cursor = self.db.execute(
                "UPDATE attempts SET state='response_saved',body_blob=?,http_status=?,headers_json=?,elapsed=?,finished=?,result_json=? WHERE id=? AND state='in_flight'",
                (blob, status, json.dumps(headers), elapsed, time.time(), json.dumps({"body_redacted": redacted}), attempt_id),
            )
            if cursor.rowcount != 1:
                raise StoreConflict("Attempt is not in flight")

    def finish_attempt(self, attempt: dict, outcome: dict, *, terminal: bool, failure_limit: int):
        encoded = json.dumps(outcome, ensure_ascii=False, allow_nan=False)
        with self.transaction():
            cursor = self.db.execute(
                "UPDATE attempts SET state=?,result_json=?,elapsed=?,finished=? WHERE id=? AND state IN ('in_flight','response_saved')",
                (outcome["status"], encoded, outcome["elapsed_seconds"], time.time(), attempt["id"]),
            )
            if cursor.rowcount != 1:
                raise StoreConflict("Attempt is already finalized")
            self.db.execute("UPDATE requests SET state=?,result_json=? WHERE id=?",
                            (outcome["status"] if terminal else "pending", encoded if terminal else None, attempt["request_id"]))
            if terminal:
                self._update_circuit(outcome["status"], failure_limit)

    def exhaust_uncertain(self, request_id: str, limit: int) -> dict:
        outcome = {"status": "infra_failed", "error": "attempt_budget_exhausted_after_interruption", "request_id": request_id, "usage": None}
        with self.transaction():
            self.db.execute("UPDATE requests SET state='infra_failed',result_json=? WHERE id=?", (json.dumps(outcome), request_id))
            self._update_circuit("infra_failed", limit)
        return outcome

    def _update_circuit(self, status: str, limit: int):
        if status == "fatal":
            self.db.execute("UPDATE run SET halt_reason='fatal_request_error'")
        elif status == "infra_failed":
            self.db.execute("UPDATE run SET failure_streak=failure_streak+1")
            self.db.execute("UPDATE run SET halt_reason='consecutive_request_failures' WHERE failure_streak>=?", (limit,))
        else:
            self.db.execute("UPDATE run SET failure_streak=0")

    def ensure_active(self):
        reason = self.rows("run")[0]["halt_reason"]
        if reason:
            raise RunHalted(f"Run halted: {reason}; inspect records and start a new run")

    def export_jsonl(self, output: Path):
        with self._mutex, output.open("w", encoding="utf-8") as stream:
            for table in ("run", "records", "links", "requests", "attempts"):
                for row in self.rows(table):
                    stream.write(json.dumps({"record_version": 1, "table": table, **row}, ensure_ascii=False) + "\n")


@contextmanager
def read_database(path: Path):
    database = path.resolve() / "records.sqlite3"
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise StoreConflict("Unsupported store version")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


def inspect_run(path: Path) -> dict:
    with read_database(path) as database:
        return {
            "run": dict(database.execute("SELECT * FROM run").fetchone()),
            "records": dict(database.execute("SELECT kind,COUNT(*) FROM records GROUP BY kind")),
            "requests": dict(database.execute("SELECT state,COUNT(*) FROM requests GROUP BY state")),
            "attempts": dict(database.execute("SELECT state,COUNT(*) FROM attempts GROUP BY state")),
        }


def export_run(path: Path, output: Path):
    output = output.resolve()
    if output == (path / "records.sqlite3").resolve() or (path / "blobs").resolve() in output.parents:
        raise ValueError("Export must not overwrite the database or blobs")
    with read_database(path) as database, output.open("w", encoding="utf-8") as stream:
        for table in ("run", "records", "links", "requests", "attempts"):
            for row in database.execute(f"SELECT * FROM {table} ORDER BY rowid"):
                stream.write(json.dumps({"record_version": 1, "table": table, **dict(row)}, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

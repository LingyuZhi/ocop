import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from filelock import Timeout

from ocop.runtime.storage import RunStore, StoreConflict


def test_resume_requires_identical_config_and_preserves_manifest(tmp_path):
    with RunStore(tmp_path, {"seed": 42}, run_id="run") as store:
        manifest = store.rows("run")[0]
        assert json.loads(store.read_blob(manifest["manifest_blob"]))["config"] == {"seed": 42}
    with RunStore(tmp_path, {"seed": 42}, run_id="run") as store:
        assert store.rows("run")[0] == manifest
    with pytest.raises(StoreConflict, match="configuration"):
        with RunStore(tmp_path, {"seed": 43}, run_id="run"):
            pass


def test_immutable_records_and_foreign_keys(tmp_path):
    with RunStore(tmp_path, {}) as store:
        task = store.put_record("task", "t1", {"split": "train"})
        proposal = store.put_record("proposal", "p1", {"slot": 0}, parents={"task": task})
        assert store.put_record("proposal", "p1", {"slot": 0}, parents={"task": task}) == proposal
        with pytest.raises(StoreConflict):
            store.put_record("proposal", "p1", {"slot": 1}, parents={"task": task})
        with pytest.raises(StoreConflict):
            store.put_record("proposal", "p1", {"slot": 0})
        with pytest.raises(sqlite3.IntegrityError):
            store.put_record("trajectory", "bad", {}, parents={"proposal": "missing"})
        assert not any(row["logical_key"] == "bad" for row in store.rows("records"))
        parent = proposal
        for kind in ("trajectory", "execution", "label", "sample", "checkpoint", "evaluation"):
            parent = store.put_record(kind, "example", {}, parents={"source": parent})
        assert len(store.rows("links")) == 7


def test_circuit_recovery_record_and_reset_are_atomic(tmp_path):
    with RunStore(tmp_path, {}) as store:
        store.halt("consecutive_request_failures")
        store.db.execute("""CREATE TEMP TRIGGER interrupt_recovery BEFORE UPDATE ON run
                            BEGIN SELECT RAISE(ABORT, 'simulated interruption'); END""")
        with pytest.raises(sqlite3.IntegrityError, match="simulated interruption"):
            store.recover_circuit("episode", {"verified": True})
        assert store.get_record("recovery", "episode") is None
        assert store.rows("run")[0]["halt_reason"] == "consecutive_request_failures"
        store.db.execute("DROP TRIGGER interrupt_recovery")
        store.recover_circuit("episode", {"verified": True})
        assert store.get_record("recovery", "episode") == {"verified": True}
        assert store.rows("run")[0]["halt_reason"] is None
        with pytest.raises(StoreConflict):
            store.recover_circuit("episode", {"verified": True})


def test_request_identity_includes_owner_and_exact_payload(tmp_path):
    with RunStore(tmp_path, {}) as store:
        request = store.request("q", {"temperature": 0.7}, None)
        assert store.request("q", {"temperature": 0.7}, None) == request
        with pytest.raises(StoreConflict):
            store.request("q", {"temperature": 0.0}, None)
        owner = store.put_record("task", "t", {})
        with pytest.raises(StoreConflict):
            store.request("q", {"temperature": 0.7}, owner)


def test_run_lock_prevents_recovering_an_active_writer(tmp_path):
    with RunStore(tmp_path, {}, run_id="run"):
        with pytest.raises(Timeout):
            with RunStore(tmp_path, {}, run_id="run"):
                pass


def test_request_cannot_start_two_concurrent_attempts(tmp_path):
    with RunStore(tmp_path, {}) as store:
        request = store.request("request", {}, None)
        store.start_attempt(request["id"])
        with pytest.raises(StoreConflict, match="not pending"):
            store.start_attempt(request["id"])


def test_interrupted_attempt_is_uncertain_and_durable_response_is_retained(tmp_path):
    with RunStore(tmp_path, {}, run_id="run") as store:
        uncertain = store.start_attempt(store.request("unknown", {}, None)["id"])
        saved = store.start_attempt(store.request("saved", {}, None)["id"])
        store.save_response(saved["id"], body=b'{"id":"r"}', status=200, headers={}, elapsed=0.25)
    with RunStore(tmp_path, {}, run_id="run") as store:
        attempts = {row["id"]: row for row in store.rows("attempts")}
        assert attempts[uncertain["id"]]["state"] == "uncertain"
        assert attempts[saved["id"]]["state"] == "response_saved"
        assert store.read_blob(attempts[saved["id"]]["body_blob"]) == b'{"id":"r"}'


def test_concurrent_record_writes_and_export(tmp_path):
    with RunStore(tmp_path, {}) as store:
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(lambda i: store.put_record("task", str(i), {"i": i}), range(32)))
        assert len(set(ids)) == 32
        output = tmp_path / "export.jsonl"
        store.export_jsonl(output)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert sum(row["table"] == "records" and row["kind"] == "task" for row in records) == 32


def test_blob_integrity_check(tmp_path):
    with RunStore(tmp_path, {}) as store:
        digest = store.put_blob(b"original")
        (store.path / "blobs" / digest).write_bytes(b"changed")
        with pytest.raises(StoreConflict, match="checksum"):
            store.read_blob(digest)


@pytest.mark.parametrize("run_id", ["../other", "/absolute", "", "x/y"])
def test_invalid_run_paths(tmp_path, run_id):
    with pytest.raises(ValueError):
        RunStore(tmp_path, {}, run_id=run_id)

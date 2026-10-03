import json
from pathlib import Path

import pytest

from ocop.cli import main
from ocop.storage import RunStore


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "graph"


@pytest.mark.parametrize(("name", "status", "valid"), [("valid.json", 0, True), ("invalid.json", 1, False)])
def test_validate_command(name, status, valid, capsys):
    assert main(["graph", "validate", str(EXAMPLES / name)]) == status
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is valid
    assert output["raw_content"] == (EXAMPLES / name).read_text()
    assert output["final_graph"] is not None if valid else output["final_graph"] is None


def test_reasoning_file_is_preserved(tmp_path, capsys):
    reasoning = tmp_path / "reasoning.txt"
    reasoning.write_text("Native reasoning\n", encoding="utf-8")
    assert main(["graph", "validate", str(EXAMPLES / "valid.json"), "--reasoning-file", str(reasoning)]) == 0
    assert json.loads(capsys.readouterr().out)["raw_reasoning"] == "Native reasoning\n"


@pytest.mark.parametrize("command", ["contract", "schema"])
def test_contract_commands(command, capsys):
    assert main(["graph", command]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), dict)


def test_missing_file_reports_io_error(tmp_path, capsys):
    assert main(["graph", "validate", str(tmp_path / "missing.json")]) == 2
    output = capsys.readouterr()
    assert output.err.startswith("ocop:")
    assert output.out == ""


@pytest.mark.parametrize("command", ["prepare"])
def test_reserved_commands_fail_explicitly(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_read_only_run_inspection_and_export(tmp_path, capsys):
    with RunStore(tmp_path, {}, run_id="run") as store:
        store.put_record("task", "t", {"split": "smoke"})
        records_before = len(store.rows("records"))
        path = store.path
    assert main(["runs", "show", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["records"]["task"] == 1
    output = tmp_path / "records.jsonl"
    assert main(["runs", "export", str(path), "--output", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert sum(row["table"] == "records" for row in rows) == records_before


@pytest.mark.parametrize(("answer", "reason"), [("#### 5", "success"), ("#### 6", "wrong_answer"), ("five", "format_error")])
def test_execute_cli_scores_without_retry_and_resumes(tmp_path, monkeypatch, capsys, answer, reason):
    import httpx

    import ocop.execution as execution
    from ocop.llm import RequestRunner

    root = EXAMPLES.parents[1]
    config = json.loads((root / "config/prototype.json").read_text())
    config["artifacts_dir"] = str(tmp_path / "artifacts")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    calls = []
    def handler(request):
        inputs = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert "reference_answer" not in inputs
        calls.append(inputs)
        return httpx.Response(200, json={"model": "openai/gpt-4o-mini", "provider": "OpenAI", "choices": [
            {"finish_reason": "stop", "message": {"content": answer}}]})
    monkeypatch.setattr(execution, "RequestRunner", lambda *args: RequestRunner(*args, transport=httpx.MockTransport(handler)))
    command = ["execute", "--task", str(root / "examples/execution/task.json"), "--trajectory", str(EXAMPLES / "valid.json"),
               "--config", str(config_path), "--credentials", str(tmp_path / "absent.env"), "--run-id", "run", "--repeat-id", "0"]
    assert main(command) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["score"]["reason"] == reason
    assert len(calls) == 5
    assert report["usage"]["worker"]["known_usage_sum"] is None
    assert report["usage"]["worker"]["unknown_usage_attempts"] == 4
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["score"] == report["score"]
    assert len(calls) == 5
    assert main(command[:-1] + ["1"]) == 0
    assert json.loads(capsys.readouterr().out)["request_count"] == 10
    assert len(calls) == 10


def test_execute_cli_rejects_invalid_graph_before_credentials(capsys):
    assert main(["execute", "--task", str(EXAMPLES.parent / "execution/task.json"),
                 "--trajectory", str(EXAMPLES / "invalid.json"), "--repeat-id", "0"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_graph"

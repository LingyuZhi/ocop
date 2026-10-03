import hashlib
import json
from importlib.resources import files
from typing import Any


VERSION = "ocop.graph.v1"


def load_contract() -> dict[str, Any]:
    resource = files("ocop.graph").joinpath("contracts/v1.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def contract_hash() -> str:
    payload = {"organization": load_contract(), "schema": trajectory_schema()}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def trajectory_schema() -> dict[str, Any]:
    contract = load_contract()
    worker = {"type": "string", "enum": contract["worker_ids"]}

    def action(name: str, properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"type": {"const": name}, **properties},
            "required": ["type", *properties],
            "additionalProperties": False,
        }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": VERSION,
        "type": "object",
        "properties": {
            "version": {"const": VERSION},
            "steps": {"type": "array", "items": {"$ref": "#/$defs/step"}},
        },
        "required": ["version", "steps"],
        "additionalProperties": False,
        "$defs": {
            "step": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string", "minLength": 1, "pattern": "\\S"},
                    "action": {
                        "oneOf": [
                            action("ASSIGN_ROLE", {"worker_id": worker, "role": {"enum": list(contract["roles"])}}),
                            action("ADD_EDGE", {"source": worker, "target": worker}),
                            action("STOP", {}),
                        ]
                    },
                },
                "required": ["explanation", "action"],
                "additionalProperties": False,
            }
        },
    }

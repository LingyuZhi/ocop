import argparse
import json
from pathlib import Path

from ocop.collection.verification import verify


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only verification of GSM8K collection records")
    parser.add_argument("path", type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.path, allow_pending=args.allow_pending)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

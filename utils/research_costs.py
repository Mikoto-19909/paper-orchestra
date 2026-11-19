"""Export a local research ledger report without making a network request."""
import argparse
import json
from pathlib import Path

from utils.research_gateway import Gateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not (args.ledger / "ledger.sqlite3").is_file():
        parser.error("Ledger database not found")
    report = Gateway(args.ledger, "report", "formal").report()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

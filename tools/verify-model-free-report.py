#!/usr/bin/env python3
"""Fail closed on absent, skipped, duplicate or failed required native tests."""
import sys
import xml.etree.ElementTree as E
from pathlib import Path

def main():
    required = set((Path(__file__).resolve().parents[1] / "tests/model-free-required.txt").read_text().splitlines())
    cases = E.parse(sys.argv[1]).findall(".//testcase")
    floor = max(len(required), int(sys.argv[2]))
    names = [c.get("name") for c in cases]
    if len(cases) < floor or len(set(names)) != len(names) or None in names:
        raise SystemExit(f"Invalid or incomplete report: {len(cases)} cases, expected at least {floor} distinct named tests")
    if required - set(names):
        raise SystemExit(f"Missing required tests: {sorted(required - set(names))}")
    for case in cases:
        if case.get("status") not in (None, "run") or any(case.find(k) is not None for k in ("failure", "error", "skipped")):
            raise SystemExit(f"Unsuccessful test: {case.get('name')}")
    print(f"Verified {len(cases)} model-free tests; external model inference is separate")
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read the literal immutable shared-workflow revision; no second pin."""
import re
import sys
from pathlib import Path
lines = Path(sys.argv[1]).read_text().splitlines()
pattern = r"\s+uses: myrqyry/obs-plugin-ci/\.github/workflows/build\.yml@([0-9a-f]{40})\s*"
matches = [m.group(1) for line in lines if (m := re.fullmatch(pattern, line))]
if len(matches) != 1:
    raise SystemExit("Expected one literal shared CI workflow SHA")
print(matches[0])

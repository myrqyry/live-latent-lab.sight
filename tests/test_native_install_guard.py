"""Verify the actual generated native install guard precedes all file writes."""
import sys
from pathlib import Path
install = (Path(sys.argv[1]) / "cmake_install.cmake").read_text()
marker = "Sight refuses system-wide installation"
if marker not in install or install.index(marker) > install.index("file(INSTALL"):
    raise SystemExit("Missing or late system prefix guard")
print("Native install prefix guard precedes file writes")

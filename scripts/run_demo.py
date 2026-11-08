from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if importlib.util.find_spec("streamlit") is None:
        print("缺少 Streamlit。请先运行：python -m pip install -r requirements.txt")
        return 1
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "streamlit", "run", str(root / "ui" / "app.py")]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())

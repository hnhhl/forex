#!/usr/bin/env python3
import uvicorn
import sys
from pathlib import Path

# Ensure project root (parent of this file's directory) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the app object directly to avoid import-by-string issues
from src.api.server import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reloader to prevent ModuleNotFoundError in subprocess
        workers=1,
    ) 
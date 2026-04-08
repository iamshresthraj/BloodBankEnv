"""
OpenEnv-compliant server/app.py entry point.
This module creates the FastAPI app instance following the OpenEnv standard structure.
The actual environment logic lives in bloodbank/server.py — this file re-exports the app.
"""

import sys
import os

# Add parent directory to path so bloodbank package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bloodbank.server import app

__all__ = ["app"]

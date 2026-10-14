"""
TrafficMind AI - Root Application Entry Point
Points to the refactored frontend application in frontend/app.py.

Usage:
    python -m streamlit run frontend/app.py
    (or: python -m streamlit run app.py)
"""

import sys
from pathlib import Path

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.app import main

if __name__ == "__main__":
    main()

#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
PYTHONPATH=. python -m streamlit run frontend/dashboard.py

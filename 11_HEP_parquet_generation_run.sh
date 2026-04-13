#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python HEP_parquet_generation_dashboard.py "$@"

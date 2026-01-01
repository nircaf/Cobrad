#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run Streamlit app in headless mode (for SSH access)
# Access from your LOCAL machine at: http://localhost:8503
echo "=============================================="
echo "Starting Streamlit server..."
echo "To view in your browser, open on your LOCAL machine:"
echo "  http://localhost:8503"
echo ""
echo "If not using SSH port forwarding, connect with:"
echo "  ssh -L 8503:localhost:8503 claustrum"
echo "=============================================="

streamlit run 5_streamlit_view.py --server.headless=true --server.port=8503


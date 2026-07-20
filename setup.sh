#!/bin/bash
# Quick setup — run from the project root:
#   chmod +x setup.sh && ./setup.sh
set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run the UI:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "To run the CLI test:"
echo "  source venv/bin/activate"
echo "  python run_cli.py"

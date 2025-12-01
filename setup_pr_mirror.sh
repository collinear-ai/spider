#!/bin/bash
# Quick setup script for PR Mirror

echo "Setting up PR Mirror task generation..."

# Create PR data directory
mkdir -p pr_data
cd pr_data

echo ""
echo "Choose an option:"
echo "1. Download from HuggingFace (grpc-go example)"
echo "2. Download from HuggingFace (pandas example)"
echo "3. Use your own file (skip download)"
read -p "Enter choice [1-3]: " choice

case $choice in
  1)
    echo "Downloading grpc-go dataset..."
    wget -q https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench/raw/main/go/grpc__grpc-go_dataset.jsonl
    echo "✓ Downloaded grpc__grpc-go_dataset.jsonl"
    FILE="grpc__grpc-go_dataset.jsonl"
    REPO="grpc/grpc-go"
    ;;
  2)
    echo "Downloading pandas dataset..."
    wget -q https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench/raw/main/python/pandas-dev__pandas_dataset.jsonl
    echo "✓ Downloaded pandas-dev__pandas_dataset.jsonl"
    FILE="pandas-dev__pandas_dataset.jsonl"
    REPO="pandas-dev/pandas"
    ;;
  3)
    read -p "Enter path to your PR data file: " FILE
    read -p "Enter repository (owner/repo): " REPO
    ;;
  *)
    echo "Invalid choice"
    exit 1
    ;;
esac

cd ..

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config/swe/pr-mirror-example.yaml:"
echo "   - Set mirror_org to your GitHub username"
echo "   - Set repository.github_url to: $REPO"
echo "   - Set file path to: pr_data/$FILE"
echo ""
echo "2. Start server:"
echo "   uvicorn server.app:app --host 0.0.0.0 --port 9000"
echo ""
echo "3. Run task generation:"
echo "   python run_task_gen.py config/swe/pr-mirror-example.yaml"

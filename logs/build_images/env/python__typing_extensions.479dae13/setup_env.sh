#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/meghanar-19/python__typing_extensions.479dae13 /testbed
cd /testbed
source /opt/miniconda3/bin/activate
conda create -n testbed python=3.10 -y
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
python -m pip install -e .

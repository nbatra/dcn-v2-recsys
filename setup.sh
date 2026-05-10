#!/bin/bash
# Setup script for DCN-v2 Recommendation project
# Downloads MovieLens 25M dataset and creates the Python environment

set -e

echo "=== DCN-v2 Recommendation Setup ==="
echo ""

# Step 1: Create virtual environment
echo "[1/3] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv --python 3.13 .venv
else
    echo "  .venv already exists, skipping."
fi

# Step 2: Install dependencies
echo "[2/3] Installing dependencies..."
uv pip install \
    torch \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    faiss-cpu \
    matplotlib \
    scipy \
    pyarrow \
    statsmodels \
    jupyterlab \
    ipywidgets

# Step 3: Download MovieLens 25M
echo "[3/3] Downloading MovieLens 25M dataset..."
mkdir -p data
if [ ! -d "data/ml-25m" ]; then
    echo "  Downloading from grouplens.org (~250MB compressed)..."
    curl -L -o data/ml-25m.zip "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    echo "  Extracting..."
    unzip -q data/ml-25m.zip -d data/
    rm data/ml-25m.zip
    echo "  Done. Dataset at data/ml-25m/"
else
    echo "  data/ml-25m/ already exists, skipping."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Run notebooks in order starting from 01_data_loading_and_exploration.ipynb"
echo "  2. NB04/07/10 train DCN-v2 rankers (~2-5 min each)"
echo "  3. NB05/08/11 run paired evaluation (~5-10 min each)"
echo "  4. Full pipeline takes ~2-3 hours on M4 Max"
echo ""
echo "  .venv/bin/jupyter lab notebooks/"

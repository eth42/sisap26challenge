#!/usr/bin/env bash

# Taken from
# https://github.com/sisap-challenges/sisap26-python-baseline/blob/main/download_datasets.sh
# License:
# MIT License

# Copyright (c) 2026 SISAP Challenges

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# download_datasets.sh
#
# Downloads all SISAP 2026 benchmark datasets from HuggingFace.
#
# Usage:
#   ./download_datasets.sh [--small-only]
#
#   --small-only   Download only the small development datasets (fast, < 1 GB).
#                  Skips the large full-scale datasets (wikipedia ~15 GB, nq ~7 GB).
#
# After running this script every dataset is ready to use:
#   python search.py --task task1 --dataset wikipedia-small
#   python search.py --task task2 --dataset llama-dev
#   python search.py --task task3 --dataset fiqa-dev

set -euo pipefail

# ---------------------------------------------------------------------------
# Check for hf CLI
# ---------------------------------------------------------------------------

if ! command -v hf &>/dev/null; then
    echo "Error: hf command not found."
    echo "Please install it with: pip install -U huggingface_hub[cli]"
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

SMALL_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--small-only" ]] && SMALL_ONLY=true
done

REPO="SISAP-Challenges/SISAP2026"

# ---------------------------------------------------------------------------
# Helper function to download a dataset directory
# ---------------------------------------------------------------------------

download_dataset() {
    local dataset_dir="$1"
    local dataset_name="$2"
    
    echo ""
    echo "-- $dataset_name --"
    
    # Download the entire directory using hf
    hf download "$REPO" \
        --repo-type dataset \
        --include "$dataset_dir/*" \
        --local-dir data
    
    echo "  done: data/$dataset_dir/"
}

# ---------------------------------------------------------------------------
# Task 1 – K-nearest neighbor graph  (k=15, dot product, normalized vectors)
# ---------------------------------------------------------------------------

echo ""
echo "=== Task 1: K-nearest neighbor graph ==="

download_dataset "wikipedia-small" "wikipedia-small (682 MB)"

if [[ "$SMALL_ONLY" == false ]]; then
    download_dataset "wikipedia" "wikipedia (full, ~15 GB)"
else
    echo ""
    echo "-- wikipedia (full) skipped (--small-only) --"
fi

download_dataset "task-1-spot-check" "task-1-spot-check (validation dataset)"

# ---------------------------------------------------------------------------
# Task 2 – Maximum Inner Product Search  (k=30, dot product, not normalized)
# ---------------------------------------------------------------------------

echo ""
echo "=== Task 2: Maximum Inner Product Search ==="

download_dataset "llama-dev" "llama-dev (134 MB)"

download_dataset "task-2-spot-check" "task-2-spot-check (validation dataset)"

# ---------------------------------------------------------------------------
# Task 3 – Sparse high-dimensional vectors  (k=30, dot product, SPLADE-v3)
# ---------------------------------------------------------------------------

echo ""
echo "=== Task 3: Sparse vector search ==="

download_dataset "fiqa-dev" "fiqa-dev (188 MB)"

if [[ "$SMALL_ONLY" == false ]]; then
    download_dataset "nq" "nq (full, ~6.9 GB)"
else
    echo ""
    echo "-- nq (full) skipped (--small-only) --"
fi

download_dataset "task-3-spot-check" "task-3-spot-check (validation dataset)"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "All done. Datasets available:"
echo ""
echo "  Task 1 (all-kNN graph, k=15):"
echo "    wikipedia-small   data/wikipedia-small/"
if [[ "$SMALL_ONLY" == false ]]; then
echo "    wikipedia         data/wikipedia/"
fi
echo ""
echo "  Task 2 (MIPS, k=30):"
echo "    llama-dev         data/llama-dev/"
echo ""
echo "  Task 3 (sparse search, k=30):"
echo "    fiqa-dev          data/fiqa-dev/"
if [[ "$SMALL_ONLY" == false ]]; then
echo "    nq                data/nq/"
fi
echo ""
echo "  Spot-check (validation datasets):"
echo "    task-1-spot-check data/task-1-spot-check/"
echo "    task-2-spot-check data/task-2-spot-check/"
echo "    task-3-spot-check data/task-3-spot-check/"
echo ""
echo "Run search.py with any of these dataset names, e.g.:"
echo "  python search.py --task task1 --dataset wikipedia-small"
echo "  python search.py --task task2 --dataset llama-dev"
echo "  python search.py --task task3 --dataset fiqa-dev"

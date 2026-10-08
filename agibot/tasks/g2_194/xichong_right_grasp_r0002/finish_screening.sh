#!/usr/bin/env bash
set -euo pipefail
# Run only on the collection robot; this does not transfer episodes or control it.
cd /home/agi/vla_ct/data_preparation/g2_194/xichong_right_grasp_r0002_20260920
while tmux has-session -t grasp_r0002_decode 2>/dev/null; do
    sleep 10
done
test -f image_decode.json
test -f numeric_screen.json
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
nice -n 15 python3 -u compare_subsets.py \
    --numeric-report numeric_screen.json --image-report image_decode.json \
    --output-dir subset_comparison

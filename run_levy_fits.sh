#!/usr/bin/env bash
# Runs all sinh-arcsinh / Levy-base fits (about 2000 steps each) in parallel, then evaluates them.
# Needs python3 with numpy and matplotlib. Run from this folder:   ./run_levy_fits.sh [steps]
# Takes roughly 10-25 minutes depending on the number of cores; logs in results/log_*.txt, figures in results/.
set -euo pipefail
cd "$(dirname "$0")"
STEPS=${1:-2000}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1   # one thread per fit; the fits run in parallel instead
CORES=$( (sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2) | head -n 1)
mkdir -p results
# model target alpha   (alpha = 2: Gaussian latent distribution; < 2: Levy-stable)
cat > results/jobs.txt <<EOF
sas real 2
sas real 1.5
gauss real 1.5
gauss syn 2
sas syn 2
sas syn 1.9
sas syn 1.5
EOF
echo "running $(wc -l < results/jobs.txt | tr -d ' ') fits on $CORES cores, $STEPS steps each"
< results/jobs.txt xargs -P "$CORES" -L 1 sh -c 'python3 train.py $0 '"$STEPS"' $1 $2 > results/log_$0_$1_a$2.txt 2>&1 && echo "done: $0 $1 alpha=$2"'
echo "evaluating ..."
python3 evaluate_levy.py > results/eval_levy.txt
echo "finished: see results/metrics_levy.json, results/fig_levy_real.png, results/fig_levy_closed.png"

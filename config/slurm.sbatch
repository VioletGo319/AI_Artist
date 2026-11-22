#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@60
#SBATCH --export=PATH=/usr/bin:/bin
set -euo pipefail

# Submitted only by the authorized private-host scheduler with policy-specific flags.
cd "$1"
export PATH="/usr/bin:/bin:$(dirname "$2")"
export ART_STATE_DIR="$1/private"
export ART_WEIGHTS_DIR="$1/weights"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$2" -m artcurator hpc-worker --day "$3" --batch "$4" --deadline "$5"

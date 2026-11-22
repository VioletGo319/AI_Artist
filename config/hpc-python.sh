#!/bin/bash
# Python executable adapter for CentOS 7 hosts. No GPU is exposed on login nodes.
set -euo pipefail
curator_root="$(cd "$(dirname "$0")/.." && pwd -P)"
runtime="$curator_root/private/runtime.env"
if [[ ! -f "$runtime" ]]; then
    printf 'Missing private/runtime.env; configure Apptainer first.\n' >&2
    exit 2
fi
# This is a private, user-controlled shell configuration, never downloaded data.
source "$runtime"
export APPTAINER_CACHEDIR="$curator_root/work/apptainer-cache"
# Old hosts without squashfuse unpack SIF files. Use node-local storage: NFS
# temporary rootfs cleanup can fail on .nfs files even after successful runs.
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/ai-art-curator-$UID"
umask 077
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
if [[ -L "$APPTAINER_TMPDIR" || ! -O "$APPTAINER_TMPDIR" ]]; then
    printf 'Apptainer temporary directory must be owned by this user, without a symlink.\n' >&2
    exit 2
fi
chmod 700 "$APPTAINER_TMPDIR"
: "${HPC_APPTAINER:?Set the absolute Apptainer executable path}"
if [[ "$HPC_APPTAINER" != /* || ! -x "$HPC_APPTAINER" ]]; then
    printf 'HPC_APPTAINER must be an absolute executable path.\n' >&2
    exit 2
fi
options=(exec --cleanenv --no-home --pid --ipc --bind "$curator_root:$curator_root" --pwd "$curator_root")
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    : "${CUDA_VISIBLE_DEVICES:?The Slurm job must allocate a GPU}"
    : "${HPC_GPU_IMAGE:?Configure and validate the GPU container first}"
    : "${HPC_GPU_PYTHON:?Set Python inside the GPU container}"
    image="$HPC_GPU_IMAGE"
    python="$HPC_GPU_PYTHON"
    options+=(--nv --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    options+=(--env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}")
    if [[ -n "${TMPDIR:-}" && -d "$TMPDIR" ]]; then
        options+=(--bind "$TMPDIR:$TMPDIR" --env "TMPDIR=$TMPDIR")
    fi
else
    : "${HPC_CONTROL_IMAGE:?Configure the CPU control container}"
    : "${HPC_CONTROL_PYTHON:?Set Python inside the control container}"
    image="$HPC_CONTROL_IMAGE"
    python="$HPC_CONTROL_PYTHON"
    options+=(--env CUDA_VISIBLE_DEVICES=)
fi
if [[ "$image" != /* || ! -f "$image" || "$python" != /* ]]; then
    printf 'Container image must exist at an absolute path; Python must be an absolute path.\n' >&2
    exit 2
fi
options+=(--env "ART_CONFIG=$curator_root/config.yaml" --env "ART_STATE_DIR=$curator_root/private")
options+=(--env "ART_WEIGHTS_DIR=$curator_root/weights" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1)
exec "$HPC_APPTAINER" "${options[@]}" "$image" "$python" "$@"

"""Release this job's resources without inspecting or resetting shared GPUs."""

import gc
import os
import signal
import subprocess
import sys


def release_cuda_cache():
    # Never import torch or initialize a CUDA context just to perform cleanup.
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except Exception as error:
            # Process exit releases the context even when CUDA itself has failed.
            print(f"CUDA cache cleanup failed; worker will exit: {error}", file=sys.stderr)


def run_isolated(command, *, cwd, env, timeout, check=False):
    """Reap a dedicated child process group on success, failure or interruption."""
    with subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True) as process:
        try:
            code = process.wait(timeout=timeout)
            if check and code:
                raise subprocess.CalledProcessError(code, command)
            return code
        finally:
            # This PGID belongs exclusively to the child we just created.
            # vLLM descendants may survive after their launcher has exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

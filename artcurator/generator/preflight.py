"""CPU-only model/cache checks. Passing this is not a GPU inference result."""

from importlib.metadata import version
import inspect
import os
from types import SimpleNamespace

from artcurator.generator.backends import check_cache
from artcurator.generator.upsampler import caption_messages
from artcurator.settings import FLUX


def check(env, *, dependencies_only=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ValueError("Model preflight is CPU-only; clear CUDA_VISIBLE_DEVICES.")
    import torch
    import flux3.pipeline as pipeline
    from transformers import Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration
    from flux3.dit import packed_attention
    from flux3.sampling import timesteps

    assert Qwen3VLForConditionalGeneration and Qwen3_5ForConditionalGeneration
    if env.settings.generation.upsampler_backend == "vllm":
        from flux3.pup import Upsampler
        import vllm
        assert Upsampler and vllm
    if list(inspect.signature(pipeline.Pipeline.generate).parameters)[:9] != [
            "self", "model", "prompt", "references", "width", "height", "steps", "guidance", "seed"]:
        raise RuntimeError("Unsupported official FLUX Pipeline interface.")
    # Small CPU kernel checks; never instantiate the full model on a login node.
    q = torch.zeros(1, 2, 4, 8)
    if packed_attention(q, q, q, [(0, 4)]).shape != (4, 16) or len(timesteps(16, 16.0)) != 17:
        raise RuntimeError("Official FLUX primitive check failed.")
    report = dict(model=FLUX, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  transformers=version("transformers"), upsampler=env.settings.generation.upsampler_backend,
                  dependencies="passed", cache="not_checked", gpu_inference="not_tested")
    if env.settings.generation.upsampler_backend == "transformers":
        if (torch.__version__.split("+")[0] != "2.6.0" or torch.version.cuda != "12.4"
                or report["transformers"] != "5.24.0"):
            raise RuntimeError("CUDA 12 compatibility profile requires torch 2.6.0+cu124 and transformers 5.24.0.")
    if not dependencies_only:
        check_cache(env, SimpleNamespace(backend="flux3"))
        caption_messages(env.weights / "pup", "A quiet landscape")
        report["cache"] = "passed"
    if torch.cuda.is_initialized():
        raise RuntimeError("Preflight unexpectedly initialized CUDA.")
    return report

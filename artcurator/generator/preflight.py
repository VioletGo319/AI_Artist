"""CPU-only model/cache checks. Passing this is not a GPU inference result."""

from importlib.metadata import version
import inspect
import os
from types import SimpleNamespace

from artcurator.generator.backends import check_cache
from artcurator.generator.upsampler import caption_messages
from artcurator.settings import FLUX


def check_processors(weights):
    """Load the real cached processors, not model weights, without CUDA."""
    from transformers import AutoProcessor

    encoder = AutoProcessor.from_pretrained(weights / "flux-3-image-klein" / "text_encoder",
                                           local_files_only=True)
    pup = AutoProcessor.from_pretrained(weights / "pup", local_files_only=True)
    messages = caption_messages(weights / "pup", "A solitary pavilion with mist-softened depth and open space.")
    text = pup.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = pup.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] + 3584 > 8192:
        raise RuntimeError("Official PUP template exceeds its context budget.")
    stop = [pup.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    if any(value is None or value == pup.tokenizer.unk_token_id for value in stop):
        raise RuntimeError("Official PUP stop tokens are unavailable.")
    return dict(text_encoder=type(encoder).__name__, upsampler=type(pup).__name__,
                input_tokens=ids.shape[1], status="passed")


def check(env, *, dependencies_only=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ValueError("Model preflight is CPU-only; clear CUDA_VISIBLE_DEVICES.")
    import torch
    import torchvision
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
                  torchvision=torchvision.__version__, transformers=version("transformers"),
                  upsampler=env.settings.generation.upsampler_backend,
                  dependencies="passed", cache="not_checked", gpu_inference="not_tested")
    if env.settings.generation.upsampler_backend == "transformers":
        if (torch.__version__.split("+")[0] != "2.6.0" or torch.version.cuda != "12.4"
                or torchvision.__version__.split("+")[0] != "0.21.0"
                or report["transformers"] != "5.24.0"):
            raise RuntimeError("CUDA 12 profile requires torch 2.6.0+cu124, torchvision 0.21.0 and transformers 5.24.0.")
    if not dependencies_only:
        check_cache(env, SimpleNamespace(backend="flux3"))
        caption_messages(env.weights / "pup", "A quiet landscape")
        report["processors"] = check_processors(env.weights)
        report["cache"] = "passed"
    if torch.cuda.is_initialized():
        raise RuntimeError("Preflight unexpectedly initialized CUDA.")
    return report

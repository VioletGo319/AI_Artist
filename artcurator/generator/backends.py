from importlib.util import find_spec
import math
import os
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw, ImageFilter


class NoGPU(RuntimeError):
    pass


def gpu(only_a100=False):
    if find_spec("torch") is None:
        raise NoGPU("PyTorch with CUDA is unavailable.")
    import torch

    if not torch.cuda.is_available():
        raise NoGPU("No visible CUDA GPU.")
    if only_a100 and (torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0)
                      or not 38 <= torch.cuda.get_device_properties(0).total_memory / 1024**3 < 45):
        raise NoGPU("Slurm must expose exactly one allocated A100 40 GB GPU.")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise NoGPU("This CUDA device lacks native BF16 support.")
    return torch


def verify_shards(directory: Path):
    import json

    files = set()
    for index in directory.glob("*.safetensors.index.json"):
        files.update(json.loads(index.read_text())["weight_map"].values())
    if not files:
        files = {path.name for path in directory.glob("*.safetensors")}
    return bool(files) and all((directory / file).is_file() and (directory / file).stat().st_size > 0
                               for file in files)


def check_cache(env, plan):
    if plan.backend == "mock":
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if plan.backend == "flux3":
        os.environ["WEIGHTS_DIR"] = str(env.weights)
        base = env.weights / "flux-3-image-klein"
        required = [base / "ae.safetensors", base / "model.safetensors",
                    base / "text_encoder/config.json", env.weights / "pup/config.json",
                    env.weights / "pup/system_prompts.json"]
        if (any(not f.is_file() or not f.stat().st_size for f in required)
                or not verify_shards(base / "text_encoder") or not verify_shards(env.weights / "pup")):
            raise RuntimeError(f"FLUX BF16 cache incomplete at {env.weights}. See README for the one-time download.")
        if find_spec("flux3") is None:
            raise RuntimeError("Install the pinned official FLUX Python package in the GPU environment.")
    elif not (env.weights / "diffusers" / plan.model / "model_index.json").is_file():
        raise RuntimeError(f"Diffusers cache incomplete under {env.weights / 'diffusers'}. See README.")


def make_upsampler(env, plan):
    from flux3.pup import Upsampler

    model = Upsampler(str(env.weights / "pup"), plan.upsampler_memory)
    return lambda prompt: model(prompt, [])


def renderer(env, plan):
    if plan.backend == "mock":
        return mock_image
    torch = gpu()
    if plan.backend == "flux3":
        from flux3.pipeline import Pipeline

        base = env.weights / "flux-3-image-klein"
        if plan.cpu_offload:
            # Move the text encoder, DiT and VAE separately. Pipeline's own offload
            # only changes when the DiT first moves to CUDA.
            from artcurator.generator.flux import staged_pipeline

            pipeline = staged_pipeline(base)
        else:
            pipeline = Pipeline({"distilled": str(base)}, str(base / "text_encoder"),
                                str(base / "ae.safetensors"))

        @torch.inference_mode()
        def draw(prompt, seed, width, height):
            return pipeline.generate("distilled", prompt, [], width, height, plan.steps, None, seed)

        return draw

    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        env.weights / "diffusers" / plan.model, torch_dtype=torch.bfloat16, local_files_only=True
    )
    if plan.cpu_offload:
        pipeline.enable_model_cpu_offload(gpu_id=0)
    else:
        pipeline.to("cuda:0")

    @torch.inference_mode()
    def draw(prompt, seed, width, height):
        return pipeline(prompt=prompt, generator=torch.Generator(device="cpu").manual_seed(seed),
                        width=width, height=height, num_inference_steps=plan.steps,
                        guidance_scale=plan.guidance).images[0]

    return draw


def mock_image(prompt, seed, width, height):
    """Deterministic visual fixture for exercising the offline pipeline."""
    rng = Random(seed)
    palette = [(106, 75, 160), (204, 107, 89), (54, 151, 146), (154, 124, 53), (85, 113, 202)]
    color = rng.choice(palette)
    size = (360, max(180, int(360 * height / width)))
    im = Image.new("RGB", size, (15, 16, 23))
    light = ImageDraw.Draw(im)
    w, h = size
    cx, cy = rng.randrange(w // 4, w * 3 // 4), rng.randrange(h // 4, h * 3 // 4)
    for radius in range(max(size), 3, -4):
        glow = max(.10, .75 * (1 - radius / max(size)) ** 3)
        light.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=tuple(int(x * glow) for x in color))
    im = im.filter(ImageFilter.GaussianBlur(15))
    draw = ImageDraw.Draw(im, "RGBA")
    for _ in range(260):
        px, py, brightness = rng.randrange(w), rng.randrange(h), rng.randrange(20, 150)
        draw.point((px, py), fill=(230, 225, 242, brightness))
    skew, turns = rng.uniform(.33, 1.3), rng.randint(12, 29)
    for i in range(turns):
        radius = w / 8 + 4 * i
        points = []
        for n in range(121):
            angle = n * math.tau / 120
            ripple = 1 + .19 * math.sin(angle * 3 + i * .11 + seed % 7)
            points.append((cx + radius * math.cos(angle) * ripple,
                           cy + radius * skew * math.sin(angle) * ripple + i ** 1.35 * .5))
        draw.line(points, fill=(*tuple(min(255, x + 94) for x in color), 155 - 3 * i), width=1)
    return im.resize((width, height), Image.Resampling.LANCZOS)

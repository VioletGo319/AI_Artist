"""Text-to-image-only staging using the pinned BFL Pipeline and its components."""

import torch
from flux3.pipeline import Pipeline


class StagedPipeline(Pipeline):
    def __init__(self, base):
        super().__init__({"distilled": str(base)}, str(base / "text_encoder"),
                         str(base / "ae.safetensors"), device="cpu", offload=True)

    @torch.inference_mode()
    def generate(self, model, prompt, references, width, height, steps, guidance, seed):
        from flux3.pipeline import (IN_CHANNELS, denoise_flow_map, latent_ids, latent_shape,
                                    pack, scaled_ids, timesteps, to_pil)

        device = "cuda:0"
        encoder = self.text_encoder
        encoder.model.to(device)
        encoder.device = device
        try:
            caption = encoder.encode(prompt)
        finally:
            encoder.model.to("cpu")
            encoder.device = "cpu"
        torch.cuda.empty_cache()

        h, w = latent_shape(height, width)
        noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=torch.Generator().manual_seed(seed))
        ids = latent_ids(h, w, device)
        image = (noise.to(torch.bfloat16).float().to(device), scaled_ids(ids))
        dit = self.dits[model].to(device)
        try:
            tokens = denoise_flow_map(dit, pack([[caption, None, image]]), timesteps(steps, 16.0))
        finally:
            dit.to("cpu")
        del caption, image
        torch.cuda.empty_cache()
        self.autoencoder.to(device)
        try:
            return to_pil(self.autoencoder.decode(tokens, ids))
        finally:
            self.autoencoder.to("cpu")


def staged_pipeline(base):
    return StagedPipeline(base)

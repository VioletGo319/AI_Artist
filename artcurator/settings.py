from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field
import os
import yaml

ROOT = Path(__file__).resolve().parent.parent
FLUX = "black-forest-labs/flux-3-image-klein"
Z_IMAGE = "Tongyi-MAI/Z-Image-Turbo"
SIZES = {"1:1": (1024, 1024), "16:9": (1280, 720), "3:2": (1248, 832), "9:16": (720, 1280)}
Backend = Literal["flux3", "diffusers", "mock"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Generation(Strict):
    backend: Backend = "flux3"
    count: int = Field(8, ge=1, le=16)
    aspect_ratios: list[Literal["1:1", "16:9", "3:2", "9:16"]] = Field(default_factory=lambda: ["1:1"], min_length=1)
    cpu_offload: bool = True
    upsampler_memory: float = Field(.86, ge=.6, le=.95)
    duplicate_distance: int = Field(3, ge=0, le=12)
    aesthetic_score: bool = False
    diffusers_model: str = Field(Z_IMAGE, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*/[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    diffusers_steps: int = Field(9, ge=1, le=100)
    diffusers_guidance: float = Field(0, ge=0, le=30)
    diffusers_license_approved: bool = False

    def parameters(self):
        if self.backend == "flux3":
            return FLUX, 16, None
        if self.backend == "mock":
            return "mock/procedural", 1, None
        if self.diffusers_model != Z_IMAGE and not self.diffusers_license_approved:
            raise ValueError("Review the model license and set diffusers_license_approved before using this model.")
        return self.diffusers_model, self.diffusers_steps, self.diffusers_guidance


class Preference(Strict):
    exploration: float = Field(.3, ge=.05, le=1)


class Schedule(Strict):
    mode: Literal["local", "slurm", "off"] = "slurm"
    start_hour: int = Field(8, ge=0, le=23)
    runtime_minutes: int = Field(120, ge=1, le=120)
    admission_minutes: int = Field(10, ge=1, le=60)
    cpus: int = Field(8, ge=1, le=64)
    memory: str = Field("64G", pattern=r"^[1-9][0-9]*[MG]$")


class Mail(Strict):
    retries: int = Field(3, ge=1, le=5)
    reply_days: int = Field(30, ge=1, le=365)
    require_dmarc: bool = True


class GitPublish(Strict):
    remote: str = Field("origin", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    branch: str = Field("main", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_./-]*$")


class Settings(Strict):
    timezone: str = "America/New_York"
    generation: Generation = Field(default_factory=Generation)
    preference: Preference = Field(default_factory=Preference)
    scheduler: Schedule = Field(default_factory=Schedule)
    mail: Mail = Field(default_factory=Mail)
    publish: GitPublish = Field(default_factory=GitPublish)


class Environment:
    def __init__(self, root: Path = ROOT, values=None, load_config=True):
        self.root = root
        self.values = values if values is not None else {**dotenv_values(root / ".env"), **os.environ}
        config = self.path("ART_CONFIG", "config.yaml")
        payload = yaml.safe_load(config.read_text()) if load_config else {}
        self.settings = Settings.model_validate(payload or {})
        self.tz = ZoneInfo(self.settings.timezone)
        if self.get("ART_BACKEND"):
            generation = self.settings.generation.model_dump()
            generation["backend"] = self.get("ART_BACKEND")
            self.settings.generation = Generation.model_validate(generation)
        self.state = self.path("ART_STATE_DIR", "private")
        self.public = self.path("ART_PUBLIC_DIR", "web/public")
        self.weights = self.path("ART_WEIGHTS_DIR", "weights")
        if (self.state.is_relative_to(self.public) or self.public.is_relative_to(self.state)
                or self.weights.is_relative_to(self.public)):
            raise ValueError("Private state, weights and public output must use separate directories.")

    def get(self, key, default=""):
        return self.values.get(key) or default

    def flag(self, key):
        value = self.get(key, "false").lower()
        if value not in {"true", "false"}:
            raise ValueError(f"{key} must be true or false.")
        return value == "true"

    def path(self, key, default):
        return (self.root / self.get(key, default)).expanduser().resolve()

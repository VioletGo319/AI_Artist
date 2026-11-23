from datetime import date
import json
from random import Random
import uuid

from pydantic import Field

from artcurator.preference_engine.engine import inspiration
from artcurator.settings import Backend, MAX_CANDIDATES, SIZES, Strict
from artcurator.store import utcnow

BATCH_PATTERN = r"^[a-f0-9]{32}$"
ATTEMPTS = 4


class Slot(Strict):
    theme: str
    palette: str
    composition: str
    style: str
    title: str
    prompt: str
    width: int = Field(ge=64, le=2048, multiple_of=16)
    height: int = Field(ge=64, le=2048, multiple_of=16)
    seeds: list[int] = Field(min_length=ATTEMPTS, max_length=ATTEMPTS)


class Plan(Strict):
    id: str = Field(pattern=BATCH_PATTERN)
    day: date
    backend: Backend
    model: str
    steps: int = Field(ge=1, le=100)
    guidance: float | None
    cpu_offload: bool
    upsampler_memory: float = Field(ge=.6, le=.95)
    duplicate_distance: int = Field(ge=0, le=12)
    aesthetic_score: bool
    time_budget: bool = False
    cleanup_seconds: int = Field(60, ge=60, le=180)
    slots: list[Slot] = Field(min_length=1, max_length=MAX_CANDIDATES)

    def work_id(self, slot):
        return f"{self.id}-{slot:02d}"


def make_plan(store, env, day, count, route):
    with store.connect(write=True) as db:
        if row := db.execute("SELECT * FROM batches WHERE day=?", (day,)).fetchone():
            return Plan.model_validate_json(row["plan"]), row["route"]
        previous = db.execute("SELECT plan FROM batches WHERE day < ? ORDER BY day DESC LIMIT 1", (day,)).fetchone()
        latest = Plan.model_validate_json(previous["plan"]).model_dump()["slots"] if previous else []
        likes = [json.loads(row["snapshot"]) for row in db.execute(
            "SELECT snapshot FROM approvals ORDER BY chosen_at DESC LIMIT 500"
        )]
        exposures = [json.loads(row["data"]) for row in db.execute("""
            SELECT works.data FROM works JOIN deliveries ON deliveries.batch=works.batch
            WHERE deliveries.decision='approve' ORDER BY deliveries.sent_at DESC LIMIT 1000
        """)]
        generation = env.settings.generation
        model, steps, guidance = generation.parameters()
        identity = uuid.uuid4()
        rng = Random(identity.int)
        ideas = inspiration(identity.int, count, latest, likes, env.settings.preference.exploration, exposures,
                            direction=generation.creative_direction)
        used_seeds = {seed for row in db.execute("SELECT plan FROM batches")
                      for slot in Plan.model_validate_json(row["plan"]).slots for seed in slot.seeds}
        seeds = []
        while len(seeds) < count * ATTEMPTS:
            value = rng.getrandbits(32)
            if value not in used_seeds:
                used_seeds.add(value)
                seeds.append(value)
        slots = []
        for i, idea in enumerate(ideas):
            width, height = SIZES[generation.aspect_ratios[i % len(generation.aspect_ratios)]]
            slots.append(Slot(**idea, width=width, height=height, seeds=seeds[i * ATTEMPTS:(i + 1) * ATTEMPTS]))
        plan = Plan(id=identity.hex, day=day, backend=generation.backend, model=model, steps=steps,
                    guidance=guidance, cpu_offload=generation.cpu_offload, upsampler_memory=generation.upsampler_memory,
                    duplicate_distance=generation.duplicate_distance, aesthetic_score=generation.aesthetic_score,
                    time_budget=generation.mode == "time_budget", cleanup_seconds=generation.cleanup_seconds, slots=slots)
        save_plan(db, plan, route)
        return plan, route


def save_plan(db, plan, route):
    db.execute("INSERT INTO batches(id,day,count,plan,route,created) VALUES (?,?,?,?,?,?)",
               (plan.id, plan.day.isoformat(), len(plan.slots), plan.model_dump_json(), route, utcnow()))

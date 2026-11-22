from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
import subprocess
import sys

from PIL import Image, ImageStat

from artcurator.generator.backends import NoGPU, check_cache, gpu, make_upsampler, renderer
from artcurator.generator.plan import Plan
from artcurator.generator.resources import release_cuda_cache, run_isolated
from artcurator.store import atomic, utcnow


def png(image):
    out = BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def thumbnail(image, side=480, format="WEBP"):
    preview = image.copy()
    preview.thumbnail((side, side), Image.Resampling.LANCZOS)
    out = BytesIO()
    preview.convert("RGB").save(out, format, quality=84)
    return out.getvalue()


def measure(image):
    small = image.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
    stats = ImageStat.Stat(small)
    gray = small.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    value = 0
    for row in range(8):
        for col in range(8):
            i = row * 9 + col
            value = (value << 1) | (gray[i] > gray[i + 1])
    return stats.stddev[0], stats.mean[0], f"{value:016x}"


def directory(store, plan):
    return store.root / "candidates" / plan.id


def get_plan(store, batch):
    with store.connect() as db:
        row = db.execute("SELECT plan FROM batches WHERE id=?", (batch,)).fetchone()
    if not row:
        raise ValueError("Unknown batch.")
    return Plan.model_validate_json(row["plan"])


def expand(store, env, plan):
    gpu()
    check_cache(env, plan)
    folder = directory(store, plan)
    missing = [(i, slot) for i, slot in enumerate(plan.slots, 1) if not (folder / f"{i:02d}.prompt").is_file()]
    if not missing:
        return
    upsample = None
    try:
        upsample = make_upsampler(env, plan)
        for i, slot in missing:
            caption = upsample(slot.prompt)
            if not caption.strip():
                raise RuntimeError("FLUX prompt upsampler returned an empty caption.")
            atomic(folder / f"{i:02d}.prompt", caption.encode())
    finally:
        upsample = None
        release_cuda_cache()


def generation_state(store, plan, state, unsealed=True):
    with store.connect(write=True) as db:
        if unsealed and db.execute("SELECT 1 FROM deliveries WHERE batch=?", (plan.id,)).fetchone():
            return
        db.execute("UPDATE batches SET state=? WHERE id=?", (state, plan.id))


def finalize(store, plan):
    count = len(store.artworks(plan.id))
    generation_state(store, plan, "ready" if count == len(plan.slots) else "partial" if count else "empty")
    return count


def run(store, env, plan, deadline, hpc=False):
    if datetime.now(timezone.utc) >= deadline:
        return 0
    with store.lock("gpu-worker"):
        with store.connect() as db:
            if db.execute("SELECT 1 FROM deliveries WHERE batch=?", (plan.id,)).fetchone():
                return len(store.artworks(plan.id))
        draw = None
        try:
            if plan.backend != "mock":
                gpu(only_a100=hpc)
            check_cache(env, plan)
            recover_written(store, plan)
            complete = {art["slot"] for art in store.artworks(plan.id)}
            if len(complete) == len(plan.slots):
                return finalize(store, plan)
            generation_state(store, plan, "running")
            folder = directory(store, plan)
            folder.mkdir(parents=True, mode=0o700, exist_ok=True)
            if plan.backend == "flux3":
                command = [sys.executable, "-m", "artcurator", "_expand", "--batch", plan.id]
                run_isolated(command, cwd=env.root, env=child_environment(env), check=True,
                               timeout=max(1, (deadline - datetime.now(timezone.utc)).total_seconds()))
            draw = renderer(env, plan)
            for i, slot in enumerate(plan.slots, 1):
                if i in complete:
                    continue
                if datetime.now(timezone.utc) >= deadline:
                    break
                prompt = (folder / f"{i:02d}.prompt").read_text() if plan.backend == "flux3" else slot.prompt
                for seed in slot.seeds:
                    if datetime.now(timezone.utc) >= deadline:
                        break
                    checkpoint = folder / f"{i:02d}-{seed}.png"
                    if checkpoint.is_file():
                        with Image.open(checkpoint) as saved:
                            image = saved.convert("RGB")
                    else:
                        image = draw(prompt, seed, slot.width, slot.height).convert("RGB")
                    if persist_candidate(store, plan, i, seed, prompt, image, checkpoint):
                        break
        except NoGPU:
            raise
        finally:
            draw = None
            try:
                finalize(store, plan)
            finally:
                release_cuda_cache()
        return len(store.artworks(plan.id))


def recover_written(store, plan):
    """Index a PNG that reached disk just before an interrupted database commit."""
    folder = directory(store, plan)
    complete = {item["slot"] for item in store.artworks(plan.id)}
    for index, slot in enumerate(plan.slots, 1):
        if index in complete:
            continue
        for seed in slot.seeds:
            path = folder / f"{index:02d}-{seed}.png"
            if not path.is_file():
                continue
            prompt = (folder / f"{index:02d}.prompt").read_text() if plan.backend == "flux3" else slot.prompt
            stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            with Image.open(path) as image:
                if persist_candidate(store, plan, index, seed, prompt, image.convert("RGB"), path, stamp):
                    break


def persist_candidate(store, plan, index, seed, prompt, image, checkpoint, generated_at=None):
    slot = plan.slots[index - 1]
    if image.size != (slot.width, slot.height):
        raise RuntimeError(f"Incorrect output size {image.size} for slot {index}.")
    contrast, brightness, phash = measure(image)
    if contrast < 6 or not 8 < brightness < 247:
        return False
    raw = checkpoint.read_bytes() if checkpoint.is_file() else png(image)
    digest = sha256(raw).hexdigest()
    identity = plan.work_id(index)
    with store.connect(write=True) as db:
        if db.execute("SELECT 1 FROM works WHERE id=?", (identity,)).fetchone():
            return True
        if db.execute("SELECT 1 FROM works WHERE sha=?", (digest,)).fetchone():
            return False
        hashes = db.execute("""
            SELECT phash, works.data FROM works JOIN batches ON batches.id=works.batch
            ORDER BY batches.day DESC LIMIT 300
        """)
        for item in hashes:
            if ((int(phash, 16) ^ int(item["phash"], 16)).bit_count() <= plan.duplicate_distance
                    and json.loads(item["data"])["model"].startswith("mock/") == (plan.backend == "mock")):
                return False
        if not checkpoint.is_file():
            atomic(checkpoint, raw)
        atomic(checkpoint.parent / f"{identity}.webp", thumbnail(image, 720))
        record = dict(id=identity, slot=index, title=slot.title, theme=slot.theme, palette=slot.palette,
                      composition=slot.composition, style=slot.style, input_prompt=slot.prompt, prompt=prompt,
                      seed=seed, width=slot.width, height=slot.height, steps=plan.steps, model=plan.model,
                      date=plan.day.isoformat(), generated_at=generated_at or utcnow(), image=checkpoint.name,
                      thumb=f"{identity}.webp", mock=plan.backend == "mock")
        if plan.aesthetic_score:
            record["aesthetic"] = round(min(1, contrast / 65) * .6 + (1 - abs(brightness - 127.5) / 127.5) * .4, 3)
        db.execute("INSERT INTO works(id,batch,slot,data,sha,phash) VALUES (?,?,?,?,?,?)",
                   (identity, plan.id, index, json.dumps(record), digest, phash))
    return True


def child_environment(env):
    return {**os.environ, **{key: value for key, value in env.values.items() if value is not None},
            "ART_STATE_DIR": str(env.state), "ART_WEIGHTS_DIR": str(env.weights),
            "ART_CONFIG": str(env.path("ART_CONFIG", "config.yaml"))}


def launch_local(env, plan, deadline):
    command = [env.get("ART_GPU_PYTHON", sys.executable), "-m", "artcurator", "_worker",
               "--batch", plan.id, "--deadline", deadline.isoformat()]
    try:
        return run_isolated(command, cwd=env.root, env=child_environment(env),
                            timeout=max(0, (deadline - datetime.now(timezone.utc)).total_seconds()))
    except subprocess.TimeoutExpired:
        return 124

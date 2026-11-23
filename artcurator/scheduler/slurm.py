from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
import re
import shlex
import subprocess
from zipfile import ZipFile, ZIP_STORED

from PIL import Image

from artcurator.generator.plan import Plan, save_plan
from artcurator.generator.worker import directory, finalize, get_plan, persist_candidate, recover_written
from artcurator.settings import MAX_RUNTIME_MINUTES, MAX_CANDIDATES

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
HOST = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*")
MAX_ARCHIVE = 128_000_000


class Slurm:
    def __init__(self, env, dry=False):
        self.env = env
        self.host = env.get("HPC_HOST")
        self.root = env.get("HPC_REMOTE_ROOT", "/approved/ai-art-curator")
        self.python = env.get("HPC_REMOTE_PYTHON", self.root + "/.venv/bin/python")
        self.partition = env.get("HPC_PARTITION", "APPROVED_PARTITION")
        self.qos = env.get("HPC_QOS", "APPROVED_QOS")
        self.account = env.get("HPC_ACCOUNT", "APPROVED_ACCOUNT")
        self.workgroup = env.get("HPC_WORKGROUP")
        if not dry:
            if not all(env.flag(key) for key in ("HPC_ENABLED", "HPC_AUTHORIZED", "HPC_OPPORTUNISTIC")):
                raise ValueError("Slurm requires HPC_ENABLED, HPC_AUTHORIZED and HPC_OPPORTUNISTIC.")
            if not all(env.get(key) for key in ("HPC_HOST", "HPC_PARTITION", "HPC_QOS", "HPC_ACCOUNT",
                                                "HPC_REMOTE_ROOT", "HPC_REMOTE_PYTHON")):
                raise ValueError("Approved Slurm host, account, partition, QoS, root and Python are required.")
        if (any(not TOKEN.fullmatch(value) for value in (self.partition, self.qos, self.account))
                or (self.workgroup and not TOKEN.fullmatch(self.workgroup))
                or any(not PurePosixPath(value).is_absolute() or "\n" in value for value in (self.root, self.python))
                or (self.host and not HOST.fullmatch(self.host))):
            raise ValueError("Invalid Slurm settings.")

    def arguments(self, plan, deadline, runtime, scheduled_day=None):
        if not 1 <= runtime <= MAX_RUNTIME_MINUTES:
            raise ValueError("Slurm runtime must be between 1 and 30 minutes.")
        # Slurm interprets its time strings in the approved submission host's timezone.
        time_limit = f"{runtime // 60:02d}:{runtime % 60:02d}:00"
        return ["sbatch", "--parsable", f"--account={self.account}", f"--partition={self.partition}",
                f"--qos={self.qos}", "--gres=gpu:a100:1", f"--time={time_limit}",
                f"--deadline={deadline.astimezone(self.env.tz):%Y-%m-%dT%H:%M:%S}",
                "--no-requeue", f"--job-name=ac-{plan.id}", "--export=PATH=/usr/bin:/bin",
                f"--cpus-per-task={self.env.settings.scheduler.cpus}", f"--mem={self.env.settings.scheduler.memory}",
                f"--chdir={self.root}", f"--output={self.root}/private/slurm-%j.out",
                f"{self.root}/config/slurm.sbatch", self.root, self.python,
                scheduled_day or plan.day.isoformat(), plan.id, deadline.isoformat()]

    def ssh_command(self, command):
        remote = f"cd {shlex.quote(self.root)} && exec {shlex.join(command)}"
        if self.workgroup:
            # Caviness workgroup requires one quoted command string. The '@'
            # form joins argv and loses nested quoting for sh -c and paths.
            remote = shlex.join(["workgroup", "--group", self.workgroup, "--command", remote])
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", "-o", "StrictHostKeyChecking=yes",
                self.host or "APPROVED_HOST", remote]

    def ssh(self, command, data=None):
        return subprocess.run(self.ssh_command(command), input=data, capture_output=True, timeout=55)

    def worker(self, action, *args, data=None):
        return self.ssh(["env", f"ART_STATE_DIR={self.root}/private", self.python, "-m", "artcurator", action, *args],
                        data=data)

    def stage(self, plan):
        checked(self.worker("hpc-stage", data=plan.model_dump_json().encode()), "stage")

    def submit(self, plan, deadline, runtime, scheduled_day):
        result = self.ssh(self.arguments(plan, deadline, runtime, scheduled_day))
        if result.returncode != 0:
            return None
        match = re.fullmatch(rb"([0-9]+)(?:;[^\s]+)?\s*", result.stdout)
        return match[1].decode() if match else None

    def active(self, plan):
        # Slurm 17.11 has no --me. Resolve the SSH account's UID remotely,
        # rather than querying the private scheduler host's local username.
        query = 'exec squeue --user="$(id -u)" ' + shlex.join(
            ["--noheader", f"--name=ac-{plan.id}", "--format=%i"])
        result = self.ssh(["sh", "-c", query])
        return checked(result, "query").strip().decode().splitlines()

    def cancel(self, jobs):
        if any(not re.fullmatch(r"[0-9]+", job) for job in jobs):
            raise RuntimeError("Unexpected Slurm job ID.")
        if jobs:
            checked(self.ssh(["scancel", *jobs]), "cancel")

    def retrieve(self, plan):
        return checked(self.worker("hpc-export", "--batch", plan.id), "retrieve")

    def retrieve_pages(self, plan):
        # Bound memory and SSH transfer size even when a time-budget batch is large.
        after = 0
        while after < len(plan.slots):
            payload = checked(self.worker("hpc-export", "--batch", plan.id,
                                          "--after", str(after), "--limit", "8"), "retrieve")
            if len(payload) > MAX_ARCHIVE:
                raise ValueError("Oversized Slurm archive page.")
            with ZipFile(BytesIO(payload)) as archive:
                info = archive.getinfo("index.json")
                if info.file_size > 2_000_000:
                    raise ValueError("Oversized Slurm archive index.")
                manifest = json.loads(archive.read("index.json"))
            slots = [row["slot"] for row in manifest["records"]]
            if (manifest["batch"] != plan.id or len(slots) > 8
                    or any(type(slot) is not int or not after < slot <= len(plan.slots) for slot in slots)
                    or slots != sorted(set(slots))):
                raise ValueError("Invalid Slurm archive pagination.")
            yield payload
            if len(slots) < 8:
                break
            after = slots[-1]


def checked(result, action):
    if result.returncode:
        raise RuntimeError(f"Slurm {action} failed (exit {result.returncode}): {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def stage_request(store, payload):
    plan = Plan.model_validate_json(payload)
    if plan.backend == "mock":
        raise ValueError("Slurm accepts a real model only.")
    with store.connect(write=True) as db:
        previous = db.execute("SELECT plan FROM batches WHERE day=?", (plan.day.isoformat(),)).fetchone()
        if previous:
            if Plan.model_validate_json(previous["plan"]) != plan:
                raise ValueError("A different plan already exists on the worker for this day.")
        else:
            save_plan(db, plan, "slurm")
    return plan


def export_request(store, batch, after=0, limit=None):
    if type(after) is not int or not 0 <= after <= MAX_CANDIDATES:
        raise ValueError("Invalid export cursor.")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 8):
        raise ValueError("Export pages contain one to eight images.")
    plan = get_plan(store, batch)
    with store.lock("gpu-worker"):
        recover_written(store, plan)
        out = BytesIO()
        with ZipFile(out, "w", compression=ZIP_STORED) as archive, store.connect() as db:
            records = []
            for row in db.execute("SELECT data,sha FROM works WHERE batch=? AND slot>? ORDER BY slot LIMIT ?",
                                  (plan.id, after, limit or MAX_CANDIDATES)):
                art = json.loads(row["data"])
                raw = (directory(store, plan) / art["image"]).read_bytes()
                if sha256(raw).hexdigest() != row["sha"]:
                    raise ValueError("Checkpoint integrity failure.")
                archive.writestr(art["id"] + ".png", raw)
                records.append(dict(slot=art["slot"], prompt=art["prompt"], seed=art["seed"],
                                    generated_at=art["generated_at"], sha=row["sha"]))
            archive.writestr("index.json", json.dumps(dict(batch=plan.id, records=records)))
        if out.tell() > MAX_ARCHIVE:
            raise ValueError("Slurm archive exceeds the configured size.")
        return out.getvalue()


def import_results(store, plan, payload, *, finalize_batch=True):
    if len(payload) > MAX_ARCHIVE:
        raise ValueError("Slurm archive exceeds the configured size.")
    with store.lock("gpu-worker"), ZipFile(BytesIO(payload)) as archive:
        entries = archive.infolist()
        if len(entries) > len(plan.slots) + 1 or sum(entry.file_size for entry in entries) > MAX_ARCHIVE:
            raise ValueError("Invalid Slurm archive size.")
        manifest = json.loads(archive.read("index.json"))
        records = manifest["records"]
        if manifest["batch"] != plan.id or not isinstance(records, list) or len(records) != len(entries) - 1:
            raise ValueError("Slurm archive does not match the requested batch.")
        names = {"index.json"}
        prepared = []
        for item in records:
            index = item["slot"]
            seed = item["seed"]
            prompt = item["prompt"]
            stamp = datetime.fromisoformat(item["generated_at"])
            if (type(index) is not int or not 1 <= index <= len(plan.slots)
                    or stamp.tzinfo is None or seed not in plan.slots[index - 1].seeds or not isinstance(prompt, str) or not 0 < len(prompt) <= 60_000):
                raise ValueError("Invalid remote slot metadata.")
            name = plan.work_id(index) + ".png"
            if name in names:
                raise ValueError("Repeated remote slot.")
            names.add(name)
            raw = archive.read(name)
            if sha256(raw).hexdigest() != item["sha"]:
                raise ValueError("Invalid remote artwork digest.")
            with Image.open(BytesIO(raw), formats=["PNG"]) as image:
                if image.size != (plan.slots[index - 1].width, plan.slots[index - 1].height):
                    raise ValueError("Invalid remote artwork resolution.")
                prepared.append((index, seed, prompt, image.convert("RGB"), stamp.astimezone(timezone.utc).isoformat()))
        if {entry.filename for entry in entries} != names:
            raise ValueError("Unexpected file in Slurm archive.")
        for index, seed, prompt, image, stamp in prepared:
            persist_candidate(store, plan, index, seed, prompt, image,
                              directory(store, plan) / f"{index:02d}-{seed}.png", stamp)
        return finalize(store, plan) if finalize_batch else len(store.artworks(plan.id))

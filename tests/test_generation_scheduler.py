from datetime import datetime, timedelta, timezone
import json
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from artcurator.generator import backends, worker
from artcurator.generator.plan import make_plan
from artcurator.scheduler import service
from artcurator.scheduler.slurm import Slurm, export_request, import_results, stage_request
from artcurator.settings import FLUX, SIZES, Z_IMAGE, Environment, Generation, Schedule
from artcurator.store import Store


NOW = datetime(2026, 11, 22, 15, 0, tzinfo=timezone.utc)
DAY = "2026-11-22"


def test_generation_configuration_and_license(setup):
    env, store = setup
    assert Generation().parameters() == (FLUX, 16, None)
    assert Generation().count == 8
    assert Generation(backend="diffusers").parameters() == (Z_IMAGE, 9, 0.0)
    assert SIZES["1:1"] == (1024, 1024)
    with pytest.raises(ValueError, match="license"):
        Generation(backend="diffusers", diffusers_model="example/model").parameters()
    env.settings.generation = Generation(backend="flux3", aspect_ratios=list(SIZES))
    plan, _ = make_plan(store, env, DAY, 8, "local")
    assert [(slot.width, slot.height) for slot in plan.slots[:4]] == list(SIZES.values())
    assert plan.cpu_offload and len({seed for slot in plan.slots for seed in slot.seeds}) == 32
    other, _ = make_plan(store, env, "2026-11-23", 8, "local")
    assert not {(s.theme, s.palette) for s in plan.slots} & {(s.theme, s.palette) for s in other.slots}


def test_gpu_not_available_is_graceful(setup, monkeypatch):
    env, store = setup
    monkeypatch.setattr(backends, "find_spec", lambda name: None)
    with pytest.raises(backends.NoGPU, match="unavailable"):
        backends.gpu()
    monkeypatch.setattr(service, "launch_local", lambda *args: 3)
    assert service.schedule(store, env, NOW) == "skipped: 0 candidates (worker exit 3)"
    assert "daily" in service.schedule(store, env, NOW)
    assert not store.artworks(store.batch(DAY)["id"])


def test_live_slurm_requires_explicit_authorization(setup):
    env, store = setup
    env.settings.scheduler.mode = "slurm"
    with patch("subprocess.run", side_effect=AssertionError("must not reach SSH")):
        with pytest.raises(ValueError, match="HPC_AUTHORIZED"):
            Slurm(env)
        assert "disabled" in service.schedule(store, env, NOW)
        assert "APPROVED_PARTITION" in service.dry_run(env, NOW)["slurm_example"]


def test_default_hpc_mode_never_starts_a_local_gpu(tmp_path):
    env = Environment(tmp_path, values={}, load_config=False)
    store = Store(env.state)
    assert env.settings.scheduler.mode == "slurm"
    assert env.settings.scheduler.runtime_minutes == 30
    with patch.object(service, "launch_local", side_effect=AssertionError("must not start local GPU")), \
            patch("subprocess.run", side_effect=AssertionError("must not reach SSH")):
        assert "disabled" in service.schedule(store, env, NOW)
        with pytest.raises(ValueError, match="Local GPU generation is disabled"):
            service.start_local(store, env, NOW, DAY, 8)
    assert store.batch() is None


def test_slurm_runtime_and_single_gpu(setup):
    env, _ = setup
    with pytest.raises(ValueError):
        Schedule(runtime_minutes=31)
    dry = service.dry_run(env, NOW)["slurm_example"]
    assert "--time=00:30:00" in dry and "--gres=gpu:a100:1" in dry and "--no-requeue" in dry
    assert "--export=PATH=/usr/bin:/bin" in dry and "--export=NIL" not in dry
    assert "--deadline=2026-11-22T10:40:00" in dry
    with pytest.raises(ValueError, match="30 minutes"):
        Slurm(env, dry=True).arguments(SimpleNamespace(id="a" * 32), NOW, 31)
    script = (Environment().root / "config/slurm.sbatch").read_text()
    assert "#SBATCH --time=00:30:00" in script and "#SBATCH --gres=gpu:a100:1" in script
    assert "nvidia-smi" not in script
    result = subprocess.run(["bash", "-n", "config/slurm.sbatch"], capture_output=True)
    assert result.returncode == 0


def test_slurm_17_query_filters_by_remote_account(setup):
    env, _ = setup
    client = Slurm(env, dry=True)
    plan = SimpleNamespace(id="a" * 32)
    with patch.object(client, "ssh", return_value=SimpleNamespace(
            returncode=0, stdout=b"12345\n", stderr=b"")) as ssh:
        assert client.active(plan) == ["12345"]
    command = ssh.call_args.args[0]
    assert command[:2] == ["sh", "-c"]
    assert '--user="$(id -u)"' in command[2]
    assert "--me" not in command[2]
    assert f"--name=ac-{plan.id}" in command[2]


def test_caviness_workgroup_preserves_nested_command_arguments(setup):
    env, _ = setup
    env.values.update(HPC_WORKGROUP="weiqian_stat", HPC_REMOTE_ROOT="/work/a project")
    client = Slurm(env, dry=True)
    inner = ['sh', '-c', 'printf "%s\\n" "a b"; id -u']
    remote = shlex.split(client.ssh_command(inner)[-1])
    assert remote[:4] == ["workgroup", "--group", "weiqian_stat", "--command"]
    assert remote[4] == "cd '/work/a project' && exec " + shlex.join(inner)
    env.values["HPC_WORKGROUP"] = "bad;command"
    with pytest.raises(ValueError, match="Invalid Slurm"):
        Slurm(env, dry=True)


class FakeSlurm:
    def __init__(self, remote):
        self.remote = remote
        self.calls = 0
        self.jobs = []
        self.canceled = []
        self.accept = False

    def stage(self, plan):
        stage_request(self.remote, plan.model_dump_json().encode())

    def submit(self, plan, deadline, runtime, admit_deadline):
        self.calls += 1
        self.plan = plan
        assert runtime <= 30
        self.jobs = ["17"] if self.accept else []
        return self.jobs[0] if self.jobs else None

    def active(self, plan):
        return self.jobs[:]

    def cancel(self, jobs):
        self.canceled += jobs
        self.jobs.clear()

    def retrieve(self, plan):
        return export_request(self.remote, plan.id)


def setup_slurm(tmp_path, accept=False):
    env = Environment(tmp_path, values={k: "true" for k in ("HPC_ENABLED", "HPC_AUTHORIZED", "HPC_OPPORTUNISTIC")},
                      load_config=False)
    env.settings.scheduler.mode = "slurm"
    client = FakeSlurm(Store(tmp_path / "remote"))
    client.accept = accept
    return env, Store(env.state), client


def test_slurm_unavailable_skips_and_never_resubmits(tmp_path):
    env, store, client = setup_slurm(tmp_path)
    assert "pending reconciliation" in service.schedule(store, env, NOW, client)
    assert "already recorded" in service.schedule(store, env, NOW + timedelta(minutes=1), client)
    assert "already recorded" in service.schedule(store, env, NOW + timedelta(hours=3), client)
    with store.connect() as db:
        found = db.execute("SELECT state FROM schedules WHERE day=?", (DAY,)).fetchone()
    assert found["state"] == "skipped"
    assert client.calls == 1


def test_expired_slurm_job_is_cancelled_and_partial_results_are_retrieved(tmp_path):
    env, store, client = setup_slurm(tmp_path, accept=True)
    assert "Slurm job 17" in service.schedule(store, env, NOW, client)
    plan = client.plan
    slot = plan.slots[0]
    seed = slot.seeds[0]
    image = backends.mock_image(slot.prompt, seed, slot.width, slot.height)
    path = worker.directory(client.remote, plan) / f"01-{seed}.png"
    assert worker.persist_candidate(client.remote, plan, 1, seed, slot.prompt, image, path)
    later = NOW + timedelta(hours=3)
    service.schedule(store, env, later, client)
    assert client.canceled == ["17"]
    service.schedule(store, env, later + timedelta(minutes=1), client)
    assert client.calls == 1 and store.batch(DAY)["state"] == "partial"
    assert len(store.artworks(plan.id)) == 1
    assert import_results(store, plan, client.retrieve(plan)) == 1


def test_checkpoint_resume_does_not_repeat_completed_candidates(setup, monkeypatch):
    env, store = setup
    plan, _ = make_plan(store, env, DAY, 3, "local")
    handles = []

    def interrupted(prompt, seed, w, h):
        if handles:
            raise InterruptedError("injected preemption")
        handles.append(seed)
        return backends.mock_image(prompt, seed, w, h)

    monkeypatch.setattr(worker, "renderer", lambda *args: interrupted)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=2)
    with pytest.raises(InterruptedError, match="preemption"):
        worker.run(store, env, plan, deadline)
    assert store.batch(DAY)["state"] == "partial"
    first = store.artworks(plan.id)[0]

    def finish(prompt, seed, w, h):
        assert seed not in plan.slots[0].seeds
        return backends.mock_image(prompt, seed, w, h)

    monkeypatch.setattr(worker, "renderer", lambda *args: finish)
    assert worker.run(store, env, plan, deadline) == 3
    assert worker.run(store, env, plan, deadline) == 3
    assert store.artworks(plan.id)[0] == first


def test_duplicate_and_blank_images_rejected(setup):
    from PIL import Image
    env, store = setup
    plan, _ = make_plan(store, env, DAY, 3, "local")
    slot = plan.slots[0]
    path = worker.directory(store, plan)
    seed = slot.seeds[0]
    assert not worker.persist_candidate(store, plan, 1, seed, slot.prompt,
                                        Image.new("RGB", (1024, 1024)), path / "blank.png")
    image = backends.mock_image(slot.prompt, seed, slot.width, slot.height)
    assert worker.persist_candidate(store, plan, 1, seed, slot.prompt, image, path / "one.png")
    assert not worker.persist_candidate(store, plan, 2, seed, slot.prompt, image, path / "two.png")


def test_png_written_before_database_commit_recovers(setup):
    env, store = setup
    plan, _ = make_plan(store, env, DAY, 2, "local")
    slot = plan.slots[0]
    seed = slot.seeds[0]
    folder = worker.directory(store, plan)
    folder.mkdir(parents=True)
    image = backends.mock_image(slot.prompt, seed, slot.width, slot.height)
    (folder / f"01-{seed}.png").write_bytes(worker.png(image))
    worker.recover_written(store, plan)
    assert len(store.artworks(plan.id)) == 1
    worker.recover_written(store, plan)
    assert len(store.artworks(plan.id)) == 1


def test_slurm_resume_uses_new_day_quota(tmp_path):
    env, store, client = setup_slurm(tmp_path, accept=True)
    service.schedule(store, env, NOW, client)
    plan = client.plan
    slot = plan.slots[0]
    seed = slot.seeds[0]
    image = backends.mock_image(slot.prompt, seed, slot.width, slot.height)
    worker.persist_candidate(client.remote, plan, 1, seed, slot.prompt, image,
                             worker.directory(client.remote, plan) / f"01-{seed}.png")
    client.jobs.clear()
    service.schedule(store, env, NOW + timedelta(hours=3), client)
    tomorrow = NOW + timedelta(days=1)
    assert "queued" in service.schedule(store, env, tomorrow, client, resume_day=DAY)
    assert client.calls == 2 and client.plan.id == plan.id
    assert "already recorded" in service.schedule(store, env, tomorrow, client, resume_day=DAY)
    assert client.calls == 2


def test_daily_claim_is_atomic_across_connections(setup):
    from concurrent.futures import ThreadPoolExecutor
    env, store = setup
    plan, _ = make_plan(store, env, DAY, 1, "local")
    with ThreadPoolExecutor(max_workers=2) as threads:
        claims = list(threads.map(lambda _: service.claim(store, DAY, plan, "local", NOW, "running"), range(2)))
    assert sorted(claims) == [False, True]

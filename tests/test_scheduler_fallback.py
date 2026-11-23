from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from artcurator.scheduler import service
from artcurator.scheduler.slurm import Slurm
from artcurator.generator import worker
from artcurator.generator.backends import mock_image
from artcurator.settings import Schedule
from test_generation_scheduler import FakeSlurm, setup_slurm

NOW = datetime(2026, 11, 23, 13, tzinfo=timezone.utc)
DAY = "2026-11-23"


class RetrySlurm(FakeSlurm):
    allocated = False
    evidence = True

    def submit(self, plan, deadline, runtime, day):
        super().submit(plan, deadline, runtime, day)
        self.last_deadline = deadline
        self.last_runtime = runtime
        self.jobs = [str(16 + self.calls)] if self.accept else []
        return self.jobs[0] if self.jobs else None

    def cancel_pending(self, plan, jobs):
        if not self.allocated:
            self.cancel(jobs)

    def unallocated_receipt(self, plan, job):
        if not self.evidence or self.allocated:
            return None
        return dict(job=job, name=f"ac-{plan.id}", state="CANCELLED", elapsed_seconds=0,
                    allocated_nodes=0, node_list="None assigned", allocated_tres="")


def setup_retry(tmp_path):
    env, store, old = setup_slurm(tmp_path, accept=True)
    env.settings.scheduler.max_submissions = 2
    client = RetrySlurm(old.remote)
    client.accept = True
    return env, store, client


def test_one_fallback_after_zero_allocation_and_never_a_third(tmp_path):
    env, store, client = setup_retry(tmp_path)
    assert "queued" in service.schedule(store, env, NOW, client)
    service.schedule(store, env, NOW + timedelta(minutes=11), client)
    assert client.canceled == ["17"] and client.calls == 1
    assert "queued" in service.schedule(store, env, NOW + timedelta(minutes=12), client)
    assert client.calls == 2 and client.last_runtime == 30
    assert client.last_deadline == datetime(2026, 11, 24, 5, tzinfo=timezone.utc)
    with store.connect() as db:
        audit = db.execute("SELECT * FROM schedule_fallbacks WHERE day=?", (DAY,)).fetchone()
    assert audit["first_job"] == "17" and audit["second_job"] == "18"
    assert json.loads(audit["first_schedule"])["state"] == "skipped"
    assert json.loads(audit["accounting"])["allocated_nodes"] == 0
    client.jobs.clear()
    assert "both daily" in service.schedule(store, env, NOW + timedelta(hours=1), client)
    assert "both daily" in service.schedule(store, env, NOW + timedelta(hours=2), client)
    assert client.calls == 2


def test_allocated_or_unproven_first_job_cannot_trigger_fallback(tmp_path):
    env, store, client = setup_retry(tmp_path)
    service.schedule(store, env, NOW, client)
    client.allocated = True
    service.schedule(store, env, NOW + timedelta(minutes=11), client)
    assert client.jobs == ["17"] and client.canceled == []
    client.jobs.clear()
    assert "does not prove" in service.schedule(store, env, NOW + timedelta(minutes=20), client)
    client.allocated = False
    client.evidence = False
    assert "does not prove" in service.schedule(store, env, NOW + timedelta(minutes=21), client)
    assert client.calls == 1


def test_ambiguous_first_submission_is_never_retried(tmp_path):
    env, store, client = setup_retry(tmp_path)
    client.accept = False
    service.schedule(store, env, NOW, client)
    service.schedule(store, env, NOW + timedelta(hours=2), client)
    assert client.calls == 1
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM schedule_fallbacks").fetchone()[0] == 0


def test_fallback_reservation_survives_staging_failure(tmp_path, monkeypatch):
    env, store, client = setup_retry(tmp_path)
    service.schedule(store, env, NOW, client)
    client.jobs.clear()

    def broken_stage(plan):
        raise RuntimeError("injected staging interruption")

    monkeypatch.setattr(client, "stage", broken_stage)
    with pytest.raises(RuntimeError, match="staging interruption"):
        service.schedule(store, env, NOW + timedelta(minutes=12), client)
    assert "already recorded" in service.schedule(store, env, NOW + timedelta(minutes=13), client)
    assert client.calls == 1
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM schedule_fallbacks").fetchone()[0] == 1


def test_existing_skipped_day_can_use_fallback_but_not_after_last_start(tmp_path):
    env, store, client = setup_retry(tmp_path)
    env.settings.scheduler.max_submissions = 1
    service.schedule(store, env, NOW, client)
    client.jobs.clear()
    service.schedule(store, env, NOW + timedelta(minutes=12), client)
    env.settings.scheduler.max_submissions = 2
    late = datetime(2026, 11, 24, 4, 40, tzinfo=timezone.utc)
    assert "no same-day fallback" in service.schedule(store, env, late, client)
    assert client.calls == 1
    # A separate historical-checkpoint retry is not inferred from a skipped day.
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM schedule_fallbacks").fetchone()[0] == 0


@pytest.mark.parametrize("fields,valid", [
    ("CANCELLED by 4578|0|0|None assigned|", True),
    ("DEADLINE|0|0|None assigned|", True),
    ("CANCELLED|0|1|r00g01|cpu=8,gres/gpu=1", False),
    ("CANCELLED|1|0|None assigned|", False),
    ("CANCELLED|0|0|None assigned|gres/gpu=1", False),
    ("RUNNING|0|0|None assigned|", False),
    ("COMPLETED|0|0|None assigned|", False),
    ("|0|0|None assigned|", False),
])
def test_accounting_requires_terminal_zero_allocation(setup, monkeypatch, fields, valid):
    env, _ = setup
    client = Slurm(env, dry=True)
    plan = SimpleNamespace(id="a" * 32)
    raw = f"17|ac-{plan.id}|{fields}\n".encode()
    monkeypatch.setattr(client, "ssh", lambda command: SimpleNamespace(returncode=0, stdout=raw, stderr=b""))
    assert bool(client.unallocated_receipt(plan, "17")) == valid


def test_accounting_rejects_missing_or_wrong_job(setup, monkeypatch):
    env, _ = setup
    client = Slurm(env, dry=True)
    plan = SimpleNamespace(id="a" * 32)
    for raw in (b"", b"17|another-job|CANCELLED|0|0|None assigned|\n"):
        monkeypatch.setattr(client, "ssh", lambda command: SimpleNamespace(returncode=0, stdout=raw, stderr=b""))
        assert client.unallocated_receipt(plan, "17") is None


def test_saved_partial_candidates_block_fallback(tmp_path):
    env, store, client = setup_retry(tmp_path)
    service.schedule(store, env, NOW, client)
    plan = client.plan
    slot = plan.slots[0]
    seed = slot.seeds[0]
    assert worker.persist_candidate(client.remote, plan, 1, seed, slot.prompt,
                                    mock_image(slot.prompt, seed, slot.width, slot.height),
                                    worker.directory(client.remote, plan) / f"01-{seed}.png")
    client.jobs.clear()
    service.schedule(store, env, NOW + timedelta(minutes=20), client)
    assert store.batch(DAY)["state"] == "partial" and client.calls == 1


def test_pending_cancellation_is_state_and_name_scoped(setup, monkeypatch):
    env, _ = setup
    client = Slurm(env, dry=True)
    commands = []

    def call(command):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(client, "ssh", call)
    client.cancel_pending(SimpleNamespace(id="a" * 32), ["17"])
    assert commands == [["scancel", "--state=PENDING", "--name=ac-" + "a" * 32, "17"]]
    with pytest.raises(ValueError):
        Schedule(max_submissions=3)

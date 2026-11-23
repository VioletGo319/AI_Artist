from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from artcurator.generator import worker
from artcurator.generator.backends import mock_image
from artcurator.generator.plan import make_plan
from artcurator.mailer import service as mail
from artcurator.mailer.format import gallery_message, mail_thumbnail
from artcurator.mailer.replies import parse_command
from artcurator.curator.publish import publish, WORK
from artcurator.preference_engine.engine import FANTASY_SCENES, inspiration
from artcurator.scheduler import service
from artcurator.scheduler.slurm import Slurm, export_request, import_results, stage_request
from artcurator.store import Store

NOW = datetime(2026, 11, 23, 13, tzinfo=timezone.utc)


def test_large_vivid_plan_retains_variety_and_explicit_direction(setup):
    env, store = setup
    env.settings.generation.mode = "time_budget"
    env.settings.generation.creative_direction = "vivid_fantasy"
    plan, _ = make_plan(store, env, NOW.date().isoformat(), env.settings.generation.scheduled_count(), "local")
    assert len(plan.slots) == 256 and plan.time_budget
    assert len({seed for slot in plan.slots for seed in slot.seeds}) == 1024
    ideas = inspiration(123, 256, [], [], .3, direction="vivid_fantasy")
    assert sum(item["theme"] in FANTASY_SCENES for item in ideas) > 128
    assert set(FANTASY_SCENES) <= {item["theme"] for item in ideas}
    assert any(item["theme"] not in FANTASY_SCENES for item in ideas)
    assert len({item["title"] for item in ideas}) == 256
    assert all("vibrant" in item["prompt"] and "muted" not in item["prompt"] for item in ideas)
    # Even a previous plan covering every pair cannot exhaust the prompt library.
    assert len(inspiration(456, 256, ideas, [], .3, direction="vivid_fantasy")) == 256


def test_time_budget_continues_past_eight_and_preserves_cleanup(setup, monkeypatch):
    env, store = setup
    env.settings.generation.mode = "time_budget"
    env.settings.generation.duplicate_distance = 0
    plan, _ = make_plan(store, env, NOW.date().isoformat(), 256, "local")
    for slot in plan.slots:
        slot.width = slot.height = 64
    clock = [0.0]

    class Clock:
        @staticmethod
        def now(tz):
            return NOW + timedelta(seconds=clock[0])

    monkeypatch.setattr(worker, "datetime", Clock)
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    draws = []

    def renderer(*args):
        clock[0] += 120  # Model loading consumes the same 30-minute budget.

        def draw(prompt, seed, width, height):
            draws.append(seed)
            clock[0] += 75
            return mock_image(prompt, seed, width, height)
        return draw

    monkeypatch.setattr(worker, "renderer", renderer)
    saved = worker.run(store, env, plan, NOW + timedelta(minutes=30))
    assert 8 < saved < 256
    assert 60 <= 1800 - clock[0] < 150
    assert len(draws) == len(set(draws))
    assert len(store.artworks(plan.id)) == saved
    # Near the deadline, no model load or extra attempt occurs on resume.
    monkeypatch.setattr(worker, "renderer", lambda *args: pytest.fail("must reserve cleanup"))
    clock[0] = 1770
    assert worker.run(store, env, plan, NOW + timedelta(minutes=30)) == saved


def test_large_batch_pages_and_three_digit_approval(setup, tmp_path, monkeypatch):
    env, store = setup
    env.settings.generation.backend = "diffusers"
    env.settings.generation.mode = "time_budget"
    env.settings.generation.duplicate_distance = 0
    plan, _ = make_plan(store, env, datetime.now(env.tz).date().isoformat(), 256, "slurm")
    for slot in plan.slots:
        slot.width = slot.height = 64
    # Persist the resized test plan on both sides; no real GPU is used.
    with store.connect(write=True) as db:
        db.execute("UPDATE batches SET plan=? WHERE id=?", (plan.model_dump_json(), plan.id))
    remote = Store(tmp_path / "remote")
    stage_request(remote, plan.model_dump_json().encode())
    for index in [*range(1, 10), 100, 256]:
        slot = plan.slots[index - 1]
        seed = slot.seeds[0]
        image = mock_image(slot.prompt, seed, 64, 64)
        assert worker.persist_candidate(remote, plan, index, seed, slot.prompt, image,
                                        worker.directory(remote, plan) / f"{index:02d}-{seed}.png")
    client = Slurm(env, dry=True)

    def fetch(action, *args):
        assert action == "hpc-export"
        options = dict(zip(args[::2], args[1::2]))
        return SimpleNamespace(returncode=0, stderr=b"", stdout=export_request(
            remote, plan.id, int(options["--after"]), int(options["--limit"])))

    monkeypatch.setattr(client, "worker", fetch)
    pages = list(client.retrieve_pages(plan))
    assert len(pages) == 2
    worker.generation_state(store, plan, "running")
    import_results(store, plan, pages[0], finalize_batch=False)
    assert mail.send(store, env, store.batch(), True) == "empty: batch still running"
    for page in pages:
        import_results(store, plan, page, finalize_batch=False)
    assert worker.finalize(store, plan) == 11
    assert mail.send(store, env, store.batch(), True) == "sent"
    mail.simulate(store, store.batch(), "100；256")
    assert mail.check(store, env, True) == (1, 2)
    assert publish(store, env, mock=True)[0] == 2
    manifest = json.loads((env.public / "gallery/manifest.json").read_text())
    assert {item["id"] for item in manifest["artworks"]} == {plan.work_id(100), plan.work_id(256)}
    assert publish(store, env, mock=True)[0] == 0
    assert parse_command("257", {100, 256}) is None
    assert WORK.fullmatch(plan.work_id(256)) and not WORK.fullmatch(plan.work_id(257))


def test_large_email_has_all_candidates_and_bounded_attachments(tmp_path):
    image = Image.effect_noise((720, 720), 100).convert("RGB")
    image.save(tmp_path / "preview.webp")
    assert len(mail_thumbnail(image)) <= 32_000
    batch = {"id": "a" * 32, "day": "2026-11-23"}
    delivery = {"recipient": "owner@example.test", "message": "<test@example.test>", "token": "nonce"}
    items = [dict(slot=i, id=f'{batch["id"]}-{i:02d}', title=f"Luminous Fantasy {i}",
                  model="Tongyi-MAI/Z-Image-Turbo", thumb="preview.webp") for i in range(1, 257)]
    message, html = gallery_message(batch, items, delivery, tmp_path, "curator@example.test")
    assert len(message.as_bytes()) < 20_000_000
    assert len(html.encode()) < 100_000
    assert html.count("cid:") == 256 and "256 — Luminous Fantasy 256" in html
    assert len([part for part in message.walk() if part.get_content_maintype() == "image"]) == 256


def test_paginated_import_failure_keeps_batch_unmailable(tmp_path):
    from test_generation_scheduler import setup_slurm

    env, store, client = setup_slurm(tmp_path, accept=True)
    env.settings.generation.mode = "time_budget"
    service.schedule(store, env, NOW, client)
    assert len(client.plan.slots) == 256 and client.calls == 1
    client.jobs.clear()

    def broken_pages(plan):
        yield export_request(client.remote, plan.id)
        raise RuntimeError("transfer interrupted")

    client.retrieve_pages = broken_pages
    with pytest.raises(RuntimeError, match="transfer interrupted"):
        service.schedule(store, env, NOW + timedelta(minutes=35), client)
    assert store.batch()["state"] == "running"
    assert client.calls == 1

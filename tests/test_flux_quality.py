from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from artcurator.generator import worker
from artcurator.generator.backends import mock_image
from artcurator.generator.plan import make_plan
from artcurator.generator.upsampler import caption_messages
from artcurator.preference_engine.engine import inspiration, THEMES
from artcurator.scheduler import service
from artcurator.settings import FLUX, Generation
from artcurator.store import atomic
from test_generation_scheduler import setup_slurm


def test_ethereal_is_compositional_and_explores_beyond_examples():
    ideas = inspiration(81, 256, [], [], .3, direction="ethereal")
    assert set(THEMES) == {item["theme"] for item in ideas}
    assert len({item["title"] for item in ideas}) == 256
    assert sum(item["theme"] in {"Game Worlds", "Storm Canyon", "Spring Snow, Warm Sun"}
               for item in ideas) < 100
    assert len({item["composition"] for item in ideas}) >= 5
    assert all("negative space" in item["prompt"] and "one clear visual anchor" in item["prompt"]
               and "Do not fill" in item["prompt"] for item in ideas)
    assert inspiration(81, 256, [], [], .3, direction="ethereal") == ideas


def test_official_pup_system_prompt_is_used_unchanged(tmp_path):
    system = "Official model-specific instructions, not a substitute prompt."
    (tmp_path / "system_prompts.json").write_text(json.dumps({"t2i": system}))
    assert caption_messages(tmp_path, "user brief") == [
        {"role": "system", "content": system}, {"role": "user", "content": "user brief"}]
    (tmp_path / "system_prompts.json").write_text('{"t2i":""}')
    with pytest.raises(ValueError):
        caption_messages(tmp_path, "user brief")


def test_flux_readiness_failure_does_not_spend_daily_claim(tmp_path):
    env, store, client = setup_slurm(tmp_path, accept=True)
    env.settings.generation.backend = "flux3"
    client.preflight = lambda: (_ for _ in ()).throw(RuntimeError("missing FLUX weights"))
    with pytest.raises(RuntimeError, match="missing FLUX"):
        service.schedule(store, env, datetime(2026, 11, 24, 15, tzinfo=timezone.utc), client)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM schedules").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM batches").fetchone()[0] == 0
    assert client.calls == 0


def test_flux_uses_expanded_captions_and_recovers_without_reexpansion(setup, monkeypatch):
    env, store = setup
    env.settings.generation = Generation(backend="flux3", upsampler_backend="transformers",
                                         creative_direction="ethereal", duplicate_distance=0)
    plan, _ = make_plan(store, env, "2026-11-24", 2, "local")
    for slot in plan.slots:
        slot.width = slot.height = 64
    assert plan.model == FLUX and plan.steps == 16 and plan.guidance is None
    assert plan.upsampler_backend == "transformers"
    monkeypatch.setattr(worker, "gpu", lambda **kwargs: None)
    monkeypatch.setattr(worker, "check_cache", lambda *args: None)
    calls = []

    def expand(env, plan, deadline, start=1):
        calls.append("pup")
        for i, slot in enumerate(plan.slots, 1):
            atomic(worker.directory(store, plan) / f"{i:02d}.prompt", f"Expanded official PUP caption {i}".encode())

    def renderer(env, plan):
        assert calls == ["pup"]
        def draw(prompt, seed, width, height):
            assert prompt.startswith("Expanded official PUP")
            return mock_image(prompt, seed, width, height)
        return draw

    monkeypatch.setattr(worker, "expand_page", expand)
    monkeypatch.setattr(worker, "renderer", renderer)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
    assert worker.run(store, env, plan, deadline) == 2
    assert all(x["upsampler_runtime"] == "transformers" and x["prompt"] != x["input_prompt"]
               for x in store.artworks(plan.id))
    monkeypatch.setattr(worker, "expand_page", lambda *args: pytest.fail("must not re-expand completed batch"))
    assert worker.run(store, env, plan, deadline) == 2


def test_failed_pup_never_falls_back_to_raw_prompts(setup, monkeypatch):
    env, store = setup
    env.settings.generation = Generation(backend="flux3", upsampler_backend="transformers")
    plan, _ = make_plan(store, env, "2026-11-24", 1, "local")
    monkeypatch.setattr(worker, "gpu", lambda **kwargs: None)
    monkeypatch.setattr(worker, "check_cache", lambda *args: None)
    monkeypatch.setattr(worker, "expand_page", lambda *args: (_ for _ in ()).throw(RuntimeError("PUP failed")))
    monkeypatch.setattr(worker, "renderer", lambda *args: pytest.fail("must not render an unexpanded prompt"))
    with pytest.raises(RuntimeError, match="PUP failed"):
        worker.run(store, env, plan, datetime.now(timezone.utc) + timedelta(minutes=5))
    assert not store.artworks(plan.id)

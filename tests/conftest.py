from datetime import datetime, timedelta, timezone
import pytest

from artcurator.generator import worker
from artcurator.generator.backends import mock_image
from artcurator.generator.plan import make_plan
from artcurator.settings import Environment
from artcurator.store import Store


@pytest.fixture
def setup(tmp_path):
    env = Environment(tmp_path, values={"ART_BACKEND": "mock", "ART_PUBLIC_DIR": "demo/public"}, load_config=False)
    env.settings.scheduler.mode = "local"
    return env, Store(env.state)


@pytest.fixture
def mock_batch(setup):
    env, store = setup
    day = datetime.now(env.tz).date().isoformat()
    plan, _ = make_plan(store, env, day, 4, "local")
    assert worker.run(store, env, plan, future()) == 4
    return env, store, store.batch()


def future():
    return datetime.now(timezone.utc) + timedelta(minutes=10)


def fake_generation(monkeypatch, env, store, count=3):
    env.settings.generation.backend = "diffusers"
    env.settings.generation.aspect_ratios = ["3:2", "9:16"]
    monkeypatch.setattr(worker, "gpu", lambda only_a100=False: None)
    monkeypatch.setattr(worker, "check_cache", lambda *args: None)
    monkeypatch.setattr(worker, "renderer", lambda *args: mock_image)
    plan, _ = make_plan(store, env, datetime.now(env.tz).date().isoformat(), count, "local")
    assert worker.run(store, env, plan, future()) == count
    return store.batch()

"""Exercise generation and checkpoint recovery under the HPC's ASCII locale."""
import os
import subprocess
import sys
from pathlib import Path


def test_unicode_prompts_generate_and_recover_in_ascii_locale(tmp_path):
    code = r'''
from datetime import datetime, timedelta, timezone
import locale
from pathlib import Path
import sys

from artcurator.generator import worker
from artcurator.generator.backends import mock_image
from artcurator.generator.plan import make_plan
from artcurator.settings import Environment, Generation
from artcurator.store import Store, atomic

assert locale.getpreferredencoding(False).lower() in ("ascii", "ansi_x3.4-1968", "us-ascii")
env = Environment(Path(sys.argv[1]), values={}, load_config=False)
env.settings.generation = Generation(backend="flux3", duplicate_distance=0)
store = Store(env.state)
plan, _ = make_plan(store, env, "2026-11-24", 2, "local")
for slot in plan.slots:
    slot.width = slot.height = 64
prompt = "Mist \u2014 floating gardens, \u7a7a\u7075 composition and warm light."
folder = worker.directory(store, plan)
for i in (1, 2):
    atomic(folder / f"{i:02d}.prompt", prompt.encode("utf-8"))
worker.gpu = lambda **kwargs: None
worker.check_cache = lambda *args: None
worker.expand_page = lambda *args: None
worker.renderer = lambda *args: mock_image
assert worker.run(store, env, plan, datetime.now(timezone.utc) + timedelta(minutes=1)) == 2
assert all(art["prompt"] == prompt for art in store.artworks(plan.id))
with store.connect(write=True) as db:
    db.execute("DELETE FROM works WHERE id=?", (plan.work_id(1),))
worker.recover_written(store, plan)
assert len(store.artworks(plan.id)) == 2
assert all(art["prompt"] == prompt for art in store.artworks(plan.id))
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "LC_ALL": "C", "LANG": "C",
                                 "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"},
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

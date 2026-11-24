from hashlib import sha256
from io import BytesIO
import json
from PIL import Image
from types import SimpleNamespace

import pytest

from artcurator.curator.publish import publish
from artcurator.curator.retention import cleanup, inventory, pixel_digest, remote_prune
from artcurator.mailer import service
from artcurator.store import Busy


def prepared(mock_batch):
    env, store, batch = mock_batch
    env.settings.retention.site_url = "https://museum.example/"
    service.send(store, env, batch, mock=True)
    service.simulate(store, batch, "01, 03")
    service.check(store, env, mock=True)
    publish(store, env, mock=True)
    def fetch(url, limit):
        raw = (env.public / url.removeprefix(env.settings.retention.site_url)).read_bytes()
        assert len(raw) <= limit
        return raw
    return env, store, batch, fetch


def test_deployed_batch_prunes_duplicates_but_preserves_audit_and_republication(mock_batch):
    env, store, batch, fetch = prepared(mock_batch)
    folder = store.root / "candidates" / batch["id"]
    assert cleanup(store, env, mock=True, fetch=fetch)["batches"][0]["action"] == "eligible"
    assert folder.is_dir()
    result = cleanup(store, env, apply=True, mock=True, fetch=fetch)
    assert result["removed_bytes"] > 0 and not folder.exists()
    assert len(store.artworks(batch["id"])) == 4  # Includes rejected metadata, never public files.
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM approvals").fetchone()[0] == 2
        assert db.execute("SELECT local_pruned,remote_pruned FROM retention_events").fetchone()[:] == (1, 1)
    assert publish(store, env, mock=True)[0] == 0
    assert len(list((env.public / "gallery/art").glob("*.png"))) == 2
    assert cleanup(store, env, apply=True, mock=True, fetch=lambda *args: pytest.fail("already pruned"))["batches"] == []


def test_unapproved_batch_is_never_deleted(mock_batch):
    env, store, batch = mock_batch
    env.settings.retention.site_url = "https://museum.example/"
    service.send(store, env, batch, mock=True)
    assert not cleanup(store, env, apply=True, mock=True,
                       fetch=lambda *args: pytest.fail("no closed batch"))["batches"]
    assert (store.root / "candidates" / batch["id"]).is_dir()


@pytest.mark.parametrize("failure", ["not_deployed", "corrupt_remote", "corrupt_local", "unknown_file", "symlink"])
def test_cleanup_fails_closed_without_a_complete_verified_public_copy(mock_batch, tmp_path, failure):
    env, store, batch, fetch = prepared(mock_batch)
    folder = store.root / "candidates" / batch["id"]
    original_fetch = fetch
    if failure == "not_deployed":
        fetch = lambda *args: b'{"version":1,"artworks":[]}'
    elif failure == "corrupt_remote":
        fetch = lambda url, limit: b"corrupt" if url.endswith(".png") else original_fetch(url, limit)
    elif failure == "corrupt_local":
        next((env.public / "gallery/art").glob("*.png")).write_bytes(b"corrupt")
    elif failure == "unknown_file":
        (folder / "keep-user-notes.txt").write_text("user data")
    else:
        target = tmp_path / "outside.txt"
        target.write_text("do not delete")
        (folder / "01.prompt").symlink_to(target)
    result = cleanup(store, env, apply=True, mock=True, fetch=fetch)
    assert result["waiting"] and folder.exists()
    assert len(list(folder.glob("*.png"))) == 4


def test_public_integrity_is_still_enforced_after_private_pruning(mock_batch):
    env, store, _, fetch = prepared(mock_batch)
    cleanup(store, env, apply=True, mock=True, fetch=fetch)
    next((env.public / "gallery/art").glob("*.png")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        publish(store, env, mock=True)


def test_remote_inventory_and_lock_guard_idempotent_pruning(mock_batch):
    _, store, batch = mock_batch
    receipt = worker_receipt(store, batch["id"])
    with store.lock("gpu-worker"), pytest.raises(Busy):
        remote_prune(store, receipt)
    with pytest.raises(ValueError, match="inventory mismatch"):
        remote_prune(store, {**receipt, "items": {}})
    assert remote_prune(store, receipt)["removed_bytes"] > 0
    assert remote_prune(store, receipt)["removed_bytes"] == 0
    assert len(store.artworks(batch["id"])) == 4


def test_remote_cleanup_failure_can_resume_after_local_pruning(mock_batch):
    env, store, batch, fetch = prepared(mock_batch)
    with store.connect(write=True) as db:
        db.execute("UPDATE batches SET route='slurm' WHERE id=?", (batch["id"],))
    client = SimpleNamespace(active=lambda plan: [], worker=lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout=b"", stderr=b"SSH unavailable"))
    first = cleanup(store, env, apply=True, mock=True, fetch=fetch, client=client)
    assert first["waiting"] and not (store.root / "candidates" / batch["id"]).exists()
    client.worker = lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=b"", stdout=json.dumps(
        dict(batch=batch["id"], pruned=True, removed_bytes=100)).encode())
    second = cleanup(store, env, apply=True, mock=True, fetch=fetch, client=client)
    assert second["batches"][0]["remote_done"]
    assert publish(store, env, mock=True)[0] == 0


def test_active_schedule_prevents_pruning(mock_batch):
    env, store, batch, fetch = prepared(mock_batch)
    with store.connect(write=True) as db:
        db.execute("INSERT INTO schedules(day,batch,mode,state) VALUES (?,?,?,?)",
                   (batch["day"], batch["id"], "slurm", "queued"))
    result = cleanup(store, env, apply=True, mock=True, fetch=fetch)
    assert result["waiting"] and (store.root / "candidates" / batch["id"]).is_dir()


def worker_receipt(store, batch):
    item = store.artworks(batch)[0]
    raw = (store.root / "candidates" / batch / item["image"]).read_bytes()
    return dict(batch=batch, items=inventory(store, batch), approved=[item["id"]],
                approved_pixels={item["id"]: pixel_digest(raw)})


@pytest.mark.parametrize("changed_pixels", [False, True])
def test_remote_pixel_proof_handles_png_reencoding_but_rejects_changed_art(mock_batch, changed_pixels):
    _, store, batch = mock_batch
    receipt = worker_receipt(store, batch["id"])
    item = store.artworks(batch["id"])[0]
    path = store.root / "candidates" / batch["id"] / item["image"]
    original = path.read_bytes()
    with Image.open(BytesIO(original)) as image:
        if changed_pixels:
            image.putpixel((0, 0), (255, 0, 255))
        image.save(path, format="PNG", compress_level=0)
    assert path.read_bytes() != original
    with store.connect(write=True) as db:
        db.execute("UPDATE works SET sha=? WHERE id=?", (sha256(path.read_bytes()).hexdigest(), item["id"]))
    if changed_pixels:
        with pytest.raises(ValueError, match="pixel/checksum mismatch"):
            remote_prune(store, receipt)
        assert path.is_file()
    else:
        assert remote_prune(store, receipt)["pruned"]


def test_remote_prune_recovers_after_files_removed_before_completion_marker(mock_batch):
    _, store, batch = mock_batch
    receipt = worker_receipt(store, batch["id"])
    remote_prune(store, receipt)
    with store.connect(write=True) as db:
        db.execute("UPDATE retention_events SET local_pruned=0 WHERE batch=?", (batch["id"],))
    assert remote_prune(store, receipt)["removed_bytes"] == 0
    with store.connect() as db:
        assert db.execute("SELECT local_pruned FROM retention_events").fetchone()[0] == 1

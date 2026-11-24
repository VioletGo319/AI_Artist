"""Remove closed-batch duplicates only after verifying deployed approved assets.

Never removes SQLite rows, credentials, model weights, public originals, or a
batch still awaiting approval. Remote pruning is a separate idempotent receipt.
"""
from hashlib import sha256
import json
import re
import shutil
import subprocess
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from artcurator.curator.publish import FIELDS, read_manifest
from artcurator.generator.worker import get_plan
from artcurator.settings import SLOT_PATTERN
from artcurator.store import utcnow

BATCH = re.compile(r"[a-f0-9]{32}")


def download(url, limit):
    request = Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "AI-Art-Curator/retention"})
    with urlopen(request, timeout=20) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Oversized deployed gallery asset.")
    return raw


def healthy(store):
    with store.connect() as db:
        if [row[0] for row in db.execute("PRAGMA quick_check")] != ["ok"]:
            raise RuntimeError("Database integrity check failed; cleanup disabled.")


def inventory(store, batch):
    if not BATCH.fullmatch(batch):
        raise ValueError("Invalid cleanup batch.")
    with store.connect() as db:
        row = db.execute("SELECT state FROM batches WHERE id=?", (batch,)).fetchone()
        if not row or row["state"] not in {"ready", "partial"}:
            raise ValueError("Cleanup requires a completed batch.")
        return {row["id"]: row["sha"] for row in db.execute(
            "SELECT id,sha FROM works WHERE batch=? ORDER BY slot", (batch,))}


def prune_files(store, batch):
    candidates = store.root / "candidates"
    folder = candidates / batch
    if any(path.is_symlink() for path in (store.root, candidates, folder)):
        raise ValueError("Cleanup refuses symbolic-link directories.")
    if not folder.exists():
        return 0
    if not folder.is_dir():
        raise ValueError("Invalid candidate directory.")
    allowed = re.compile(r"(?:" + SLOT_PATTERN + r"-[0-9]+\.png|" + batch + "-" +
                         SLOT_PATTERN + r"\.webp|" + SLOT_PATTERN + r"\.prompt|\.curator-[a-zA-Z0-9_-]+)")
    files = list(folder.iterdir())
    if any(path.is_symlink() or not path.is_file() or not allowed.fullmatch(path.name) for path in files):
        raise ValueError("Unknown files in candidate directory; nothing removed.")
    size = sum(path.stat().st_size for path in files)
    shutil.rmtree(folder)
    return size


def remote_prune(store, receipt):
    """Private SSH controller requests this only after deployment proof.

    Validate the complete retrieved inventory and the worker lock again locally;
    keep all worker DB rows so a retry cannot recreate the day's allocation.
    """
    batch = receipt.get("batch", "")
    with store.lock("gpu-worker"), store.lock("retention"):
        healthy(store)
        if not receipt.get("items") or inventory(store, batch) != receipt["items"]:
            raise ValueError("Remote cleanup inventory mismatch.")
        with store.connect(write=True) as db:
            db.execute("INSERT OR IGNORE INTO retention_events(batch,receipt,updated) VALUES (?,?,?)",
                       (batch, json.dumps(receipt), utcnow()))
        removed = prune_files(store, batch)
        with store.connect(write=True) as db:
            db.execute("UPDATE retention_events SET local_pruned=1,updated=? WHERE batch=?", (utcnow(), batch))
        return dict(batch=batch, removed_bytes=removed, pruned=True)


def cleanup(store, env, *, apply=False, mock=False, fetch=download, client=None):
    if mock and env.public == env.root / "web/public":
        raise ValueError("Mock cleanup requires an isolated public directory.")
    base = env.settings.retention.site_url.rstrip("/") + "/"
    url = urlsplit(base)
    if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("Cleanup requires the deployed site's HTTPS base URL.")
    report = dict(dry_run=not apply, removed_bytes=0, batches=[], waiting=[])
    with store.lock("scheduler"), store.lock("publish"), store.lock("gpu-worker"), store.lock("retention"):
        healthy(store)
        with store.connect() as db:
            batches = db.execute("""SELECT b.* FROM batches b JOIN deliveries d ON d.batch=b.id
                LEFT JOIN retention_events e ON e.batch=b.id
                WHERE d.medium=? AND d.status='sent' AND d.decision='approve'
                AND b.state IN ('ready','partial')
                AND (e.batch IS NULL OR e.local_pruned=0 OR (b.route='slurm' AND e.remote_pruned=0))
                ORDER BY b.day""", ("mock" if mock else "gmail",)).fetchall()
        if not batches:
            return report
        _, local = read_manifest(env.public / "gallery/manifest.json")
        deployed = json.loads(fetch(base + "gallery/manifest.json", 2_000_000))
        if deployed.get("version") != 1 or not isinstance(deployed.get("artworks"), list):
            raise ValueError("Invalid deployed manifest; nothing removed.")
        remote = {item["id"]: item for item in deployed["artworks"]}
        if len(remote) != len(deployed["artworks"]):
            raise ValueError("Duplicate deployed IDs; nothing removed.")
        for batch in batches:
            identity = batch["id"]
            with store.connect() as db:
                approved = db.execute("""SELECT w.id,w.sha,w.data,a.chosen_at FROM works w
                    JOIN approvals a ON a.work=w.id
                    JOIN replies r ON r.id=a.reply AND r.batch=w.batch AND r.result='approve'
                    WHERE w.batch=?""", (identity,)).fetchall()
                active = db.execute("SELECT 1 FROM schedules WHERE batch=? AND state IN ('staging','submitting','queued','running')",
                                    (identity,)).fetchone()
                event = db.execute("SELECT * FROM retention_events WHERE batch=?", (identity,)).fetchone()
            if not approved or active:
                report["waiting"].append(dict(batch=identity, reason="not closed or still active"))
                continue
            try:
                for row in approved:
                    item = json.loads(row["data"])
                    expected = {**{key: item[key] for key in FIELDS}, "approved_at": row["chosen_at"],
                                "image": f"gallery/art/{row['id']}.png", "thumb": f"gallery/thumbs/{row['id']}.webp"}
                    for published in (local.get(row["id"], {}), remote.get(row["id"], {})):
                        if (any(published.get(key) != value for key, value in expected.items())
                                or (not mock and (published.get("mock") or item["model"].startswith("mock/")))):
                            raise ValueError("approved work not fully deployed")
                    for kind in ("image", "thumb"):
                        path = env.public / expected[kind]
                        if path.is_symlink() or not path.is_file():
                            raise ValueError("published asset missing or symlinked")
                        raw = path.read_bytes()
                        if kind == "image" and sha256(raw).hexdigest() != row["sha"]:
                            raise ValueError("published original checksum mismatch")
                        online = fetch(base + expected[kind], len(raw))
                        if not raw or sha256(online).digest() != sha256(raw).digest():
                            raise ValueError("deployed asset checksum mismatch")
                receipt = dict(batch=identity, items=inventory(store, identity),
                               approved=[row["id"] for row in approved], site=base, verified_at=utcnow())
                if not apply:
                    report["batches"].append(dict(batch=identity, action="eligible"))
                    continue
                with store.connect(write=True) as db:
                    db.execute("INSERT OR IGNORE INTO retention_events(batch,receipt,updated) VALUES (?,?,?)",
                               (identity, json.dumps(receipt), utcnow()))
                if not event or not event["local_pruned"]:
                    report["removed_bytes"] += prune_files(store, identity)
                    preview = store.root / "mail" / f"preview-{batch['day']}.html"
                    if preview.is_file() and not preview.is_symlink():
                        preview.unlink()
                    with store.connect(write=True) as db:
                        db.execute("UPDATE retention_events SET local_pruned=1,updated=? WHERE batch=?", (utcnow(), identity))
                remote_done = batch["route"] != "slurm" or bool(event and event["remote_pruned"])
                if batch["route"] == "slurm" and env.settings.retention.remote and not remote_done:
                    from artcurator.scheduler.slurm import Slurm, checked
                    client = client or Slurm(env)
                    if client.active(get_plan(store, identity)):
                        raise ValueError("remote batch is still active")
                    result = json.loads(checked(client.worker("hpc-prune", data=json.dumps(receipt).encode()), "prune"))
                    if result.get("batch") != identity or result.get("pruned") is not True:
                        raise ValueError("remote cleanup receipt mismatch")
                    report["removed_bytes"] += result["removed_bytes"]
                    remote_done = True
                with store.connect(write=True) as db:
                    db.execute("UPDATE retention_events SET remote_pruned=?,updated=? WHERE batch=?",
                               (int(remote_done), utcnow(), identity))
                report["batches"].append(dict(batch=identity, action="pruned", remote_done=remote_done))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                report["waiting"].append(dict(batch=identity, reason=str(error)))
    return report

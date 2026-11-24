from hashlib import sha256
from io import BytesIO
import json
import os
import re
import subprocess

from PIL import Image

from artcurator.generator.worker import thumbnail
from artcurator.settings import SLOT_PATTERN
from artcurator.store import atomic, atomic_json

WORK = re.compile(r"[a-f0-9]{32}-" + SLOT_PATTERN)
FIELDS = ("id", "title", "theme", "palette", "composition", "style", "prompt", "seed", "model",
          "date", "generated_at", "steps", "width", "height")


class FilesystemGallery:
    """Approved artwork storage. An object-store implementation can provide put() later."""

    def __init__(self, gallery):
        self.gallery = gallery

    def put(self, identity, raw):
        original = self.gallery / "art" / f"{identity}.png"
        thumb = self.gallery / "thumbs" / f"{identity}.webp"
        if original.is_symlink() or thumb.is_symlink():
            raise ValueError("Public gallery files cannot be symbolic links.")
        if original.is_file():
            if original.read_bytes() != raw:
                raise ValueError(f"Public artwork collision for {identity}.")
        else:
            atomic(original, raw, 0o644)
        if not thumb.is_file():
            with Image.open(BytesIO(raw)) as image:
                atomic(thumb, thumbnail(image, 720), 0o644)
        return f"gallery/art/{identity}.png", f"gallery/thumbs/{identity}.webp"


def read_manifest(path):
    value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "artworks": []}
    if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("artworks"), list):
        raise ValueError("Invalid public gallery manifest.")
    indexed = {}
    for entry in value["artworks"]:
        identity = entry["id"]
        if (not WORK.fullmatch(identity) or identity in indexed
                or entry.get("image") != f"gallery/art/{identity}.png"
                or entry.get("thumb") != f"gallery/thumbs/{identity}.webp"):
            raise ValueError("Invalid or repeated public artwork ID.")
        indexed[identity] = entry
    return value, indexed


def publish(store, env, mock=False, push=False):
    if mock and (env.public == env.root / "web" / "public" or push or env.flag("GIT_PUSH_ENABLED")):
        raise ValueError("Mock publication requires an isolated ART_PUBLIC_DIR and Git push disabled.")
    if not mock and push and not env.flag("GIT_PUSH_ENABLED"):
        raise ValueError("Set GIT_PUSH_ENABLED=true to authorize automated Git push.")
    if not mock and push and env.public != env.root / "web" / "public":
        raise ValueError("Git publication requires the repository's standard web/public directory.")
    gallery = env.public / "gallery"
    if any(path.is_symlink() for path in (gallery, gallery / "art", gallery / "thumbs", gallery / "manifest.json")):
        raise ValueError("Public gallery paths cannot be symbolic links.")
    manifest_path = gallery / "manifest.json"
    with store.lock("publish"):
        for folder in (gallery, gallery / "art", gallery / "thumbs"):
            for path in folder.glob(".curator-*"):
                if path.is_file() and not path.is_symlink():
                    path.unlink()
        if push:
            prepare_git(env)
        manifest, indexed = read_manifest(manifest_path)
        if not mock and any(entry.get("mock") or entry.get("model", "").startswith("mock/") for entry in indexed.values()):
            raise ValueError("Production manifest contains mock artwork.")
        with store.connect() as db:
            approved = db.execute("""SELECT works.batch,works.data,works.sha,approvals.chosen_at
                FROM approvals JOIN works ON works.id=approvals.work
                JOIN replies ON replies.id=approvals.reply AND replies.batch=works.batch
                JOIN deliveries ON deliveries.batch=works.batch
                WHERE deliveries.medium=? AND deliveries.decision='approve' AND replies.result='approve'
                ORDER BY approvals.chosen_at,works.slot""", ("mock" if mock else "gmail",)).fetchall()
        added = 0
        files = {manifest_path}
        adapter = FilesystemGallery(gallery)
        for row in approved:
            item = json.loads(row["data"])
            identity = item["id"]
            if (not WORK.fullmatch(identity) or not identity.startswith(row["batch"] + "-")
                    or not re.fullmatch(SLOT_PATTERN + r"-[0-9]+\.png", item["image"])):
                raise ValueError("Invalid approved artwork path.")
            if item["mock"] and not mock:
                continue
            source = store.root / "candidates" / row["batch"] / item["image"]
            # Closed, deployed batches may have had their private duplicates pruned.
            # The immutable approved public original remains verified against SQLite.
            if not source.exists() and identity in indexed:
                source = gallery / "art" / f"{identity}.png"
            if source.is_symlink():
                raise ValueError("Approved originals cannot be symbolic links.")
            raw = source.read_bytes()
            if sha256(raw).hexdigest() != row["sha"]:
                raise ValueError(f"Approved original failed integrity check: {identity}.")
            image, thumb = adapter.put(identity, raw)
            files.update((env.public / image, env.public / thumb))
            if identity not in indexed:
                public_record = {**{key: item[key] for key in FIELDS},
                                 "approved_at": row["chosen_at"], "image": image, "thumb": thumb}
                if item["mock"]:
                    public_record["mock"] = True
                indexed[identity] = public_record
                added += 1
        if added:
            manifest["artworks"] = sorted(indexed.values(), key=lambda item: (item["date"], item["id"]), reverse=True)
            atomic_json(manifest_path, manifest, 0o644)
        elif not manifest_path.exists():
            atomic_json(manifest_path, manifest, 0o644)
        action = deliver_git(env, files) if push else "local"
        return added, action


def git(env, *args, accepted=(0,)):
    result = subprocess.run(["git", *args], cwd=env.root, capture_output=True, text=True, timeout=45,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if result.returncode not in accepted:
        raise RuntimeError(f"git {args[0]} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def prepare_git(env):
    cfg = env.settings.publish
    if git(env, "symbolic-ref", "--short", "HEAD").strip() != cfg.branch:
        raise ValueError(f"Publishing requires the checked-out {cfg.branch} branch.")
    if git(env, "diff", "--cached", "--name-only").strip():
        raise ValueError("Publishing requires an empty Git index.")
    git(env, "fetch", "--no-tags", cfg.remote, f"refs/heads/{cfg.branch}")
    baseline = git(env, "rev-parse", "FETCH_HEAD").strip()
    git(env, "merge-base", "--is-ancestor", baseline, "HEAD")
    return baseline


def deliver_git(env, files):
    cfg = env.settings.publish
    relative = {file.relative_to(env.root).as_posix() for file in files}
    baseline = git(env, "rev-parse", "FETCH_HEAD").strip()
    ahead = git(env, "rev-list", f"{baseline}..HEAD").splitlines()
    for commit in ahead:
        if (not git(env, "show", "-s", "--format=%s", commit).strip().startswith("Publish approved AI art ")
                or not set(git(env, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()) <= relative):
            raise ValueError("Existing local commits need to be reviewed and pushed separately.")
    git(env, "add", "--", *sorted(relative))
    staged = set(git(env, "diff", "--cached", "--name-only").splitlines())
    if not staged <= relative:
        raise ValueError("Git index contains unrelated paths.")
    if staged:
        from datetime import date

        git(env, "commit", "-m", f"Publish approved AI art {date.today().isoformat()}", "--", *sorted(staged))
    if not staged and not ahead:
        return "no Git changes"
    git(env, "push", cfg.remote, f"HEAD:refs/heads/{cfg.branch}")
    return "pushed"

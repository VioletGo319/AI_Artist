import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
import signal
import sqlite3
import subprocess
import sys

from artcurator.curator.publish import publish
from artcurator.generator.backends import NoGPU
from artcurator.generator.worker import expand, get_plan, run
from artcurator.mailer import service as mail
from artcurator.mailer.transport import authorize
from artcurator.scheduler.service import dry_run, reconcile_local, schedule, start_local
from artcurator.scheduler.slurm import export_request, stage_request
from artcurator.settings import Environment, MAX_CANDIDATES
from artcurator.store import Busy, Store


def arguments(argv):
    parser = argparse.ArgumentParser(description="Private generation and human-approved publication.")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Run one local daily batch.")
    gen.add_argument("--count", type=int, choices=range(1, MAX_CANDIDATES + 1))
    gen.add_argument("--resume", action="store_true", help="Resume a saved, undelivered local batch.")
    gen.add_argument("--day", type=date.fromisoformat, help="Historical checkpoint day for --resume.")
    for command in ("preview-email", "send-email", "mock-reply"):
        sub = commands.add_parser(command)
        selection = sub.add_mutually_exclusive_group()
        selection.add_argument("--latest", action="store_true", help="Default: newest available batch.")
        selection.add_argument("--day", type=date.fromisoformat)
        if command == "send-email":
            sub.add_argument("--mock", action="store_true")
        if command == "mock-reply":
            sub.add_argument("--text", required=True)
            sub.add_argument("--from", dest="sender")
    commands.add_parser("check-replies").add_argument("--mock", action="store_true")
    sub = commands.add_parser("publish")
    sub.add_argument("--approved-only", action="store_true", required=True)
    sub.add_argument("--mock", action="store_true")
    sub.add_argument("--push", action="store_true")
    sub = commands.add_parser("scheduler")
    options = sub.add_mutually_exclusive_group()
    options.add_argument("--dry-run", action="store_true")
    options.add_argument("--resume", type=date.fromisoformat, help="Use today's Slurm slot to resume an undelivered batch.")
    sub = commands.add_parser("tick", help="Run the scheduled generation, mail polling and publication on the private device.")
    sub.add_argument("--mock", action="store_true")
    commands.add_parser("status")
    commands.add_parser("gmail-auth")
    commands.add_parser("gmail-status", help="Check OAuth setup without sending mail.").add_argument(
        "--connect", action="store_true", help="Verify the OAuth mailbox with a read-only profile request.")
    commands.add_parser("hpc-stage")
    commands.add_parser("model-check", help="CPU-only FLUX dependency/cache preflight; no GPU allocation.").add_argument(
        "--dependencies-only", action="store_true")
    sub = commands.add_parser("hpc-export")
    sub.add_argument("--batch", required=True)
    sub.add_argument("--after", type=int, default=0)
    sub.add_argument("--limit", type=int, default=None)
    sub = commands.add_parser("_expand")
    sub.add_argument("--batch", required=True)
    sub.add_argument("--start", type=int, default=1)
    sub.add_argument("--limit", type=int, default=MAX_CANDIDATES)
    for command in ("_worker", "hpc-worker"):
        sub = commands.add_parser(command)
        sub.add_argument("--batch", required=True)
        sub.add_argument("--deadline", type=datetime.fromisoformat, required=True)
        if command == "hpc-worker":
            sub.add_argument("--day", type=date.fromisoformat, required=True)
    return parser.parse_args(argv)


def select_batch(store, args):
    batch = store.batch(args.day.isoformat() if args.day else None)
    if batch is None or batch["state"] not in {"ready", "partial"}:
        return None
    return batch


def report(store, env):
    with store.connect() as db:
        result = {name: db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                  for name in ("batches", "works", "deliveries", "approvals")}
        result["schedule"] = [dict(row) for row in db.execute("SELECT * FROM schedules ORDER BY day DESC LIMIT 8")]
        result["fallbacks"] = [dict(row) for row in db.execute(
            "SELECT day,first_job,second_job,deadline FROM schedule_fallbacks ORDER BY day DESC LIMIT 8")]
        result["mail"] = [dict(row) for row in db.execute("""SELECT b.day,d.medium,d.status,d.attempts,d.decision
                       FROM deliveries d JOIN batches b ON b.id=d.batch ORDER BY b.day DESC LIMIT 8""")]
    result.update(backend=env.settings.generation.backend, gmail=env.flag("GMAIL_ENABLED"),
                  hpc=env.flag("HPC_ENABLED"), push=env.flag("GIT_PUSH_ENABLED"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def tick(store, env, now, mock):
    errors = []
    try:
        result = schedule(store, env, now)
        print(result)
        if result.startswith("failed"):
            errors.append(f"scheduler: {result}")
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        errors.append(f"scheduler: {error}")
    if mock or env.flag("GMAIL_ENABLED"):
        with store.connect() as db:
            rows = db.execute("""SELECT b.* FROM batches b LEFT JOIN deliveries d ON d.batch=b.id
                WHERE b.state IN ('ready','partial') AND (d.status IS NULL OR d.status!='sent')
                ORDER BY b.day DESC""").fetchall()
        today = now.astimezone(env.tz).date()
        for batch in rows:
            if (today - date.fromisoformat(batch["day"])).days > env.settings.mail.reply_days:
                continue
            try:
                print(f'{batch["day"]}: {mail.send(store, env, batch, mock)}')
            except (RuntimeError, ValueError, OSError) as error:
                errors.append(f"mail delivery: {error}")
        try:
            print("Valid replies / approvals:", *mail.check(store, env, mock))
        except Exception as error:
            errors.append(f"mail replies: {error}")
    try:
        print("Published / Git:", *publish(store, env, mock, not mock and env.flag("GIT_PUSH_ENABLED")))
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        errors.append(f"publication: {error}")
    if errors:
        raise RuntimeError("\n".join(errors))


def interrupt_worker(_signal, _frame):
    raise InterruptedError("Worker was interrupted; completed images are on disk.")


def main(argv=None):
    args = arguments(argv)
    try:
        env = Environment()
        now = datetime.now(timezone.utc)
        command = args.command
        if command == "scheduler" and args.dry_run:
            print(json.dumps(dry_run(env, now), ensure_ascii=False, indent=2))
            return 0
        if command == "gmail-auth":
            authorize(env)
            print("Gmail OAuth token stored privately.")
            return 0
        if command == "gmail-status":
            result = mail.readiness(env, args.connect)
            print(json.dumps(result, indent=2))
            return 0 if result["token_present"] else 3
        store = Store(env.state)
        if command == "status":
            report(store, env)
        elif command == "generate":
            today = now.astimezone(env.tz).date()
            if args.day and (not args.resume or args.day > today or not store.batch(args.day.isoformat())):
                raise ValueError("--day needs --resume and an existing, non-future local checkpoint.")
            with store.lock("scheduler"):
                reconcile_local(store)
                result = start_local(store, env, now, today.isoformat(), args.count or env.settings.generation.count,
                                     args.resume, args.day.isoformat() if args.day else None)
                print(result)
                if result.startswith("failed"):
                    return 1
        elif command in {"preview-email", "send-email", "mock-reply"}:
            batch = select_batch(store, args)
            if not batch:
                print("Empty: no completed candidates.")
            elif command == "preview-email":
                print(mail.preview(store, env, batch))
            elif command == "send-email":
                result = mail.send(store, env, batch, args.mock)
                print(result)
                if result.startswith(("delivery rejected", "unknown")):
                    return 1
            else:
                print(mail.simulate(store, batch, args.text, args.sender))
        elif command == "check-replies":
            print("Valid replies / approvals:", *mail.check(store, env, args.mock))
        elif command == "publish":
            print("Published / Git:", *publish(store, env, args.mock, args.push))
        elif command == "scheduler":
            result = schedule(store, env, now, resume_day=args.resume.isoformat() if args.resume else None)
            print(result)
            if result.startswith("failed"):
                return 1
        elif command == "tick":
            tick(store, env, now, args.mock)
        elif command == "hpc-stage":
            payload = sys.stdin.buffer.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError("Oversized Slurm plan.")
            print(stage_request(store, payload).id)
        elif command == "model-check":
            from artcurator.generator.preflight import check
            print(json.dumps(check(env, dependencies_only=args.dependencies_only), indent=2))
        elif command == "hpc-export":
            sys.stdout.buffer.write(export_request(store, args.batch, args.after, args.limit))
        elif command == "_expand":
            expand(store, env, get_plan(store, args.batch), args.start, args.limit)
        elif command in {"hpc-worker", "_worker"}:
            hpc = command == "hpc-worker"
            plan = get_plan(store, args.batch)
            if hpc and (not os.getenv("SLURM_JOB_ID") or not os.getenv("CUDA_VISIBLE_DEVICES")
                        or plan.day > args.day or args.day != now.astimezone(env.tz).date() or plan.backend == "mock"):
                raise ValueError("Slurm worker needs a current, allocated A100 job and a real model.")
            if args.deadline.tzinfo is None:
                raise ValueError("A timezone-aware deadline is required.")
            signal.signal(signal.SIGTERM, interrupt_worker)
            signal.signal(signal.SIGUSR1, interrupt_worker)
            deadline = min(now + timedelta(minutes=env.settings.scheduler.runtime_minutes), args.deadline)
            if hpc and os.getenv("ART_ALLOCATION_START"):
                # Count container startup/model loading inside the allocation budget.
                allocated = datetime.fromtimestamp(int(os.environ["ART_ALLOCATION_START"]), timezone.utc)
                if allocated > now:
                    raise ValueError("Allocation start cannot be in the future.")
                deadline = min(deadline, allocated + timedelta(minutes=env.settings.scheduler.runtime_minutes))
            signal.signal(signal.SIGALRM, interrupt_worker)
            signal.setitimer(signal.ITIMER_REAL, max(.001, (deadline - now).total_seconds()))
            try:
                count = run(store, env, plan, deadline, hpc=hpc)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            print(f"{count} candidates saved")
        return 0
    except NoGPU as error:
        print(f"Skipped: {error}", file=sys.stderr)
        return 3
    except Busy as error:
        print(f"Skipped: {error}", file=sys.stderr)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

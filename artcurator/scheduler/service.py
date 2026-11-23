from datetime import datetime, time, timedelta, timezone
import json
import shlex
from types import SimpleNamespace

from artcurator.generator.plan import make_plan
from artcurator.generator.worker import finalize, generation_state, get_plan, launch_local, recover_written
from artcurator.scheduler.slurm import Slurm, import_results
from artcurator.store import Busy, utcnow


def window(env, now):
    local = now.astimezone(env.tz)
    day = local.date()
    start = datetime.combine(day, time(env.settings.scheduler.start_hour), env.tz).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time(), env.tz).astimezone(timezone.utc)
    return day.isoformat(), start, end


def dry_run(env, now):
    day, _, end = window(env, now)
    runtime = env.settings.scheduler.runtime_minutes
    deadline = min(now + timedelta(minutes=env.settings.scheduler.admission_minutes + runtime), end)
    slurm = Slurm(env, dry=True)
    example = SimpleNamespace(id="0" * 32, day=now.astimezone(env.tz).date())
    cmd = slurm.ssh_command(slurm.arguments(example, deadline, runtime))
    return dict(day=day, mode=env.settings.scheduler.mode, hpc_enabled=env.flag("HPC_ENABLED"),
                dry_run=True, slurm_example=shlex.join(cmd))


def set_status(store, day, state, detail="", job=None):
    with store.connect(write=True) as db:
        db.execute("UPDATE schedules SET state=?,detail=?,job=COALESCE(?,job) WHERE day=?",
                   (state, detail[:1000], job, day))
        if job:
            db.execute("UPDATE schedule_fallbacks SET second_job=? WHERE day=?", (job, day))


def claim(store, day, plan, mode, deadline, state):
    with store.connect(write=True) as db:
        return bool(db.execute("""INSERT OR IGNORE INTO schedules(day,batch,mode,state,deadline)
                            VALUES (?,?,?,?,?)""", (day, plan.id, mode, state, deadline.isoformat())).rowcount)


def reconcile_local(store):
    with store.connect() as db:
        pending = db.execute("SELECT * FROM schedules WHERE mode='local' AND state='running'").fetchall()
    for row in pending:
        try:
            with store.lock("gpu-worker"):
                plan = get_plan(store, row["batch"])
                recover_written(store, plan)
                count = finalize(store, plan)
                state = "ready" if count == len(plan.slots) else "partial" if count else "failed"
                set_status(store, row["day"], state, f"Recovered {count} saved candidates")
        except Busy:
            # A surviving worker keeps its own lock until it exits.
            continue


def start_local(store, env, now, day, count, resume=False, resume_day=None):
    if env.settings.scheduler.mode != "local" and env.settings.generation.backend != "mock":
        raise ValueError("Local GPU generation is disabled. Use scheduler for authorized HPC A100 jobs.")
    actual_day, _, end = window(env, now)
    work_day = resume_day or day
    plan, route = make_plan(store, env, work_day, count, "local")
    if route != "local":
        raise ValueError("This batch belongs to Slurm.")
    prior = store.batch(work_day)
    if prior["state"] == "ready":
        return f"already ready ({len(store.artworks(plan.id))} candidates)"
    if resume:
        with store.connect() as db:
            if db.execute("SELECT 1 FROM deliveries WHERE batch=?", (plan.id,)).fetchone():
                raise ValueError("A delivered batch is immutable.")
    deadline = min(now + timedelta(minutes=env.settings.scheduler.runtime_minutes), end)
    if not claim(store, actual_day, plan, "local", deadline, "running"):
        if not (resume and plan.backend == "mock"):
            return "skipped: daily GPU attempt already recorded"
        with store.connect(write=True) as db:
            state = db.execute("SELECT batch,mode FROM schedules WHERE day=?", (actual_day,)).fetchone()
            if state["batch"] != plan.id or state["mode"] != "local":
                return "skipped: daily attempt belongs to another batch"
            db.execute("UPDATE schedules SET state='running',deadline=? WHERE day=?", (deadline.isoformat(), actual_day))
    code = launch_local(env, plan, deadline)
    with store.lock("gpu-worker"):
        recover_written(store, plan)
        saved = finalize(store, plan)
    state = "ready" if saved == len(plan.slots) else "partial" if saved else "skipped" if code == 3 else "failed" if code else "empty"
    set_status(store, actual_day, state, f"Worker exit {code}; {saved} candidates")
    return f"{state}: {saved} candidates" + (f" (worker exit {code})" if code else "")


def reconcile_slurm(store, client, now, env=None):
    with store.connect() as db:
        pending = db.execute("""SELECT * FROM schedules WHERE mode='slurm'
                                AND state IN ('staging','submitting','queued') ORDER BY day""").fetchall()
    for row in pending:
        plan = get_plan(store, row["batch"])
        if row["state"] == "staging":
            set_status(store, row["day"], "skipped", "Interrupted before submission; no retry today.")
            finalize(store, plan)
            continue
        jobs = client.active(plan)
        expired = now >= datetime.fromisoformat(row["deadline"])
        if jobs:
            if expired:
                client.cancel(jobs)
            else:
                if env and env.settings.scheduler.max_submissions == 2:
                    # First attempt: ten-minute admission window. Fallback: up to
                    # the last same-day start. Never cancel a running allocation here.
                    cutoff = datetime.fromisoformat(row["deadline"]) - timedelta(
                        minutes=env.settings.scheduler.runtime_minutes)
                    if now >= cutoff:
                        client.cancel_pending(plan, jobs)
                set_status(store, row["day"], "queued", "Slurm has the job", jobs[0])
            continue
        if row["state"] == "submitting" and not expired:
            # SSH may have disconnected after Slurm accepted the request. Never submit again.
            continue
        pages = client.retrieve_pages(plan) if hasattr(client, "retrieve_pages") else [client.retrieve(plan)]
        generation_state(store, plan, "running")
        for payload in pages:
            import_results(store, plan, payload, finalize_batch=False)
        count = finalize(store, plan)
        state = "ready" if count == len(plan.slots) else "partial" if count else "skipped"
        set_status(store, row["day"], state, f"Slurm complete; {count} candidates")


def start_fallback(store, env, now, day, end, client, previous):
    if (env.settings.scheduler.max_submissions != 2 or previous["mode"] != "slurm"
            or previous["state"] != "skipped" or not previous["job"] or not previous["batch"]):
        return "skipped: daily GPU attempt already recorded"
    with store.connect() as db:
        if db.execute("SELECT 1 FROM schedule_fallbacks WHERE day=?", (day,)).fetchone():
            return "skipped: both daily submission attempts already recorded"
        if db.execute("SELECT 1 FROM deliveries WHERE batch=?", (previous["batch"],)).fetchone():
            return "skipped: delivered batches cannot be retried"
    plan = get_plan(store, previous["batch"])
    if store.artworks(plan.id):
        return "skipped: saved candidates exist; no fallback allocation"
    runtime = env.settings.scheduler.runtime_minutes
    if now + timedelta(minutes=runtime + 1) >= end:
        return "skipped: no same-day fallback execution window"
    if client.active(plan):
        return "waiting: a batch allocation still exists; no fallback"
    receipt = client.unallocated_receipt(plan, previous["job"])
    if receipt is None:
        return "skipped: accounting does not prove the first job was never allocated"
    with store.connect(write=True) as db:
        inserted = db.execute("""INSERT OR IGNORE INTO schedule_fallbacks
            (day,batch,first_job,first_schedule,accounting,deadline,created)
            VALUES (?,?,?,?,?,?,?)""", (day, plan.id, previous["job"], json.dumps(dict(previous)),
                                      json.dumps(receipt), end.isoformat(), utcnow())).rowcount
        if not inserted:
            return "skipped: fallback already reserved"
        db.execute("UPDATE schedules SET state='staging',job=NULL,deadline=?,detail=? WHERE day=?",
                   (end.isoformat(), "Fallback 2/2 reserved after verified zero allocation.", day))
        db.execute("UPDATE batches SET state='running' WHERE id=?", (plan.id,))
    # The reservation survives crashes and ambiguous SSH acceptance. Never repeat it.
    return submit_staged(store, plan, day, end, runtime, client)


def start_slurm(store, env, now, day, end, client, count, resume_day=None):
    with store.connect() as db:
        previous = db.execute("SELECT * FROM schedules WHERE day=?", (day,)).fetchone()
        other_active = db.execute("""SELECT 1 FROM schedules WHERE mode='slurm' AND day!=?
                         AND state IN ('submitting','queued') LIMIT 1""", (day,)).fetchone()
        if other_active:
            return "waiting: prior Slurm attempt is being reconciled"
    if previous:
        return start_fallback(store, env, now, day, end, client, previous)
    wait = env.settings.scheduler.admission_minutes
    minutes_left = int((end - now).total_seconds() // 60)
    runtime = min(env.settings.scheduler.runtime_minutes, minutes_left - wait)
    if env.settings.scheduler.max_submissions == 2 and runtime < env.settings.scheduler.runtime_minutes:
        runtime = 0
    if runtime < 1:
        with store.connect(write=True) as db:
            db.execute("INSERT OR IGNORE INTO schedules(day,mode,state,detail) VALUES (?,'slurm','skipped',?)",
                       (day, "No admission and execution window remains today."))
        return "skipped: no same-day window"
    deadline = min(end, now + timedelta(minutes=runtime + wait))
    if resume_day:
        with store.connect(write=True) as db:
            batch = db.execute("SELECT * FROM batches WHERE day=?", (resume_day,)).fetchone()
            if (not batch or batch["route"] != "slurm" or batch["state"] != "partial" or resume_day >= day):
                raise ValueError("Slurm resume requires an older Slurm batch with saved partial results.")
            if db.execute("SELECT 1 FROM deliveries WHERE batch=?", (batch["id"],)).fetchone():
                raise ValueError("A delivered batch is immutable.")
            db.execute("UPDATE batches SET state='running' WHERE id=?", (batch["id"],))
            db.execute("INSERT INTO schedules(day,batch,mode,state,deadline) VALUES (?,?,'slurm','staging',?)",
                       (day, batch["id"], deadline.isoformat()))
        plan = get_plan(store, batch["id"])
    else:
        plan, route = make_plan(store, env, day, count, "slurm")
        if route != "slurm" or plan.backend == "mock":
            raise ValueError("Slurm requires a fresh real-model batch.")
        if not claim(store, day, plan, "slurm", deadline, "staging"):
            return "skipped: daily GPU attempt already recorded"
    return submit_staged(store, plan, day, deadline, runtime, client)


def submit_staged(store, plan, day, deadline, runtime, client):
    try:
        client.stage(plan)
    except (RuntimeError, OSError):
        finalize(store, plan)
        set_status(store, day, "skipped", "Staging failed before submission.")
        raise
    set_status(store, day, "submitting", "Submission intent persisted before SSH.")
    job = client.submit(plan, deadline, runtime, day)
    if job:
        set_status(store, day, "queued", "Submitted once with a same-day deadline.", job)
        return f"queued: Slurm job {job}"
    return "pending reconciliation: Slurm acceptance unknown or no eligible resources"


def schedule(store, env, now, client=None, resume_day=None):
    with store.lock("scheduler"):
        reconcile_local(store)
        day, start, end = window(env, now)
        mode = env.settings.scheduler.mode
        if resume_day and mode != "slurm":
            raise ValueError("Use generate --resume --day for a local checkpoint.")
        if mode == "off":
            return "skipped: scheduler disabled"
        if mode == "slurm":
            if not all(env.flag(key) for key in ("HPC_ENABLED", "HPC_AUTHORIZED", "HPC_OPPORTUNISTIC")):
                return "skipped: authorized opportunistic Slurm access is disabled"
            client = client or Slurm(env)
            reconcile_slurm(store, client, now, env)
        if now < start:
            return "skipped: before today's scheduler window"
        count = env.settings.generation.scheduled_count()
        if mode == "local":
            return start_local(store, env, now, day, count)
        return start_slurm(store, env, now, day, end, client, count, resume_day)

from datetime import datetime, timedelta, timezone
from email import policy
from email.message import EmailMessage
import json
import re
from secrets import token_hex

from artcurator.mailer.format import gallery_message, subject
from artcurator.mailer.replies import fresh_content, parse_command, validate_reply
from artcurator.mailer.transport import Gmail, MockMail, RejectedSend, UnknownSend, pause_for_retry, token_path
from artcurator.preference_engine.engine import preference_snapshot
from artcurator.store import atomic, utcnow

ADDRESS = re.compile(r"[a-zA-Z0-9_.+%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def addresses(env, mock):
    defaults = ("curator@example.test", "owner@example.test")
    sender = env.get("GMAIL_SENDER", defaults[0] if mock else "").casefold()
    recipient = env.get("GMAIL_RECIPIENT", defaults[1] if mock else "").casefold()
    approver = env.get("GMAIL_APPROVER", recipient).casefold()
    if not all(ADDRESS.fullmatch(address) for address in (sender, recipient, approver)):
        raise ValueError("Set valid GMAIL_SENDER, GMAIL_RECIPIENT and GMAIL_APPROVER addresses.")
    if not mock and sender == approver and recipient != sender:
        raise ValueError("Self-approval requires the sender, recipient and approver to be the same mailbox.")
    return sender, recipient, approver


def readiness(env, connect=False):
    """Report local setup, or verify the OAuth profile without reading/sending mail."""
    sender, recipient, approver = addresses(env, False)
    client = env.path("GMAIL_CLIENT_SECRET", "private/oauth-client.json")
    if client.is_relative_to(env.public):
        raise ValueError("OAuth credentials must stay outside the public directory.")
    token = token_path(env)
    result = dict(enabled=env.flag("GMAIL_ENABLED"), sender=sender, recipient=recipient,
                  approver=approver, self_approval=sender == recipient == approver,
                  client_present=client.is_file(), token_present=token.is_file(),
                  connected=False)
    if connect:
        if not token.is_file():
            raise ValueError("OAuth token missing. Run gmail-auth on your private device first.")
        result.update(connected=True, mailbox=Gmail(env).mailbox)
    result["status"] = "connected" if connect else "not_verified" if token.is_file() else "setup_required"
    return result


def proposal(batch, medium, sender, recipient, approver):
    token = token_hex(16)
    return dict(batch=batch["id"], medium=medium, recipient=recipient, approver=approver, token=token,
                message=f'<{batch["id"]}.{token}@{sender.split("@")[1]}>')


def compose(store, batch, delivery, sender, preview=False):
    items = store.artworks(batch["id"])
    if not items:
        raise ValueError("This batch has no candidates.")
    return gallery_message(batch, items, delivery, store.root / "candidates" / batch["id"], sender, preview)


def preview(store, env, batch):
    sender, recipient, approver = addresses(env, True)
    value = proposal(batch, "mock", sender, recipient, approver)
    _, content = compose(store, batch, value, sender, preview=True)
    path = store.root / "mail" / f'preview-{batch["day"]}.html'
    atomic(path, content.encode())
    return path


def transport_for(store, env, mock):
    return MockMail(store) if mock else Gmail(env)


def send(store, env, batch, mock, transport=None):
    sender, recipient, approver = addresses(env, mock)
    items = store.artworks(batch["id"])
    if not mock and any(art["mock"] for art in items):
        raise ValueError("Mock candidates can only use the mock outbox.")
    if not items:
        return "empty"
    medium = "mock" if mock else "gmail"
    with store.lock("mail"):
        # Freeze the numbered set before persisting the outbox intent.
        with store.lock("gpu-worker"), store.connect(write=True) as db:
            row = db.execute("SELECT state FROM batches WHERE id=?", (batch["id"],)).fetchone()
            if row is None or row["state"] not in {"ready", "partial"}:
                return "empty: batch still running"
            db.execute("""INSERT OR IGNORE INTO deliveries(batch,medium,recipient,approver,token,message,status)
                           VALUES (:batch,:medium,:recipient,:approver,:token,:message,'pending')""",
                       proposal(batch, medium, sender, recipient, approver))
            delivery = db.execute("SELECT * FROM deliveries WHERE batch=?", (batch["id"],)).fetchone()
        if delivery["medium"] != medium:
            raise ValueError("The existing delivery uses a different transport.")
        if delivery["recipient"] != recipient or delivery["approver"] != approver:
            raise ValueError("The existing delivery is tied to different mailbox settings.")
        if delivery["status"] == "sent":
            return "already sent"
        transport = transport or transport_for(store, env, mock)
        mail, _ = compose(store, batch, delivery, sender)
        if len(mail.as_bytes()) > 20_000_000:
            raise ValueError("The message exceeds the configured safe size for Gmail.")
        stamp = delivery["sent_at"] or utcnow()
        found = transport.find(delivery["message"])
        if found is not None:
            return mark_sent(store, batch, found, stamp)
        if delivery["status"] in {"sending", "unknown"}:
            return "unknown send outcome; check Sent with the stored Message-ID"
        max_attempts = env.settings.mail.retries
        for attempt in range(delivery["attempts"] + 1, max_attempts + 1):
            with store.connect(write=True) as db:
                db.execute("UPDATE deliveries SET status='sending',attempts=?,sent_at=? WHERE batch=?",
                           (attempt, stamp, batch["id"]))
            try:
                result = transport.send(mail)
                return mark_sent(store, batch, result, stamp)
            except RejectedSend:
                with store.connect(write=True) as db:
                    db.execute("UPDATE deliveries SET status='rejected' WHERE batch=?", (batch["id"],))
                if attempt < max_attempts:
                    pause_for_retry(attempt)
            except UnknownSend:
                with store.connect(write=True) as db:
                    db.execute("UPDATE deliveries SET status='unknown' WHERE batch=?", (batch["id"],))
                raise
        return "delivery rejected; retry limit reached"


def mark_sent(store, batch, result, stamp):
    if not result.get("threadId") or not result.get("id"):
        raise UnknownSend("Mail provider did not return a thread and message ID.")
    with store.connect(write=True) as db:
        db.execute("UPDATE deliveries SET status='sent',thread=?,provider_id=?,sent_at=? WHERE batch=?",
                   (result["threadId"], result["id"], stamp, batch["id"]))
    return "sent"


def check(store, env, mock, transport=None, *, recheck_ignored=False):
    medium = "mock" if mock else "gmail"
    minimum = (datetime.now(timezone.utc) - timedelta(days=env.settings.mail.reply_days)).isoformat()
    with store.lock("mail"):
        with store.connect() as db:
            deliveries = db.execute("""SELECT deliveries.*,batches.day FROM deliveries
                JOIN batches ON batches.id=deliveries.batch
                WHERE medium=? AND status='sent' AND decision IS NULL AND sent_at >= ?""",
                                    (medium, minimum)).fetchall()
        if not deliveries:
            return 0, 0
        transport = transport or transport_for(store, env, mock)
        approvals, replies = 0, 0
        for delivery in deliveries:
            available = {item["slot"]: item for item in store.artworks(delivery["batch"])}
            after = int(datetime.fromisoformat(delivery["sent_at"]).timestamp() * 1000)
            # Sort by Gmail's receipt time so the first valid reply closes the batch.
            for identity, thread, timestamp, mail, authenticated_sender, original_message_id in sorted(
                    transport.replies(delivery), key=lambda row: (row[2], row[0])):
                with store.connect() as db:
                    previous = db.execute("SELECT batch,result FROM replies WHERE id=?", (identity,)).fetchone()
                    if previous and (not recheck_ignored or previous["result"] != "ignored"
                                     or previous["batch"] != delivery["batch"]):
                        continue
                try:
                    valid = (thread == delivery["thread"] and timestamp >= after
                             and validate_reply(mail, delivery, medium, env.settings.mail.require_dmarc,
                                                authenticated_sender=authenticated_sender,
                                                original_message_id=original_message_id))
                    command = parse_command(fresh_content(mail), set(available)) if valid else None
                except (ValueError, LookupError, TypeError, AttributeError):
                    command = None
                with store.connect(write=True) as db:
                    result = command[0] if command else "ignored"
                    if previous:
                        # An explicit operator retry still runs every sender,
                        # thread, batch and command check against the real mail.
                        db.execute("UPDATE replies SET result=? WHERE id=? AND batch=? AND result='ignored'",
                                   (result, identity, delivery["batch"]))
                    else:
                        db.execute("INSERT INTO replies(id,batch,result) VALUES (?,?,?)",
                                   (identity, delivery["batch"], result))
                    if not command:
                        continue
                    replies += 1
                    db.execute("UPDATE deliveries SET decision=? WHERE batch=?", (result, delivery["batch"]))
                    for index in command[1]:
                        item = available[index]
                        chosen = datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()
                        db.execute("INSERT OR IGNORE INTO approvals(work,reply,chosen_at,snapshot) VALUES (?,?,?,?)",
                                   (item["id"], identity, chosen, json.dumps(preference_snapshot(item))))
                        approvals += 1
                break
    return replies, approvals


def simulate(store, batch, command, sender=None):
    with store.connect() as db:
        delivery = db.execute("SELECT * FROM deliveries WHERE batch=? AND medium='mock' AND status='sent'",
                              (batch["id"],)).fetchone()
    if not delivery:
        raise ValueError("Send a mock delivery before simulating a reply.")
    mail = EmailMessage(policy=policy.SMTP)
    mail["From"] = sender or delivery["approver"]
    mail["To"] = "curator@example.test"
    mail["Subject"] = f'Re: {subject(batch["day"])}'
    mail["In-Reply-To"] = delivery["message"]
    mail["References"] = delivery["message"]
    mail["X-Art-Curator-Batch"] = batch["id"]
    mail.set_content(command)
    path = store.root / "mail" / "inbox" / delivery["thread"] / f"{token_hex(12)}.eml"
    atomic(path, mail.as_bytes())
    return path

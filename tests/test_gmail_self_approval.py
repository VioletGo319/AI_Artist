from base64 import urlsafe_b64encode
from datetime import datetime, timezone
from email.message import EmailMessage
from types import SimpleNamespace
import pytest

from artcurator.mailer import service
from artcurator.mailer.format import subject
from artcurator.mailer.transport import Gmail
from artcurator.settings import Environment
from artcurator.store import Store
from conftest import fake_generation
from test_mail_publish import GmailFixture


class Response:
    def __init__(self, value):
        self.value = value

    def execute(self, **kwargs):
        return self.value


def self_reply(delivery, text="APPROVE: 01", sender="xwyu@udel.edu"):
    message = EmailMessage()
    message["From"] = sender
    message["To"] = "xwyu@udel.edu"
    message["Subject"] = f'Re: {subject(delivery["day"])}'
    message["In-Reply-To"] = delivery["message"]
    message["References"] = delivery["message"]
    message.set_content(text)
    return message


def gmail_fixture(delivery, messages, mailbox="xwyu@udel.edu"):
    original = delivery["provider_id"]
    raw = {}
    entries = [{"id": original, "labelIds": ["SENT"]}]
    stamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    for index, (identity, message, labels, thread) in enumerate(messages):
        entries.append({"id": identity, "labelIds": labels})
        raw[identity] = dict(id=identity, threadId=thread, labelIds=labels,
                             internalDate=str(stamp + index),
                             raw=urlsafe_b64encode(message.as_bytes()).decode())
    client = Gmail.__new__(Gmail)
    client.mailbox = mailbox
    client.users = SimpleNamespace(
        threads=lambda: SimpleNamespace(get=lambda **kw: Response(dict(id=delivery["thread"], messages=entries))),
        messages=lambda: SimpleNamespace(get=lambda **kw: Response(raw[kw["id"]])))
    return client


def test_self_reply_requires_provider_sent_identity_and_original_thread(tmp_path, monkeypatch):
    env = Environment(tmp_path, values={key: "xwyu@udel.edu" for key in
                      ("GMAIL_SENDER", "GMAIL_RECIPIENT", "GMAIL_APPROVER")}, load_config=False)
    store = Store(env.state)
    batch = fake_generation(monkeypatch, env, store)
    assert service.addresses(env, False) == ("xwyu@udel.edu",) * 3
    sender = GmailFixture()
    assert service.send(store, env, batch, False, sender) == "sent"
    with store.connect() as db:
        delivery = dict(db.execute("SELECT deliveries.*,batches.day FROM deliveries JOIN batches ON batches.id=deliveries.batch").fetchone())
    thread = delivery["thread"]
    spoof = self_reply(delivery)
    spoof["X-Art-Curator-Authenticated-Sender"] = "xwyu@udel.edu"
    wrong_to = self_reply(delivery)
    wrong_to.replace_header("To", "other@udel.edu")
    wrong_batch = self_reply(delivery)
    wrong_batch["X-Art-Curator-Batch"] = "other"
    copied = self_reply(delivery)
    copied["Cc"] = "other@udel.edu"
    messages = [
        ("original-copy", sender.outbound, ["SENT"], thread),
        ("forged-inbound", spoof, ["INBOX"], thread),
        ("draft", self_reply(delivery), ["SENT", "DRAFT"], thread),
        ("wrong-thread", self_reply(delivery), ["SENT"], "other"),
        ("wrong-sender", self_reply(delivery, sender="other@udel.edu"), ["SENT"], thread),
        ("wrong-to", wrong_to, ["SENT"], thread),
        ("wrong-batch", wrong_batch, ["SENT"], thread),
        ("copied", copied, ["SENT"], thread),
        ("quoted", self_reply(delivery, "> APPROVE: ALL"), ["SENT"], thread),
        ("forwarded", self_reply(delivery, "APPROVE: ALL\n---------- Forwarded message ---------"), ["SENT"], thread),
    ]
    client = gmail_fixture(delivery, messages)
    assert service.check(store, env, False, client) == (0, 0)
    assert store.artworks(batch["id"])
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM approvals").fetchone()[0] == 0
    # No DMARC header is needed for a genuine self-reply authenticated by Gmail's
    # non-user-settable SENT system label in the verified OAuth mailbox.
    messages.append(("valid", self_reply(delivery), ["SENT"], thread))
    client = gmail_fixture(delivery, messages)
    assert service.check(store, env, False, client) == (1, 1)
    assert service.check(store, env, False, client) == (0, 0)


def test_other_mailbox_sent_messages_do_not_receive_self_approval_provenance():
    delivery = dict(provider_id="original", thread="thread", recipient="xwyu@udel.edu",
                    approver="xwyu@udel.edu", day="2026-11-22", message="<batch.token@udel.edu>")
    client = gmail_fixture(delivery, [("reply", self_reply(delivery), ["SENT"], "thread")],
                           mailbox="curator@gmail.com")
    assert list(client.replies(delivery)) == []


def test_readiness_requires_local_token_and_verifies_mailbox_without_sending(tmp_path, monkeypatch):
    env = Environment(tmp_path, values={key: "xwyu@udel.edu" for key in
                      ("GMAIL_SENDER", "GMAIL_RECIPIENT", "GMAIL_APPROVER")}, load_config=False)
    result = service.readiness(env)
    assert result["status"] == "setup_required"
    assert result["self_approval"] and not result["token_present"] and not result["connected"]
    with pytest.raises(ValueError, match="token missing"):
        service.readiness(env, True)
    path = tmp_path / "private/token.json"
    path.parent.mkdir()
    path.write_text("private token fixture")
    assert service.readiness(env)["status"] == "not_verified"
    monkeypatch.setattr(service, "Gmail", lambda _: SimpleNamespace(mailbox="xwyu@udel.edu"))
    result = service.readiness(env, True)
    assert result["status"] == "connected" and result["mailbox"] == "xwyu@udel.edu"
    assert "private token fixture" not in str(result)

from email.message import EmailMessage
import pytest

from artcurator.mailer import service
from artcurator.mailer.transport import parse
from artcurator.settings import Environment
from artcurator.store import Store
from conftest import fake_generation
from test_mail_publish import GmailFixture
from test_gmail_self_approval import gmail_fixture, self_reply


@pytest.mark.parametrize("previously_ignored", [False, True])
def test_gmail_rewritten_message_id_uses_verified_sent_original(tmp_path, monkeypatch, previously_ignored):
    env = Environment(tmp_path, values={"GMAIL_SENDER": "curator@gmail.com",
                      "GMAIL_RECIPIENT": "xwyu@udel.edu", "GMAIL_APPROVER": "xwyu@udel.edu"}, load_config=False)
    store = Store(env.state)
    batch = fake_generation(monkeypatch, env, store)
    sender = GmailFixture()
    assert service.send(store, env, batch, False, sender) == "sent"
    with store.connect() as db:
        delivery = dict(db.execute("SELECT deliveries.*,batches.day FROM deliveries JOIN batches ON batches.id=deliveries.batch").fetchone())
    original = parse(sender.outbound.as_bytes())
    actual_id = "<gmail-assigned-id@mail.gmail.com>"
    original.replace_header("Message-ID", actual_id)
    reply = self_reply(delivery, text="01")
    reply.replace_header("To", "curator@gmail.com")
    reply.replace_header("In-Reply-To", actual_id)
    reply.replace_header("References", actual_id)
    reply["Authentication-Results"] = "mx.google.com; dmarc=pass header.from=udel.edu"
    wrong = parse(reply.as_bytes())
    wrong.replace_header("In-Reply-To", "<unrelated@mail.gmail.com>")
    client = gmail_fixture(delivery, [("wrong", wrong, ["INBOX"], delivery["thread"]),
                           ("valid", reply, ["INBOX"], delivery["thread"])],
                           mailbox="curator@gmail.com", original_message=original)
    # The stable batch/nonce fallback reconciles an accepted send even when
    # searching for our original RFC Message-ID returns no matches.
    assert client.find(delivery["message"]) == {"id": delivery["provider_id"], "threadId": delivery["thread"]}
    if previously_ignored:
        with monkeypatch.context() as legacy:
            legacy.setattr(service, "parse_command", lambda value, available: None)
            assert service.check(store, env, False, client) == (0, 0)
        assert service.check(store, env, False, client) == (0, 0)
    assert service.check(store, env, False, client, recheck_ignored=previously_ignored) == (1, 1)
    assert service.check(store, env, False, client) == (0, 0)
    with store.connect() as db:
        assert db.execute("SELECT result FROM replies WHERE id='wrong'").fetchone()[0] == "ignored"
        assert db.execute("SELECT count(*) FROM approvals").fetchone()[0] == 1


@pytest.mark.parametrize("invalid", ["nonce", "batch", "sender", "recipient"])
def test_rewritten_parent_cannot_be_taken_from_an_unverified_original(invalid):
    batch, nonce = "a" * 32, "b" * 32
    delivery = dict(provider_id="original", thread="thread", recipient="xwyu@udel.edu",
                    approver="xwyu@udel.edu", day="2026-11-22", message=f"<{batch}.{nonce}@gmail.com>")
    original = EmailMessage()
    original["From"] = "other@gmail.com" if invalid == "sender" else "curator@gmail.com"
    original["To"] = "other@udel.edu" if invalid == "recipient" else "xwyu@udel.edu"
    original["Message-ID"] = "<gmail-assigned-id@mail.gmail.com>"
    original["X-Art-Curator-Batch"] = "c" * 32 if invalid == "batch" else batch
    original.set_content(f'Batch ID: {batch} · {"d" * 32 if invalid == "nonce" else nonce}')
    client = gmail_fixture(delivery, [], mailbox="curator@gmail.com", original_message=original)
    with pytest.raises(ValueError, match="Cannot verify"):
        list(client.replies(delivery))
    if invalid != "recipient":
        assert client.find(delivery["message"]) is None

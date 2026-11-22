from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
import json
import subprocess

import pytest

from artcurator.curator.publish import publish
from artcurator.mailer import service
from artcurator.mailer.format import subject
from artcurator.mailer.replies import fresh_content, parse_command
from artcurator.mailer.transport import IncomingReply, MockMail, RejectedSend, UnknownSend, parse
from artcurator.settings import Environment
from artcurator.store import Store

from conftest import fake_generation


@pytest.mark.parametrize(("text", "result"), [
    ("APPROVE: 01, 03, 03", ("approve", {1, 3})), ("approve: all", ("approve", {1, 3, 4})),
    ("SKIP", ("skip", set())), ("APPROVE: 01, 02", None), ("APPROVE: 0", None),
    ("approve 1", None), ("APPROVE: 1; 3", ("approve", {1, 3})), ("APPROVE: 1, ", None),
    ("03", ("approve", {3})), ("01,03", ("approve", {1, 3})),
    ("01；03", ("approve", {1, 3})), ("01，04", ("approve", {1, 4})),
    ("1; 3，04", ("approve", {1, 3, 4})), ("0", None), ("05", None),
    ("1 3", None), ("1/3", None), ("1;", None), ("0001", None),
    ("ALL", None), ("Please publish 03", None),
])
def test_command_numbers(text, result):
    assert parse_command(text, {1, 3, 4}) == result


def test_email_contains_real_inline_thumbnails_and_day(mock_batch):
    env, store, batch = mock_batch
    html = service.preview(store, env, batch).read_text()
    assert "APPROVE: ALL" in html and "SKIP" in html and html.count("data:image/") == 4
    assert service.send(store, env, batch, mock=True) == "sent"
    files = list((store.root / "mail/outbox").glob("*.eml"))
    message = parse(files[0].read_bytes())
    assert str(message["Subject"]) == subject(batch["day"])
    assert str(message["X-Art-Curator-Batch"]) == batch["id"]
    assert sum(part.get_content_maintype() == "image" for part in message.walk()) == 4
    assert "cid:" in message.get_body(("html",)).get_content()
    assert service.send(store, env, batch, mock=True) == "already sent"
    assert len(list((store.root / "mail/outbox").glob("*.eml"))) == 1


def test_reply_rejects_wrong_sender_forged_batch_and_quote(mock_batch):
    env, store, batch = mock_batch
    service.send(store, env, batch, mock=True)
    service.simulate(store, batch, "APPROVE: ALL", "someone@example.test")
    service.simulate(store, batch, "Hello\n> APPROVE: ALL")
    forged = service.simulate(store, batch, "APPROVE: ALL")
    transmitted = parse(forged.read_bytes())
    transmitted.replace_header("X-Art-Curator-Batch", "0" * 32)
    forged.write_bytes(transmitted.as_bytes())
    assert service.check(store, env, mock=True) == (0, 0)
    assert publish(store, env, mock=True)[0] == 0
    assert json.loads((env.public / "gallery/manifest.json").read_text())["artworks"] == []


def test_quoted_html_ignored():
    incoming = EmailMessage(policy=policy.SMTP)
    incoming.set_content('<html><body><div>APPROVE: 01</div><div class="gmail_quote">'
                         '<div>APPROVE: ALL</div></div></body></html>', subtype="html")
    assert fresh_content(incoming) == "APPROVE: 01"
    incoming.set_content("APPROVE: 03\n\nOn Nov 22 Museum wrote:\n> APPROVE: ALL")
    assert fresh_content(incoming) == "APPROVE: 03"
    incoming.set_content("APPROVE: ALL\n\n---------- Forwarded message ---------")
    assert fresh_content(incoming) is None


def test_approval_only_mock_isolation_and_idempotence(mock_batch):
    env, store, batch = mock_batch
    service.send(store, env, batch, mock=True)
    service.simulate(store, batch, "APPROVE: 01, 03")
    service.simulate(store, batch, "APPROVE: 01, 03")
    assert service.check(store, env, mock=True) == (1, 2)
    assert service.check(store, env, mock=True) == (0, 0)
    assert publish(store, env, mock=True)[0] == 2
    target = env.public / "gallery/manifest.json"
    stamp = target.stat().st_mtime_ns
    assert publish(store, env, mock=True)[0] == 0
    assert target.stat().st_mtime_ns == stamp
    assert {entry["id"][-2:] for entry in json.loads(target.read_text())["artworks"]} == {"01", "03"}
    assert len(list((env.public / "gallery/art").iterdir())) == 2
    with pytest.raises(ValueError, match="mock artwork"):
        publish(store, env)
    env.public = env.root / "web/public"
    with pytest.raises(ValueError, match="isolated"):
        publish(store, env, mock=True)
    assert publish(store, env)[0] == 0


def test_skip_is_terminal(mock_batch):
    env, store, batch = mock_batch
    service.send(store, env, batch, mock=True)
    service.simulate(store, batch, "SKIP")
    assert service.check(store, env, mock=True) == (1, 0)
    service.simulate(store, batch, "APPROVE: ALL")
    assert service.check(store, env, mock=True) == (0, 0)


class FailingMail(MockMail):
    def __init__(self, store, cause):
        super().__init__(store)
        self.cause = cause
        self.tries = 0

    def send(self, mail):
        self.tries += 1
        if self.cause is UnknownSend:
            super().send(mail)
        raise self.cause("injected failure")


def test_mail_limits_retries(mock_batch, monkeypatch):
    env, store, batch = mock_batch
    monkeypatch.setattr(service, "pause_for_retry", lambda attempt: None)
    client = FailingMail(store, RejectedSend)
    assert "retry limit" in service.send(store, env, batch, True, client)
    assert "retry limit" in service.send(store, env, batch, True, client)
    assert client.tries == 3


def test_uncertain_acceptance_recovers_without_sending_again(mock_batch):
    env, store, batch = mock_batch
    client = FailingMail(store, UnknownSend)
    with pytest.raises(UnknownSend):
        service.send(store, env, batch, True, client)
    assert service.send(store, env, batch, True, client) == "sent"
    assert client.tries == 1


class GmailFixture:
    def find(self, key):
        return None

    def send(self, message):
        self.outbound = message
        return {"id": "google-sent-id", "threadId": "google-thread-id"}

    def replies(self, delivery):
        answer = EmailMessage(policy=policy.default)
        answer["From"] = delivery["approver"]
        answer["Subject"] = f'Re: {subject(delivery["day"])}'
        answer["In-Reply-To"] = delivery["message"]
        answer["References"] = delivery["message"]
        answer["Authentication-Results"] = "mx.google.com; dmarc=pass (p=NONE) header.from=example.test"
        answer.set_content("APPROVE: 01")
        stamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        wrong = parse(answer.as_bytes())
        wrong.replace_header("Authentication-Results", "unknown; dmarc=pass header.from=example.test")
        yield IncomingReply("missing-dmarc", delivery["thread"], stamp, wrong)
        yield IncomingReply("wrong-thread", "other-thread", stamp, answer)
        yield IncomingReply("valid-reply", delivery["thread"], stamp, answer)


def test_production_publish_to_local_git_remote(tmp_path, monkeypatch):
    env = Environment(tmp_path, values={"GMAIL_SENDER": "curator@example.test", "GMAIL_RECIPIENT": "owner@example.test",
                                        "GIT_PUSH_ENABLED": "true"}, load_config=False)
    store = Store(env.state)
    batch = fake_generation(monkeypatch, env, store)
    client = GmailFixture()
    service.send(store, env, batch, False, client)
    assert service.check(store, env, False, client) == (1, 1)

    def git_at(path, *args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, check=True, text=True).stdout

    remote = tmp_path / "remote"
    remote.mkdir()
    git_at(remote, "init", "--bare")
    git_at(tmp_path, "init")
    git_at(tmp_path, "symbolic-ref", "HEAD", "refs/heads/main")
    git_at(tmp_path, "config", "user.email", "museum@example.test")
    git_at(tmp_path, "config", "user.name", "Art Curator Test")
    env.public.mkdir(parents=True)
    (tmp_path / "README.md").write_text("repository fixture")
    (tmp_path / ".gitignore").write_text("private/\nremote/\n")
    git_at(tmp_path, "add", "README.md", ".gitignore")
    git_at(tmp_path, "commit", "-m", "initial")
    git_at(tmp_path, "remote", "add", "origin", str(remote))
    git_at(tmp_path, "push", "origin", "main")
    (tmp_path / "README.md").write_text("uncommitted changes must stay local")
    assert publish(store, env, push=True) == (1, "pushed")
    assert publish(store, env, push=True) == (0, "no Git changes")
    contents = git_at(remote, "ls-tree", "-r", "--name-only", "main")
    assert contents.count(".png") == contents.count(".webp") == 1
    assert "private/" not in contents
    assert git_at(remote, "show", "main:README.md") == "repository fixture"

from base64 import urlsafe_b64decode, urlsafe_b64encode
from email import policy
from email.parser import BytesParser
import json
import re
from secrets import token_hex
from time import sleep
from typing import NamedTuple

from artcurator.store import atomic


SCOPES = ["https://www.googleapis.com/auth/gmail.send", "https://www.googleapis.com/auth/gmail.readonly"]


class IncomingReply(NamedTuple):
    identity: str
    thread: str
    timestamp: int
    message: object
    # Provider metadata, never a MIME header or a value supplied by the sender.
    authenticated_sender: str | None = None
    # Verified on the original Sent message; Gmail may replace our Message-ID.
    original_message_id: str | None = None


class RejectedSend(RuntimeError):
    """Provider explicitly rejected the request before accepting mail."""


class UnknownSend(RuntimeError):
    """Request may have been accepted; reconcile before taking another action."""


def token_path(env):
    path = env.path("GMAIL_TOKEN", "private/token.json")
    if path.is_relative_to(env.public):
        raise ValueError("OAuth credentials must stay outside the public directory.")
    return path


def authorize(env):
    if not env.flag("GMAIL_ENABLED"):
        raise ValueError("Set GMAIL_ENABLED=true before authorizing Gmail.")
    from google_auth_oauthlib.flow import InstalledAppFlow

    path = env.path("GMAIL_CLIENT_SECRET", "private/oauth-client.json")
    if path.is_relative_to(env.public):
        raise ValueError("OAuth credentials must stay outside the public directory.")
    flow = InstalledAppFlow.from_client_secrets_file(path, SCOPES)
    credentials = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    atomic(token_path(env), credentials.to_json().encode())


class Gmail:
    name = "gmail"

    def __init__(self, env):
        if not env.flag("GMAIL_ENABLED"):
            raise ValueError("Gmail is disabled. Use --mock for local mail.")
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        import httplib2
        from google_auth_httplib2 import AuthorizedHttp

        path = token_path(env)
        credentials = Credentials.from_authorized_user_file(path, SCOPES)
        if not credentials.valid:
            credentials.refresh(Request())
            atomic(path, credentials.to_json().encode())
        self.users = build("gmail", "v1", http=AuthorizedHttp(credentials, http=httplib2.Http(timeout=30)),
                           cache_discovery=False).users()
        profile = self.users.getProfile(userId="me").execute(num_retries=2)
        self.mailbox = profile["emailAddress"].casefold()
        if self.mailbox != env.get("GMAIL_SENDER").casefold():
            raise ValueError("GMAIL_SENDER must match the OAuth mailbox.")

    def find(self, message):
        api = self.users.messages()
        parts = delivery_parts(message)
        queries = [f"rfc822msgid:{message[1:-1]}"]
        if parts:
            # Gmail can rewrite Message-ID. Search the random delivery nonce in
            # the body, then verify the exact Sent message before reconciling.
            queries.append(f'"{parts[0]}" "{parts[1]}"')
        for query in queries:
            result = api.list(userId="me", labelIds=["SENT"], q=query,
                              maxResults=10).execute(num_retries=2)
            if result.get("nextPageToken"):
                raise RuntimeError("Ambiguous Sent search; inspect the delivery before retrying.")
            hits = []
            for entry in result.get("messages", []):
                verified = self.sent_original(entry["id"], message)
                if verified:
                    raw, _, _ = verified
                    hits.append({"id": raw["id"], "threadId": raw["threadId"]})
            if len(hits) > 1:
                raise RuntimeError("More than one sent message matches the delivery.")
            if hits:
                return hits[0]
        return None

    def sent_original(self, identity, message):
        parts = delivery_parts(message)
        if parts is None:
            return None
        raw = self.users.messages().get(userId="me", id=identity, format="raw").execute(num_retries=2)
        labels = set(raw.get("labelIds", []))
        content = raw.get("raw", "")
        if (raw.get("id") != identity or not raw.get("threadId") or "SENT" not in labels
                or labels & {"DRAFT", "SPAM", "TRASH"} or len(content) > 30_000_000):
            return None
        mail = parse(urlsafe_b64decode(content + "=" * (-len(content) % 4)))
        batch, nonce = parts
        batches = mail.get_all("X-Art-Curator-Batch", [])
        ids = mail.get_all("Message-ID", [])
        body = mail.get_body(preferencelist=("plain",))
        if (mail.defects or not one_address(mail, "From", self.mailbox)
                or len(batches) != 1 or str(batches[0]).strip() != batch
                or len(ids) != 1 or not re.fullmatch(r"<[^\s<>]+@[^\s<>]+>", str(ids[0]).strip())
                or body is None):
            return None
        text = body.get_content()
        if not isinstance(text, str) or f"Batch ID: {batch} · {nonce}" not in text.splitlines():
            return None
        return raw, mail, str(ids[0]).strip()

    def send(self, mail):
        from googleapiclient.errors import HttpError

        try:
            return self.users.messages().send(
                userId="me", body={"raw": urlsafe_b64encode(mail.as_bytes()).decode()}
            ).execute(num_retries=0)
        except HttpError as error:
            if error.resp.status in {401, 403, 429}:
                raise RejectedSend(f"Gmail rejected send (HTTP {error.resp.status}).") from error
            raise UnknownSend(f"Gmail send outcome needs review (HTTP {error.resp.status}).") from error
        except Exception as error:
            raise UnknownSend("Gmail send outcome is unknown; the next run will search Sent.") from error

    def replies(self, delivery):
        thread_id = delivery["thread"]
        self_approval = delivery["recipient"] == delivery["approver"] == self.mailbox
        blocked = {"DRAFT", "SPAM", "TRASH"}
        if not self_approval:
            blocked.add("SENT")
        thread = self.users.threads().get(userId="me", id=thread_id, format="minimal").execute(num_retries=2)
        if thread.get("id") != thread_id or delivery["provider_id"] not in {
                entry["id"] for entry in thread.get("messages", [])}:
            return
        original = self.sent_original(delivery["provider_id"], delivery["message"])
        if (original is None or original[0]["threadId"] != thread_id
                or not one_address(original[1], "To", delivery["recipient"])):
            raise ValueError("Cannot verify the original Sent gallery and its delivery nonce.")
        original_message_id = original[2]
        for entry in thread.get("messages", []):
            if entry["id"] == delivery["provider_id"] or set(entry.get("labelIds", [])) & blocked:
                continue
            raw = self.users.messages().get(userId="me", id=entry["id"], format="raw").execute(num_retries=2)
            if (raw.get("id") != entry["id"] or raw.get("threadId") != thread_id
                    or set(raw.get("labelIds", [])) & blocked):
                continue
            # For self-approval, require Gmail's own SENT label on this exact
            # message, not a thread label or a sender-controlled MIME header.
            if self_approval and "SENT" not in raw.get("labelIds", []):
                continue
            content = raw["raw"]
            if len(content) > 4_000_000:
                continue
            yield IncomingReply(raw["id"], thread_id, int(raw["internalDate"]), parse(
                urlsafe_b64decode(content + "=" * (-len(content) % 4))
            ), self.mailbox if self_approval else None, original_message_id)


def delivery_parts(message):
    matched = re.fullmatch(r"<([a-f0-9]{32})\.([a-f0-9]{32})@[^\s<>]+>", message)
    return (matched[1], matched[2]) if matched else None


def one_address(mail, field, expected):
    headers = mail.get_all(field, [])
    return (len(headers) == 1 and not headers[0].defects and len(headers[0].addresses) == 1
            and headers[0].addresses[0].addr_spec.casefold() == expected)


def parse(raw):
    return BytesParser(policy=policy.default).parsebytes(raw)


class MockMail:
    name = "mock"

    def __init__(self, store):
        self.folder = store.root / "mail"

    def find(self, message):
        identity = message[1:-1].split("@")[0]
        if (self.folder / "outbox" / f"{identity}.eml").is_file():
            return {"id": identity, "threadId": f"mock-{identity}"}
        return None

    def send(self, mail):
        message = str(mail["Message-ID"])
        identity = message[1:-1].split("@")[0]
        atomic(self.folder / "outbox" / f"{identity}.eml", mail.as_bytes())
        return self.find(message)

    def replies(self, delivery):
        folder = self.folder / "inbox" / delivery["thread"]
        for path in sorted(folder.glob("*.eml")):
            yield IncomingReply(f"mock:{path.stem}", delivery["thread"],
                                int(path.stat().st_mtime * 1000), parse(path.read_bytes()))


def pause_for_retry(attempt):
    sleep(2 ** (attempt - 1))

from base64 import urlsafe_b64decode, urlsafe_b64encode
from email import policy
from email.parser import BytesParser
import json
from secrets import token_hex
from time import sleep

from artcurator.store import atomic


SCOPES = ["https://www.googleapis.com/auth/gmail.send", "https://www.googleapis.com/auth/gmail.readonly"]


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
        if profile["emailAddress"].casefold() != env.get("GMAIL_SENDER").casefold():
            raise ValueError("GMAIL_SENDER must match the OAuth mailbox.")

    def find(self, message):
        api = self.users.messages()
        result = api.list(userId="me", labelIds=["SENT"], q=f"rfc822msgid:{message[1:-1]}",
                          maxResults=10).execute(num_retries=2)
        hits = result.get("messages", [])
        if len(hits) > 1:
            raise RuntimeError("More than one sent message matches the delivery.")
        return hits[0] if hits else None

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
        thread = self.users.threads().get(userId="me", id=thread_id, format="minimal").execute(num_retries=2)
        if thread.get("id") != thread_id or delivery["provider_id"] not in {
                entry["id"] for entry in thread.get("messages", [])}:
            return
        for entry in thread.get("messages", []):
            if entry["id"] == delivery["provider_id"] or set(entry.get("labelIds", [])) & {
                    "SENT", "DRAFT", "SPAM", "TRASH"}:
                continue
            raw = self.users.messages().get(userId="me", id=entry["id"], format="raw").execute(num_retries=2)
            if (raw.get("id") != entry["id"] or raw.get("threadId") != thread_id
                    or set(raw.get("labelIds", [])) & {"SENT", "DRAFT", "SPAM", "TRASH"}):
                continue
            content = raw["raw"]
            if len(content) > 4_000_000:
                continue
            yield raw["id"], thread_id, int(raw["internalDate"]), parse(
                urlsafe_b64decode(content + "=" * (-len(content) % 4))
            )


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
            yield f"mock:{path.stem}", delivery["thread"], int(path.stat().st_mtime * 1000), parse(path.read_bytes())


def pause_for_retry(attempt):
    sleep(2 ** (attempt - 1))

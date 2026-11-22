from html.parser import HTMLParser
import re


APPROVE = re.compile(r"APPROVE\s*:\s*(ALL|\d{1,2}(?:\s*,\s*\d{1,2})*)", re.I)
QUOTED = re.compile(r"^(?:>|On .{1,400}wrote:\s*$|[-_]{2,}\s*(?:original message|forwarded message)|From:\s|--\s*$)", re.I)
FORWARDED = re.compile(r"(?:forwarded message|begin forwarded)", re.I)


class ReplyHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.stopped = False

    def handle_starttag(self, tag, attrs):
        ignored = {"gmail_quote", "gmail_quote_container", "gmail_attr", "moz-cite-prefix",
                   "divRplyFwdMsg", "yahoo_quoted", "protonmail_quote"}
        attributes = dict(attrs)
        markers = set((attributes.get("class") or "").split()) | {attributes.get("id")}
        if tag in {"blockquote", "script", "style", "head"} or markers & ignored:
            if tag == "head":
                return
            self.stopped = True
        if tag in {"div", "br", "p", "li"} and not self.stopped:
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag in {"div", "p", "li"} and not self.stopped:
            self.text.append("\n")

    def handle_data(self, data):
        if not self.stopped:
            self.text.append(data)


def fresh_content(mail):
    part = mail.get_body(preferencelist=("plain", "html"))
    if part is None:
        return None
    content = part.get_content()
    if not isinstance(content, str) or len(content) > 100_000 or FORWARDED.search(content):
        return None
    if part.get_content_type() == "text/html":
        # Extract the body before parsing, so head/title never become approval text.
        content = re.sub(r"<head(?:\s[^>]*)?>.*?</head\s*>", "", content, flags=re.I | re.S)
        reader = ReplyHTML()
        reader.feed(content)
        content = "".join(reader.text)
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if QUOTED.match(line):
            break
        if line:
            lines.append(line)
    return lines[0] if len(lines) == 1 else None


def parse_command(value, available):
    if value is None:
        return None
    if value.upper() == "SKIP":
        return "skip", set()
    matched = APPROVE.fullmatch(value)
    if not matched:
        return None
    text = matched[1]
    picked = set(available) if text.upper() == "ALL" else {int(number) for number in text.split(",")}
    return ("approve", picked) if picked and picked.issubset(available) else None


def authenticated(mail, address):
    """Require Gmail's top auth result to report DMARC alignment for the From domain."""
    matches = mail.get_all("Authentication-Results", [])
    if not matches:
        return False
    result = re.sub(r"\([^()]*\)", "", str(matches[0]))
    sections = result.split(";")
    if sections[0].strip().casefold() != "mx.google.com":
        return False
    domain = address.rsplit("@", 1)[1]
    return any(re.search(r"\bdmarc=pass\b", part, re.I)
               and (value := re.search(r"\bheader\.from=([\w.-]+)", part, re.I))
               and value[1].casefold() == domain for part in sections[1:])


def validate_reply(mail, delivery, name, require_dmarc):
    if mail.defects or any(part.get_content_type() == "message/rfc822" for part in mail.walk()):
        return False
    addresses = mail.get_all("From", [])
    if len(addresses) != 1 or addresses[0].defects or len(addresses[0].addresses) != 1:
        return False
    address = addresses[0].addresses[0].addr_spec.casefold()
    if address != delivery["approver"]:
        return False
    if name == "gmail" and require_dmarc and not authenticated(mail, address):
        return False
    for key in ("Resent-From", "Resent-To", "Auto-Submitted", "X-Forwarded-For"):
        if key in mail:
            return False
    prefixes = mail.get_all("Subject", [])
    parents = mail.get_all("In-Reply-To", [])
    refs = mail.get_all("References", [])
    if (len(prefixes) != 1 or len(parents) != 1 or len(refs) != 1
            or not re.match(r"^Re:\s*", str(prefixes[0]), re.I)
            or re.search(r"\b(?:fwd|fw):", str(prefixes[0]), re.I)):
        return False
    from artcurator.mailer.format import subject

    if re.sub(r"^(?:Re:\s*)+", "", str(prefixes[0]), flags=re.I) != subject(delivery["day"]):
        return False
    if str(parents[0]).strip() != delivery["message"] or delivery["message"] not in str(refs[0]).split():
        return False
    batch_header = mail.get_all("X-Art-Curator-Batch", [])
    if batch_header and (len(batch_header) != 1 or str(batch_header[0]) != delivery["batch"]):
        return False
    return delivery["message"].startswith(f'<{delivery["batch"]}.{delivery["token"]}@')

from email.message import EmailMessage
from email import policy
from email.utils import formatdate
from html import escape
import base64

from PIL import Image

from artcurator.generator.worker import thumbnail


def subject(day):
    return f"AI Art Curator — Daily Gallery — {day}"


def gallery_message(batch, artworks, delivery, folder, sender, preview=False):
    mail = EmailMessage(policy=policy.SMTP)
    mail["From"] = sender
    mail["To"] = delivery["recipient"]
    mail["Subject"] = subject(batch["day"])
    mail["Date"] = formatdate(localtime=False)
    mail["Message-ID"] = delivery["message"]
    mail["X-Art-Curator-Batch"] = batch["id"]
    rows, embedded, plain = [], [], []
    for item in artworks:
        label = f'{item["slot"]:02d} — {item["title"]}'
        plain.append(f'{label} ({item["model"]})')
        content_id = f'{item["id"]}@artcurator.local'
        with Image.open(folder / item["thumb"]) as image:
            thumb = thumbnail(image, 400, format="JPEG")
        encoded = base64.b64encode(thumb).decode()
        url = f"data:image/jpeg;base64,{encoded}" if preview else f"cid:{content_id}"
        embedded.append((content_id, thumb))
        rows.append(f'<tr><td style="padding:18px 0;border-top:1px solid #4c464d">'
                    f'<img src="{url}" alt="{escape(label)}" width="360" '
                    f'style="width:100%;max-width:360px;height:auto;display:block;border-radius:6px">'
                    f'<p style="margin:11px 0 2px"><strong>{escape(label)}</strong></p>'
                    f'<p style="margin:0;color:#b9b1ba">{escape(item["model"])}</p></td></tr>')
    instructions = "Reply with APPROVE: 01, 03, 04, APPROVE: ALL, or SKIP."
    token = f'Batch ID: {batch["id"]} · {delivery["token"]}'
    mail.set_content("\n".join([subject(batch["day"]), "", *plain, "", instructions, token]))
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(subject(batch["day"]))}</title>
<body style="background:#141218;color:#f2ecdf;font-family:Arial,sans-serif;margin:0;padding:24px">
<div style="max-width:630px;margin:auto">
<p style="color:#cdb78b;letter-spacing:.18em;text-transform:uppercase">AI Art Curator</p>
<h1 style="font:48px Georgia,serif;margin:.2em 0">Daily Gallery</h1>
<p style="color:#ccc1ce">{escape(batch["day"])}</p>
<table role="presentation" style="width:100%;border-collapse:collapse">{''.join(rows)}</table>
<p>Reply directly to this email with one of these commands.</p>
<p style="line-height:2"><code>APPROVE: 01, 03, 04</code><br><code>APPROVE: ALL</code><br><code>SKIP</code></p>
<p style="color:#b9b1ba;font-size:13px">{escape(token)}</p>
</div></body></html>"""
    mail.add_alternative(html, subtype="html")
    if not preview:
        part = mail.get_payload()[-1]
        for cid, payload in embedded:
            part.add_related(payload, maintype="image", subtype="jpeg", cid=f"<{cid}>")
    return mail, html

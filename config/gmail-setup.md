# Gmail setup on the private controller

Run these steps on your private computer, never the HPC login node or GPU worker. The browser's Gmail connector and this standalone program have separate OAuth credentials. Connecting an account in a chat does not authorize the scheduled program.

## Authorizations

1. Use a Google Cloud project where you may enable the **Gmail API** and create OAuth credentials. In **Google Auth platform**, configure Branding, Audience, and Data Access. Select Internal only for an eligible organization-owned project permitted by its administrator. Otherwise use External and add the sending Google account as a test user during setup.
2. Request exactly these scopes:
   - `https://www.googleapis.com/auth/gmail.send`: send gallery emails.
   - `https://www.googleapis.com/auth/gmail.readonly`: read mail for reply approval and Sent reconciliation. This scope grants mailbox-wide access; Google does not offer an equivalent single-thread read scope.
3. Create an OAuth client of type **Desktop app**. Download its JSON to `private/oauth-client.json` inside the checkout. Do not paste credentials into chat or commit them. No service account, domain-wide delegation, app password, school password, or shared two-factor code is needed.
4. If the university blocks the OAuth application, ask its Workspace administrator to permit this client ID and these scopes. Do not bypass an administrator restriction. Permission to use HPC does not grant Workspace OAuth permission.

[Google setup guide](https://developers.google.com/workspace/gmail/api/quickstart/python) · [Gmail permissions](https://developers.google.com/workspace/gmail/api/auth/scopes)

For Gmail scopes, an External application in **Testing** receives a refresh token that expires after seven days. Testing is suitable for the initial check. For unattended operation, arrange the appropriate Internal or production configuration, including any verification and administrator approval Google requires. Tokens can also be revoked or expire for other reasons. [Token expiration](https://developers.google.com/identity/protocols/oauth2#expiration)

## Local setup and account verification

Use the controller's Python environment below (`python`), with the Gmail optional dependencies installed. Configure the local `.env`:

```dotenv
GMAIL_ENABLED=false
GMAIL_SENDER=xwyu@udel.edu
GMAIL_RECIPIENT=xwyu@udel.edu
GMAIL_APPROVER=xwyu@udel.edu
GMAIL_CLIENT_SECRET=private/oauth-client.json
GMAIL_TOKEN=private/token.json
```

```bash
chmod 600 private/oauth-client.json
python -m artcurator gmail-status
GMAIL_ENABLED=true python -m artcurator gmail-auth
GMAIL_ENABLED=true python -m artcurator gmail-status --connect
```

The first status command exits with code 3 until a token exists. The authorization opens Google's consent page; choose **xwyu@udel.edu** and grant both scopes. The callback listens on localhost. The token is stored with mode 600. The connection check verifies that the authorized mailbox matches `GMAIL_SENDER`; it only requests the account profile and may refresh the token. It neither reads messages nor sends a gallery. Stop and reauthorize the correct account if it fails.

## First real gallery and approval

After OAuth succeeds and real sending is authorized:

```bash
python -m artcurator preview-email --latest
GMAIL_ENABLED=true python -m artcurator send-email --latest
```

Open the original gallery in Gmail, use **Reply**, and keep **xwyu@udel.edu** as the sole recipient. Leave the subject unchanged, do not forward, remove any automatic signature, and write a single command above the quoted history:

```text
01,03
```

One number such as `05` is enough. Multiple numbers can use English or Chinese commas/semicolons: `01,03`, `01;03`, `01，03`, or `01；03`. The `APPROVE:` prefix is optional for numbered selections. `APPROVE: ALL` selects all candidates; `SKIP` closes the batch without publishing. The first valid reply closes the decision. Reply to the original gallery message, not to a previous approval. Self-replies must carry Gmail's own `SENT` message label and matching From/To; inbound messages that merely claim the same sender cannot approve. Original gallery messages and quoted instructions are excluded. Separate sender/approver configurations remain supported using DMARC-validated incoming replies.

```bash
GMAIL_ENABLED=true python -m artcurator check-replies
python -m artcurator publish --approved-only
python -m artcurator status
```

Inspect the local approved manifest. With authorized repository credentials and `GIT_PUSH_ENABLED=true`, use `python -m artcurator publish --approved-only --push` to update GitHub. Never use `--mock` against the production state for this check: simulated approvals belong in an isolated demo directory.

## Unattended operation

Only after a real send/reply cycle succeeds, set `GMAIL_ENABLED=true` in `.env` and enable the private-device scheduler from `cron.example` or an equivalent launchd service. `tick` sends undelivered completed batches, reads registered conversations, publishes approved images, and pushes when `GIT_PUSH_ENABLED=true`. It also invokes GPU scheduling: keep `HPC_ENABLED=false` during mail-only setup, and enable it only under the separately approved HPC configuration. No long-running controller belongs on the login node.

The computer must be awake and online for polling. No background service is installed by these instructions. Review failures in the scheduler log and `status`; do not delete delivery records to force a resend after an ambiguous network failure. The application reconciles Sent first to avoid duplicate deliveries.

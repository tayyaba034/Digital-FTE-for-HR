"""
tools/gmail_sender.py
Gmail integration via Google OAuth2 for sending application emails.
"""
import base64
import json
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Any
import structlog
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = structlog.get_logger()

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]


class GmailSender:
    def __init__(self):
        self.client_id = os.getenv("GMAIL_CLIENT_ID")
        self.client_secret = os.getenv("GMAIL_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GMAIL_REDIRECT_URI", "http://localhost:8000/auth/gmail/callback")
        self._credentials: dict | None = None  # stored per user session

    def _validate_oauth_config(self):
        if not self.client_id or not self.client_secret:
            raise ValueError("Set both GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET in backend/.env")

    def get_auth_url(self, state: str | None = None) -> str:
        """Generate Google OAuth2 authorization URL."""
        self._validate_oauth_config()
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uris": [self.redirect_uri],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=SCOPES,
            redirect_uri=self.redirect_uri,
        )
        auth_url, _ = flow.authorization_url(
            prompt="consent",
            access_type="offline",
            include_granted_scopes="true",
            state=state,
        )
        return auth_url

    def exchange_code(self, code: str) -> dict:
        """Exchange authorization code for credentials."""
        self._validate_oauth_config()
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uris": [self.redirect_uri],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=SCOPES,
            redirect_uri=self.redirect_uri,
        )
        flow.fetch_token(code=code)
        creds = flow.credentials
        return {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes or []),
        }

    def _build_service(self, creds_dict: dict):
        """Build Gmail API service from stored credentials."""
        creds = Credentials(
            token=creds_dict["token"],
            refresh_token=creds_dict.get("refresh_token"),
            token_uri=creds_dict.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=creds_dict.get("client_id", self.client_id),
            client_secret=creds_dict.get("client_secret", self.client_secret),
            scopes=creds_dict.get("scopes", SCOPES),
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("gmail", "v1", credentials=creds)

    def _build_message(
        self,
        to: str,
        subject: str,
        body_text: str,
        body_html: str | None = None,
        attachment_bytes: bytes | None = None,
        attachment_name: str = "CV.pdf",
    ) -> dict:
        """Build a Gmail API message object."""
        msg = MIMEMultipart("mixed")
        msg["to"] = to
        msg["subject"] = subject

        # Body
        if body_html:
            alt = MIMEMultipart("alternative")
            alt.attach(MIMEText(body_text, "plain"))
            alt.attach(MIMEText(body_html, "html"))
            msg.attach(alt)
        else:
            msg.attach(MIMEText(body_text, "plain"))

        # Optional attachment (CV)
        if attachment_bytes:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
            msg.attach(part)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        return {"raw": raw}

    async def send_email(
        self,
        creds_dict: dict,
        to: str,
        subject: str,
        body: str,
        cover_letter: str | None = None,
        cv_attachment: bytes | None = None,
        cv_filename: str = "CV.pdf",
    ) -> dict[str, Any]:
        """
        Send an application email via Gmail.
        Returns the Gmail message ID on success.
        """
        try:
            service = self._build_service(creds_dict)

            # Combine body + cover letter
            full_body = body
            if cover_letter:
                full_body = f"{body}\n\n---\n\nCOVER LETTER\n\n{cover_letter}"

            message = self._build_message(
                to=to,
                subject=subject,
                body_text=full_body,
                attachment_bytes=cv_attachment,
                attachment_name=cv_filename,
            )

            sent = service.users().messages().send(userId="me", body=message).execute()
            log.info("gmail.sent", to=to, message_id=sent["id"])
            return {"success": True, "message_id": sent["id"]}

        except HttpError as e:
            log.error("gmail.send_error", to=to, error=str(e))
            return {"success": False, "error": str(e)}

    async def check_replies(self, creds_dict: dict, query: str = "is:inbox") -> list[dict]:
        """Check for replies to sent applications."""
        try:
            service = self._build_service(creds_dict)
            results = service.users().messages().list(userId="me", q=query, maxResults=20).execute()
            messages = results.get("messages", [])

            emails = []
            for msg in messages:
                detail = service.users().messages().get(userId="me", id=msg["id"], format="metadata").execute()
                headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
                emails.append({
                    "id": msg["id"],
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                })
            return emails

        except HttpError as e:
            log.error("gmail.check_replies_error", error=str(e))
            return []


# Singleton
gmail_sender = GmailSender()

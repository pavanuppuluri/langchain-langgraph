"""Minimal Confluence Cloud client.

This example uses the Confluence Cloud v1 content API because it provides a
simple page-list + individual-page flow for incremental indexing.
"""

from __future__ import annotations

from typing import Iterator
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.models import PageSummary


class ConfluenceClient:
    def __init__(self, base_url: str, email: str, api_token: str, space_key: str):
        self.base_url = base_url.rstrip("/")
        self.space_key = space_key
        self.session = requests.Session()
        self.session.auth = (email, api_token)
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, **kwargs) -> dict:
        response = self.session.get(urljoin(self.base_url + "/", path.lstrip("/")), timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()

    def iter_page_summaries(self) -> Iterator[PageSummary]:
        """List only cheap metadata first; do NOT download page bodies here."""
        start = 0
        limit = 100

        while True:
            payload = self._get(
                "/wiki/rest/api/content",
                params={
                    "spaceKey": self.space_key,
                    "type": "page",
                    "status": "current",
                    "start": start,
                    "limit": limit,
                    "expand": "version",
                },
            )

            results = payload.get("results", [])
            for page in results:
                links = page.get("_links", {})
                webui = links.get("webui", "")
                source_url = urljoin(self.base_url + "/", webui.lstrip("/"))
                yield PageSummary(
                    document_id=str(page["id"]),
                    title=page["title"],
                    version=int(page["version"]["number"]),
                    source_url=source_url,
                )

            if len(results) < limit:
                break
            start += len(results)

    def get_page_storage_html(self, document_id: str) -> str:
        payload = self._get(
            f"/wiki/rest/api/content/{document_id}",
            params={"expand": "body.storage,version"},
        )
        return payload["body"]["storage"]["value"]


def storage_html_to_text(html: str) -> str:
    """Convert Confluence storage HTML to text while keeping useful boundaries."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["br", "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]):
        tag.insert_before("\n")

    text = soup.get_text(" ", strip=True)
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

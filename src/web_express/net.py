"""Safe page downloads.

web_express runs on a personal computer next to local services, and the URLs it
fetches come from search results and from web pages. So every download:

- resolves the host once, refuses anything that is not a public address, and
  connects to that exact address (a second DNS answer cannot redirect it);
- re-checks every redirect hop the same way;
- decides from the content type whether to read the body at all;
- stops reading at a byte limit and at an overall time limit.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urljoin, urlsplit

import httpx

from . import __version__

# An honest name, not a browser disguise. Some sites (Wikipedia among them) only
# answer clients that say how to reach their operator: set WEB_EXPRESS_CONTACT
# to a URL or email address to add one.
_CONTACT = os.environ.get("WEB_EXPRESS_CONTACT", "").strip()
USER_AGENT = f"web_express/{__version__} ({_CONTACT or 'local web search for AI agents'})"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.8",
}
MAX_TEXT_BYTES = 5_000_000
MAX_PDF_BYTES = 25_000_000
MAX_REDIRECTS = 5
CONNECT_TIMEOUT = 8.0
READ_TIMEOUT = 15.0
TOTAL_TIMEOUT = 25.0

_TEXT_TYPES = {"application/json", "application/xml", "application/xhtml+xml", "application/rss+xml", "application/atom+xml"}
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE)

Resolver = Callable[[str, int], Awaitable[list[str]]]


class FetchError(Exception):
    """A download that could not be completed; the message is safe to show."""


class BlockedAddress(FetchError):
    """The URL points at a local or private network address."""


@dataclass
class Download:
    url: str  # final URL after redirects
    status: int
    content_type: str
    body: bytes
    charset: str | None
    truncated: bool

    def text(self) -> str:
        charset = self.charset or _sniff_charset(self.body) or "utf-8"
        try:
            return self.body.decode(charset, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")


def kind_of(content_type: str) -> str | None:
    """Map a content type to 'html', 'pdf' or 'text'; None means unsupported."""
    if content_type in ("text/html", "application/xhtml+xml", ""):
        return "html"
    if content_type == "application/pdf":
        return "pdf"
    if content_type.startswith("text/") or content_type in _TEXT_TYPES or content_type.endswith(("+json", "+xml")):
        return "text"
    return None


def _sniff_charset(body: bytes) -> str | None:
    match = _META_CHARSET.search(body[:4096])
    return match.group(1).decode("ascii", errors="ignore") if match else None


async def _system_resolver(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"Could not resolve host {host!r}.") from exc
    return [info[4][0] for info in infos]


async def resolve_public(host: str, port: int, resolver: Resolver | None = None) -> str:
    """Return one public IP address for host, or raise BlockedAddress.

    Every address the name resolves to must be public: a name that answers with
    a mix of public and private addresses is refused outright.
    """
    addresses = await (resolver or _system_resolver)(host, port)
    if not addresses:
        raise FetchError(f"Could not resolve host {host!r}.")
    parsed = [ipaddress.ip_address(address.split("%")[0]) for address in addresses]
    if not all(address.is_global for address in parsed):
        raise BlockedAddress("Refusing to fetch a local or private network address.")
    parsed.sort(key=lambda address: address.version)  # IPv4 first: works on more networks
    return str(parsed[0])


async def _get_pinned(client: httpx.AsyncClient, url: str, resolver: Resolver | None):
    """Open a streaming GET to url, connected to an address we have checked."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("Only http and https URLs are supported.")
    if parts.username or parts.password:
        raise FetchError("URLs with embedded credentials are not supported.")
    host = parts.hostname
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise FetchError("The URL has an invalid port.") from exc
    address = await resolve_public(host, port, resolver)
    pinned = httpx.URL(url).copy_with(host=address)
    host_header = host if parts.port is None else f"{host}:{parts.port}"
    request = client.build_request(
        "GET", pinned, headers={"Host": host_header}, extensions={"sni_hostname": host}
    )
    return await client.send(request, stream=True)


async def download(
    url: str,
    *,
    resolver: Resolver | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    total_timeout: float = TOTAL_TIMEOUT,
) -> Download:
    """Download a public page, following redirects safely and capping its size."""
    url = url.strip()
    timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    # No keep-alive: connections are pinned to an IP, and two hostnames can share one.
    limits = httpx.Limits(max_keepalive_connections=0)
    try:
        async with asyncio.timeout(total_timeout):
            async with httpx.AsyncClient(
                headers=HEADERS, timeout=timeout, limits=limits, transport=transport, follow_redirects=False
            ) as client:
                for _ in range(MAX_REDIRECTS + 1):
                    response = await _get_pinned(client, url, resolver)
                    try:
                        if response.is_redirect and "location" in response.headers:
                            url = urljoin(url, response.headers["location"])
                            continue
                        return await _read_capped(url, response)
                    finally:
                        await response.aclose()
                raise FetchError("Too many redirects.")
    except TimeoutError as exc:
        raise FetchError(f"Timed out after {total_timeout:g} seconds.") from exc
    except httpx.TimeoutException as exc:
        raise FetchError("The site took too long to respond.") from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"Could not download the page ({type(exc).__name__}).") from exc


async def _read_capped(url: str, response: httpx.Response) -> Download:
    if response.status_code >= 400:
        raise FetchError(f"Page answered HTTP {response.status_code}.")
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type in ("application/octet-stream", "binary/octet-stream") and urlsplit(url).path.lower().endswith(".pdf"):
        content_type = "application/pdf"  # many servers send PDFs without naming the type
    kind = kind_of(content_type)
    if kind is None:
        raise FetchError(f"Unsupported content type {content_type!r}; web_express reads HTML, text and PDF.")
    limit = MAX_PDF_BYTES if kind == "pdf" else MAX_TEXT_BYTES
    chunks: list[bytes] = []
    size = 0
    truncated = False
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= limit:
            truncated = True
            break
    body = b"".join(chunks)[:limit]
    return Download(
        url=url,
        status=response.status_code,
        content_type=content_type,
        body=body,
        charset=response.charset_encoding,
        truncated=truncated,
    )

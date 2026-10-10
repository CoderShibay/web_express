"""Turn a downloaded page into clean, readable text."""

from __future__ import annotations

import io
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin

from .net import Download, FetchError, kind_of

MIN_MAIN_TEXT = 200  # below this, the main-content extractor probably missed the page
MAX_PDF_PAGES = 80


@dataclass
class Document:
    url: str
    title: str
    text: str
    published: str | None
    extractor: str  # which method produced the text
    truncated: bool  # the download hit the size limit
    fetched_at: float | None = None  # set when the page came from the local index
    stale: bool = False  # an old stored copy, served because the network failed


class _PlainExtractor(HTMLParser):
    """Fallback: reduce any HTML to text, keeping headings, lists and links."""

    SKIP = {"script", "style", "noscript", "svg", "template", "iframe"}
    BLOCK = {"p", "div", "section", "article", "br", "li", "tr", "ul", "ol", "table", "blockquote", "pre", "header", "main", "form"}
    HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "h6": "###### "}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._href: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.HEADINGS:
            self.parts.append("\n\n" + self.HEADINGS[tag])
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag == "a" and not self._skip_depth:
            href = dict(attrs).get("href") or ""
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                try:
                    self._href = urljoin(self.base_url, href)
                except ValueError:  # a malformed link: keep its text, drop the target
                    return
                self.parts.append("[")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.HEADINGS or tag in self.BLOCK:
            self.parts.append("\n")
        elif tag == "a" and self._href is not None:
            self.parts.append(f"]({self._href})")
            self._href = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        out: list[str] = []
        for line in lines:
            if line or (out and out[-1]):
                out.append(line)
        return "\n".join(out).strip()


def _plain_html(html: str, url: str) -> tuple[str, str]:
    parser = _PlainExtractor(url)
    parser.feed(html)
    return " ".join(parser.title.split()), parser.text()


def _extract_html(html: str, url: str) -> tuple[str, str, str | None, str]:
    """Return (title, text, published, extractor)."""
    import trafilatura

    title, published, main = "", None, ""
    try:
        tree = trafilatura.load_html(html)
        if tree is not None:
            main = trafilatura.extract(
                tree, url=url, output_format="markdown", include_links=True, include_tables=True, include_comments=False
            ) or ""
            meta = trafilatura.extract_metadata(tree, default_url=url)
            if meta is not None:
                title, published = meta.title or "", meta.date or None
    except Exception:  # a malformed page must fall through to the plain extractor
        main = ""
    if len(main) >= MIN_MAIN_TEXT:
        return title, main.strip(), published, "trafilatura"
    plain_title, plain = _plain_html(html, url)
    if len(plain) > len(main):
        return title or plain_title, plain, published, "plain"
    return title or plain_title, main.strip(), published, "trafilatura"


def _extract_pdf(body: bytes) -> tuple[str, str, str | None]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted:
            raise FetchError("This PDF is encrypted and cannot be read.")
        pages = [page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES]]
        info = reader.metadata
        title = (info.title or "") if info else ""
        date = info.creation_date if info else None
    except FetchError:
        raise
    except Exception as exc:
        raise FetchError(f"Could not read this PDF ({type(exc).__name__}).") from exc
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text:
        raise FetchError("This PDF has no extractable text (it may be scanned images).")
    return title.strip(), text, date.date().isoformat() if date else None


def extract(download: Download) -> Document:
    """Convert a Download into a Document. CPU-bound: call it in a thread."""
    try:
        return _extract(download)
    except FetchError:
        raise
    except Exception as exc:  # parsers meet hostile input; never let one page crash a search
        raise FetchError(f"Could not read this page ({type(exc).__name__}).") from exc


def _extract(download: Download) -> Document:
    kind = kind_of(download.content_type)
    if kind == "pdf":
        if download.truncated:
            raise FetchError("This PDF is larger than the download limit.")
        title, text, published = _extract_pdf(download.body)
        extractor = "pypdf"
    elif kind == "html":
        title, text, published, extractor = _extract_html(download.text(), download.url)
    else:
        title, text, published, extractor = "", download.text(), None, "text"
    return Document(
        url=download.url,
        title=" ".join(title.split()),
        text=text,
        published=published,
        extractor=extractor,
        truncated=download.truncated,
    )

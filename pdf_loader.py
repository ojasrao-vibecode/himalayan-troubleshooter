import re
import pypdf

_cache: dict[str, str] = {}


def load_pdf_text(path: str) -> str:
    """Extract and clean text from a PDF. Result is cached in memory."""
    if path in _cache:
        return _cache[path]

    reader = pypdf.PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"--- Page {i} ---\n{text.strip()}")

    raw = "\n\n".join(pages)

    # Collapse runs of blank lines and strip trailing spaces
    cleaned = re.sub(r"\n{3,}", "\n\n", raw)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)

    _cache[path] = cleaned
    return cleaned

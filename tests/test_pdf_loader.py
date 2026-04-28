from unittest.mock import MagicMock, patch
import pdf_loader


def _make_mock_reader(pages: list[str]):
    mock_page = lambda text: MagicMock(extract_text=MagicMock(return_value=text))
    reader = MagicMock()
    reader.pages = [mock_page(t) for t in pages]
    return reader


def test_extracts_text_with_page_markers():
    pdf_loader._cache.clear()
    with patch("pdf_loader.pypdf.PdfReader", return_value=_make_mock_reader(["Hello world"])):
        result = pdf_loader.load_pdf_text("fake.pdf")
    assert "--- Page 1 ---" in result
    assert "Hello world" in result


def test_skips_blank_pages():
    pdf_loader._cache.clear()
    with patch("pdf_loader.pypdf.PdfReader", return_value=_make_mock_reader(["", "Real content", ""])):
        result = pdf_loader.load_pdf_text("fake.pdf")
    assert "--- Page 1 ---" not in result
    assert "--- Page 2 ---" in result
    assert "Real content" in result


def test_cleans_excess_blank_lines():
    pdf_loader._cache.clear()
    with patch("pdf_loader.pypdf.PdfReader", return_value=_make_mock_reader(["Line1\n\n\n\n\nLine2"])):
        result = pdf_loader.load_pdf_text("fake.pdf")
    assert "\n\n\n" not in result


def test_caches_result():
    pdf_loader._cache.clear()
    reader = _make_mock_reader(["Cached content"])
    with patch("pdf_loader.pypdf.PdfReader", return_value=reader) as mock_pdf:
        pdf_loader.load_pdf_text("cached.pdf")
        pdf_loader.load_pdf_text("cached.pdf")
    # PdfReader should only be called once — second call uses cache
    mock_pdf.assert_called_once()

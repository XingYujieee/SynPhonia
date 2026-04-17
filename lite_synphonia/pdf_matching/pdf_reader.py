from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def normalize_space(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class PdfPageChunk:
    page_index: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class PdfPage:
    page_index: int
    page_label: str
    text: str
    text_available: bool
    chunks: list[PdfPageChunk]

    @property
    def text_preview(self) -> str:
        if not self.text:
            return ""
        return self.text[:120]


@dataclass(frozen=True)
class PdfDocument:
    source_path: str
    pages: list[PdfPage]
    warnings: dict[str, object]


def read_pdf_document(
    path: str | Path,
    *,
    chunk_min_chars: int = 200,
    chunk_max_chars: int = 480,
) -> PdfDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF matching requires pypdf. Install it with `python3 -m pip install pypdf`."
        ) from exc

    pdf_path = Path(path).expanduser().resolve()
    reader = PdfReader(str(pdf_path))

    pages: list[PdfPage] = []
    empty_pages: list[int] = []
    extraction_failures: list[dict[str, object]] = []

    for index, raw_page in enumerate(reader.pages):
        extracted = ""
        try:
            extracted = raw_page.extract_text() or ""
        except Exception as exc:
            extraction_failures.append(
                {
                    "page_index": index,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

        normalized = normalize_space(extracted)
        if not normalized:
            empty_pages.append(index)

        chunks = build_page_chunks(
            extracted,
            page_index=index,
            chunk_min_chars=chunk_min_chars,
            chunk_max_chars=chunk_max_chars,
        )
        pages.append(
            PdfPage(
                page_index=index,
                page_label=f"Page {index + 1}",
                text=normalized,
                text_available=bool(normalized),
                chunks=chunks,
            )
        )

    if not pages:
        raise RuntimeError(f"PDF file has no pages: {pdf_path}")

    return PdfDocument(
        source_path=str(pdf_path),
        pages=pages,
        warnings={
            "empty_pages": empty_pages,
            "extraction_failures": extraction_failures,
        },
    )


def build_page_chunks(
    text: str,
    *,
    page_index: int,
    chunk_min_chars: int = 120,
    chunk_max_chars: int = 220,
) -> list[PdfPageChunk]:
    normalized = normalize_space(text)
    if not normalized:
        return []

    raw_parts = [part.strip() for part in text.replace("\r", "\n").split("\n") if part.strip()]
    parts = [normalize_space(part) for part in raw_parts if normalize_space(part)]
    if not parts:
        parts = [normalized]

    chunks: list[PdfPageChunk] = []
    current_parts: list[str] = []
    current_length = 0

    for part in parts:
        part_length = len(part)
        if current_parts and current_length >= chunk_min_chars and current_length + part_length > chunk_max_chars:
            chunks.append(
                PdfPageChunk(
                    page_index=page_index,
                    chunk_index=len(chunks),
                    text=" ".join(current_parts),
                )
            )
            current_parts = []
            current_length = 0

        if part_length > chunk_max_chars:
            overflow_chunks = _split_long_text(part, chunk_max_chars)
            for overflow in overflow_chunks[:-1]:
                if current_parts:
                    chunks.append(
                        PdfPageChunk(
                            page_index=page_index,
                            chunk_index=len(chunks),
                            text=" ".join(current_parts),
                        )
                    )
                    current_parts = []
                    current_length = 0
                chunks.append(
                    PdfPageChunk(
                        page_index=page_index,
                        chunk_index=len(chunks),
                        text=overflow,
                    )
                )
            part = overflow_chunks[-1]
            part_length = len(part)

        current_parts.append(part)
        current_length += part_length + 1

    if current_parts:
        chunks.append(
            PdfPageChunk(
                page_index=page_index,
                chunk_index=len(chunks),
                text=" ".join(current_parts),
            )
        )

    return chunks


def _split_long_text(text: str, chunk_max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [text[:chunk_max_chars]]

    pieces: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        word_length = len(word)
        if current and current_length + word_length + 1 > chunk_max_chars:
            pieces.append(" ".join(current))
            current = []
            current_length = 0
        current.append(word)
        current_length += word_length + 1
    if current:
        pieces.append(" ".join(current))
    return pieces

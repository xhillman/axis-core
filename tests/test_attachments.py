"""Tests for attachment types and serialization helpers."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from axis_core.attachments import (
    MAX_ATTACHMENT_SIZE,
    PDF,
    Attachment,
    Image,
    serialize_attachments,
)


class TestAttachment:
    """Tests for base Attachment behavior."""

    def test_from_file_loads_data(self, tmp_path: Path) -> None:
        """Attachment.from_file should load bytes and metadata."""
        path = tmp_path / "note.txt"
        data = b"hello"
        path.write_bytes(data)

        attachment = Attachment.from_file(str(path))

        assert attachment.data == data
        assert attachment.filename == "note.txt"
        assert attachment.mime_type == "text/plain"

    def test_to_base64(self) -> None:
        """Attachment.to_base64 should encode bytes to base64 string."""
        attachment = Attachment(data=b"abc", mime_type="text/plain", filename="a.txt")

        encoded = attachment.to_base64()

        assert encoded == base64.b64encode(b"abc").decode("utf-8")

    def test_size_limit_enforced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Attachment.from_file should enforce MAX_ATTACHMENT_SIZE."""
        path = tmp_path / "big.bin"
        path.write_bytes(b"12345")

        monkeypatch.setattr("axis_core.attachments.MAX_ATTACHMENT_SIZE", 4)

        with pytest.raises(ValueError, match="too large"):
            Attachment.from_file(str(path))

    def test_default_size_limit_constant(self) -> None:
        """MAX_ATTACHMENT_SIZE should default to 10MB."""
        assert MAX_ATTACHMENT_SIZE == 10 * 1024 * 1024


class TestImageAttachment:
    """Tests for Image attachment validation."""

    def test_image_from_file_accepts_image(self, tmp_path: Path) -> None:
        """Image.from_file should accept image files by MIME type."""
        path = tmp_path / "image.png"
        path.write_bytes(b"\x89PNG")

        image = Image.from_file(str(path))

        assert isinstance(image, Image)
        assert image.mime_type.startswith("image/")

    def test_image_from_file_rejects_non_image(self, tmp_path: Path) -> None:
        """Image.from_file should reject non-image files."""
        path = tmp_path / "note.txt"
        path.write_bytes(b"hello")

        with pytest.raises(ValueError, match="image/"):
            Image.from_file(str(path))


class TestPDFAttachment:
    """Tests for PDF attachment validation."""

    def test_pdf_from_file_accepts_pdf(self, tmp_path: Path) -> None:
        """PDF.from_file should accept PDF files by MIME type."""
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4")

        pdf = PDF.from_file(str(path))

        assert isinstance(pdf, PDF)
        assert pdf.mime_type == "application/pdf"

    def test_pdf_from_file_rejects_non_pdf(self, tmp_path: Path) -> None:
        """PDF.from_file should reject non-PDF files."""
        path = tmp_path / "note.txt"
        path.write_bytes(b"hello")

        with pytest.raises(ValueError, match="application/pdf"):
            PDF.from_file(str(path))


class TestImageMimeValidation:
    """Tests for Image constructor MIME validation (Task 5.1)."""

    def test_accepts_valid_image_mime(self) -> None:
        """Image constructor should accept mime_type starting with 'image/'."""
        img = Image(data=b"\x89PNG", mime_type="image/png")
        assert img.mime_type == "image/png"

    def test_accepts_image_jpeg(self) -> None:
        """Image constructor should accept image/jpeg."""
        img = Image(data=b"\xff\xd8", mime_type="image/jpeg")
        assert img.mime_type == "image/jpeg"

    def test_rejects_non_image_mime(self) -> None:
        """Image constructor should reject mime_type not starting with 'image/'."""
        with pytest.raises(ValueError, match="image/"):
            Image(data=b"data", mime_type="text/plain")

    def test_rejects_application_pdf_mime(self) -> None:
        """Image constructor should reject application/pdf."""
        with pytest.raises(ValueError, match="image/"):
            Image(data=b"data", mime_type="application/pdf")

    def test_from_file_still_validates(self, tmp_path: Path) -> None:
        """Image.from_file should still reject non-image files."""
        path = tmp_path / "note.txt"
        path.write_bytes(b"hello")
        with pytest.raises(ValueError, match="image/"):
            Image.from_file(str(path))


class TestPDFMimeValidation:
    """Tests for PDF constructor MIME validation (Task 5.2)."""

    def test_accepts_valid_pdf_mime(self) -> None:
        """PDF constructor should accept 'application/pdf'."""
        pdf = PDF(data=b"%PDF", mime_type="application/pdf")
        assert pdf.mime_type == "application/pdf"

    def test_rejects_non_pdf_mime(self) -> None:
        """PDF constructor should reject mime_type != 'application/pdf'."""
        with pytest.raises(ValueError, match="application/pdf"):
            PDF(data=b"data", mime_type="text/plain")

    def test_rejects_image_mime(self) -> None:
        """PDF constructor should reject image MIME types."""
        with pytest.raises(ValueError, match="application/pdf"):
            PDF(data=b"data", mime_type="image/png")

    def test_from_file_still_validates(self, tmp_path: Path) -> None:
        """PDF.from_file should still reject non-PDF files."""
        path = tmp_path / "note.txt"
        path.write_bytes(b"hello")
        with pytest.raises(ValueError, match="application/pdf"):
            PDF.from_file(str(path))


class TestAttachmentSerialization:
    """Tests for attachment serialization helpers."""

    def test_serialize_attachments_metadata(self) -> None:
        """serialize_attachments should emit metadata dicts."""
        attachment = Attachment(data=b"data", mime_type="text/plain", filename="a.txt")
        metadata = {"type": "custom", "mime_type": "text/plain", "filename": "b.txt"}

        serialized = serialize_attachments([attachment, metadata])

        assert serialized[0]["type"] == "attachment"
        assert serialized[0]["mime_type"] == "text/plain"
        assert serialized[0]["filename"] == "a.txt"
        assert serialized[0]["size_bytes"] == len(b"data")
        assert serialized[1] == metadata

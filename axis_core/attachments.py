"""Attachment types and serialization helpers for axis-core.

Implements eager-loading attachments with size limits (AD-021).
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10MB


@dataclass(frozen=True)
class Attachment:
    """Base class for attachments."""

    data: bytes
    mime_type: str
    filename: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> Attachment:
        """Load attachment from file (eager loading)."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Attachment file not found: {path}")

        size = path.stat().st_size
        if size > MAX_ATTACHMENT_SIZE:
            raise ValueError(
                f"Attachment {path} is too large ({size / 1024 / 1024:.1f}MB). "
                f"Max size: {MAX_ATTACHMENT_SIZE / 1024 / 1024:.0f}MB"
            )

        data = path.read_bytes()
        mime_type = _guess_mime_type(path)

        return cls(
            data=data,
            mime_type=mime_type,
            filename=path.name,
        )

    def to_base64(self) -> str:
        """Encode attachment data for API transmission."""
        return base64.b64encode(self.data).decode("utf-8")

    def to_metadata(self) -> dict[str, Any]:
        """Serialize attachment as metadata (no raw bytes)."""
        return {
            "type": self.__class__.__name__.lower(),
            "mime_type": self.mime_type,
            "filename": self.filename,
            "size_bytes": len(self.data),
        }


@dataclass(frozen=True)
class Image(Attachment):
    """Image attachment."""

    @classmethod
    def from_file(cls, path: str | Path) -> Image:
        attachment = super().from_file(path)

        if not attachment.mime_type.startswith("image/"):
            raise ValueError(f"File {path} is not an image")

        return cls(**asdict(attachment))


@dataclass(frozen=True)
class PDF(Attachment):
    """PDF attachment."""

    @classmethod
    def from_file(cls, path: str | Path) -> PDF:
        attachment = super().from_file(path)

        if attachment.mime_type != "application/pdf":
            raise ValueError(f"File {path} is not a PDF")

        return cls(**asdict(attachment))


AttachmentMetadata = dict[str, Any]
AttachmentLike = Attachment | AttachmentMetadata


def serialize_attachments(attachments: Iterable[AttachmentLike]) -> list[AttachmentMetadata]:
    """Serialize attachments to metadata-only representations."""
    serialized: list[AttachmentMetadata] = []
    for attachment in attachments:
        if isinstance(attachment, Attachment):
            serialized.append(attachment.to_metadata())
        elif isinstance(attachment, dict):
            serialized.append(dict(attachment))
        else:
            raise TypeError(
                "attachments must be Attachment or dict, "
                f"got {type(attachment).__name__}"
            )
    return serialized


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.as_posix())
    return mime_type or "application/octet-stream"

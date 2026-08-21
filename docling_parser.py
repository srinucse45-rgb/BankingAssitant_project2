"""Docling multimodal PDF parser.

Optimized for the Smart Banking knowledge-base:
- OCR disabled by default for digitally generated PDFs.
- Table structure extraction enabled.
- Picture images enabled.
- Full-page image generation disabled.
- Uses the current Docling document model; does NOT use page.blocks.
"""

import base64
import io
import os
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from PIL import Image

_TEXT_TYPES = {"TextItem", "SectionHeaderItem", "ListItem", "CodeItem"}
_TABLE_TYPES = {"TableItem"}
_PICTURE_TYPES = {"PictureItem"}


def _build_converter() -> DocumentConverter:
    """Build a lightweight standard PDF pipeline."""

    options = PdfPipelineOptions()

    # Digital PDFs: avoid CPU-heavy OCR unless explicitly requested.
    options.do_ocr = os.getenv("DOCLING_ENABLE_OCR", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Tables are important for banking RAG.
    options.do_table_structure = True

    # Keep actual document pictures, but don't render every page.
    options.generate_picture_images = True
    options.generate_page_images = False

    # 1.0 ~= 72 DPI; keeps image processing/storage reasonable.
    options.images_scale = 1.0

    print(
        "[docling_parser] "
        f"OCR={'ON' if options.do_ocr else 'OFF'}, "
        "tables=ON, picture_images=ON, page_images=OFF"
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def _page_number(item) -> int | None:
    prov = getattr(item, "prov", None)

    if prov:
        page_no = getattr(prov[0], "page_no", None)
        if page_no is not None:
            return int(page_no)

    return None


def _bbox(item) -> list[float] | None:
    prov = getattr(item, "prov", None)

    if not prov:
        return None

    bbox = getattr(prov[0], "bbox", None)

    if bbox is None:
        return None

    return [
        float(bbox.l),
        float(bbox.t),
        float(bbox.r),
        float(bbox.b),
    ]


def _image_to_base64(image) -> str:
    if not isinstance(image, Image.Image):
        return ""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_caption(item, doc) -> str:
    """Safely obtain a picture caption across Docling versions."""

    try:
        caption_method = getattr(item, "caption_text", None)

        if callable(caption_method):
            value = caption_method(doc)

            if value:
                return str(value).strip()

    except Exception:
        pass

    return ""


def parse_document(file_path: str) -> list[dict]:
    """Parse a PDF into text, table and image elements."""

    resolved = Path(file_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"Expected PDF file, got {resolved.suffix}")

    converter = _build_converter()

    print(f"[docling_parser] Converting: {resolved.name}")

    result = converter.convert(resolved)
    doc = result.document

    # IMPORTANT:
    # Do not use doc.pages[n].blocks.
    # Current Docling exposes document elements through iterate_items().
    page_count = len(doc.pages)

    print(f"[docling_parser] Parsed document with " f"{page_count} pages")

    elements: list[dict] = []
    current_section = ""

    for item, level in doc.iterate_items(
        with_groups=False,
        traverse_pictures=False,
    ):
        try:
            item_type = item.__class__.__name__

            page_number = _page_number(item)
            bbox = _bbox(item)

            # ---------------------------------------------------------
            # TEXT / SECTION HEADERS
            # ---------------------------------------------------------
            if item_type in _TEXT_TYPES:

                text = str(getattr(item, "text", "") or "").strip()

                if not text:
                    continue

                if item_type == "SectionHeaderItem":
                    current_section = text[:200]

                elements.append(
                    {
                        "content": text,
                        "content_type": "text",
                        "metadata": {
                            "page_number": page_number,
                            "section": current_section,
                            "bbox": bbox,
                            "element_type": item_type,
                            "level": level,
                        },
                    }
                )

            # ---------------------------------------------------------
            # TABLE
            # ---------------------------------------------------------
            elif item_type in _TABLE_TYPES:

                try:
                    markdown = item.export_to_markdown(doc).strip()
                except Exception as exc:
                    print("[docling_parser] Table markdown " f"export failed: {exc}")
                    continue

                if not markdown:
                    continue

                elements.append(
                    {
                        "content": markdown,
                        "content_type": "table",
                        "metadata": {
                            "page_number": page_number,
                            "section": current_section,
                            "bbox": bbox,
                            "element_type": "TableItem",
                        },
                    }
                )

            # ---------------------------------------------------------
            # PICTURE / IMAGE
            # ---------------------------------------------------------
            elif item_type in _PICTURE_TYPES:

                image = None

                try:
                    image = item.get_image(doc)
                except Exception as exc:
                    print(
                        "[docling_parser] Could not render "
                        f"picture on page {page_number}: {exc}"
                    )

                if image is None:
                    continue

                image_base64 = _image_to_base64(image)

                if not image_base64:
                    continue

                caption = _get_caption(
                    item,
                    doc,
                )

                elements.append(
                    {
                        "content": (f"[Image: {caption}]" if caption else "[Image]"),
                        "content_type": "image",
                        "metadata": {
                            "page_number": page_number,
                            "section": current_section,
                            "bbox": bbox,
                            "image_base64": image_base64,
                            "caption": caption,
                            "element_type": "PictureItem",
                        },
                    }
                )

        except Exception as exc:
            print(
                "[docling_parser] Warning: failed to process "
                f"{item.__class__.__name__}: {exc}"
            )

    text_count = sum(x["content_type"] == "text" for x in elements)

    table_count = sum(x["content_type"] == "table" for x in elements)

    image_count = sum(x["content_type"] == "image" for x in elements)

    print(
        "[docling_parser] Extracted "
        f"{len(elements)} elements: "
        f"text={text_count}, "
        f"tables={table_count}, "
        f"images={image_count}"
    )

    return elements

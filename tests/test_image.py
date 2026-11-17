"""Unit tests for :mod:`oscilloscope_mcp.helpers.image` — BMP → PNG
conversion (round-tripped through a minimal PNG decoder) and format
sniffing.
"""

from __future__ import annotations

import struct
import zlib

import pytest

from oscilloscope_mcp.helpers.image import PNG_MAGIC, bmp_to_png, sniff_image_format


Pixel = tuple[int, int, int]


def _make_bmp(
    rows: list[list[Pixel]] | list[list[int]],
    bpp: int = 24,
    top_down: bool = False,
    palette: list[Pixel] | None = None,
    compression: int = 0,
) -> bytes:
    """Build a BITMAPINFOHEADER BMP. ``rows`` are listed top → bottom:
    RGB tuples for 24/32-bit, palette indices for 8-bit.
    """
    height = len(rows)
    width = len(rows[0])
    stride = ((width * bpp + 31) // 32) * 4
    pal_bytes = b""
    if bpp == 8:
        assert palette is not None
        pal_bytes = b"".join(bytes((b, g, r, 0)) for r, g, b in palette)
    stored = rows if top_down else list(reversed(rows))
    pixels = bytearray()
    for row in stored:
        line = bytearray()
        for px in row:
            if bpp == 8:
                line.append(px)
            else:
                r, g, b = px
                line += bytes((b, g, r)) + (b"\xff" if bpp == 32 else b"")
        line += bytes(stride - len(line))
        pixels += line
    offset = 14 + 40 + len(pal_bytes)
    file_header = b"BM" + struct.pack("<IHHI", offset + len(pixels), 0, 0, offset)
    info = struct.pack(
        "<IiiHHIIiiII", 40, width, -height if top_down else height, 1, bpp,
        compression, len(pixels), 2835, 2835, len(palette or []), 0,
    )
    return file_header + info + pal_bytes + bytes(pixels)


def _decode_png(png: bytes) -> tuple[int, int, list[list[Pixel]]]:
    """Decode the unfiltered 8-bit RGB PNGs that bmp_to_png emits."""
    assert png.startswith(PNG_MAGIC)
    pos = len(PNG_MAGIC)
    idat = b""
    width = height = 0
    while pos < len(png):
        (n,) = struct.unpack_from(">I", png, pos)
        tag = png[pos + 4:pos + 8]
        body = png[pos + 8:pos + 8 + n]
        (crc,) = struct.unpack_from(">I", png, pos + 8 + n)
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF, f"bad CRC on {tag!r}"
        if tag == b"IHDR":
            width, height, depth, ctype, _, _, _ = struct.unpack(">IIBBBBB", body)
            assert (depth, ctype) == (8, 2)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride = width * 3 + 1
    out: list[list[Pixel]] = []
    for y in range(height):
        line = raw[y * stride:(y + 1) * stride]
        assert line[0] == 0, "expected filter type 0"
        out.append([tuple(line[1 + i:4 + i]) for i in range(0, width * 3, 3)])
    return width, height, out


_RED, _GREEN, _BLUE, _WHITE = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)
# 3 px wide → 9 bytes per 24-bit row → padded to 12: exercises padding.
_ROWS = [[_RED, _GREEN, _BLUE], [_WHITE, (10, 20, 30), (0, 0, 0)]]


@pytest.mark.parametrize("top_down", [False, True])
def test_bmp24_round_trips_with_row_order_and_padding(top_down: bool) -> None:
    w, h, rows = _decode_png(bmp_to_png(_make_bmp(_ROWS, bpp=24, top_down=top_down)))
    assert (w, h) == (3, 2)
    assert rows == _ROWS


def test_bmp32_drops_alpha() -> None:
    _, _, rows = _decode_png(bmp_to_png(_make_bmp(_ROWS, bpp=32)))
    assert rows == _ROWS


def test_bmp8_expands_palette() -> None:
    palette = [_RED, _GREEN, _BLUE]
    bmp = _make_bmp([[0, 1, 2], [2, 1, 0]], bpp=8, palette=palette)
    _, _, rows = _decode_png(bmp_to_png(bmp))
    assert rows == [[_RED, _GREEN, _BLUE], [_BLUE, _GREEN, _RED]]


def test_rejects_non_bmp() -> None:
    with pytest.raises(ValueError, match="not a BMP"):
        bmp_to_png(PNG_MAGIC + bytes(100))


def test_rejects_rle_compressed_bmp() -> None:
    with pytest.raises(ValueError, match="compressed"):
        bmp_to_png(_make_bmp(_ROWS, bpp=24, compression=1))


def test_rejects_truncated_bmp() -> None:
    with pytest.raises(ValueError, match="truncated"):
        bmp_to_png(_make_bmp(_ROWS, bpp=24)[:-5])


@pytest.mark.parametrize("data, expected", [
    (PNG_MAGIC + b"rest", "PNG"),
    (b"BM" + bytes(60), "BMP"),
    (b"\xff\xd8\xff\xe0", "JPEG"),
    (b"II*\x00", "TIFF"),
    (b"#9000000", None),
])
def test_sniff_image_format(data: bytes, expected: str | None) -> None:
    assert sniff_image_format(data) == expected

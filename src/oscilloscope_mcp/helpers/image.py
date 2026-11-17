"""Screenshot image helpers — format sniffing and BMP → PNG conversion.

Some instruments only dump a Windows BMP (the MSO5000's documented
``:DISPlay:DATA?`` takes no format argument and returns ``.bmp``), while
the screenshot tool's contract is PNG. The conversion is pure stdlib
(``struct`` + ``zlib``) so the package keeps its two-dependency footprint.
"""

from __future__ import annotations

import struct
import zlib

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def sniff_image_format(data: bytes) -> str | None:
    """Return ``"PNG"`` / ``"BMP"`` / ``"JPEG"`` / ``"TIFF"`` from the
    magic bytes, or ``None`` if unrecognized.
    """
    if data.startswith(PNG_MAGIC):
        return "PNG"
    if data[:2] == b"BM":
        return "BMP"
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    return None


def bmp_to_png(data: bytes) -> bytes:
    """Convert an uncompressed 8/24/32-bit Windows BMP to an RGB PNG.

    Handles bottom-up and top-down row order and 4-byte row padding.
    32-bit images are read as BGRA/BGRX (alpha dropped). Raises
    :class:`ValueError` for anything else (RLE, 16-bit, OS/2 headers,
    truncated data).
    """
    if len(data) < 54 or data[:2] != b"BM":
        raise ValueError("not a BMP image (missing 'BM' header)")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    header_size = struct.unpack_from("<I", data, 14)[0]
    if header_size < 40:
        raise ValueError(f"unsupported BMP header size {header_size}")
    width, height, _planes, bpp, compression = struct.unpack_from(
        "<iiHHI", data, 18
    )
    colors_used = struct.unpack_from("<I", data, 46)[0]
    if width <= 0 or height == 0:
        raise ValueError(f"invalid BMP dimensions {width}x{height}")
    top_down = height < 0
    height = abs(height)

    if bpp not in (8, 24, 32):
        raise ValueError(f"unsupported BMP bit depth {bpp}")
    # 0 = BI_RGB; 3 = BI_BITFIELDS (accepted for 32-bit with BGRA masks).
    if compression == 3 and bpp == 32:
        # R/G/B masks sit right after the 40-byte info fields (offset 54)
        # for BITMAPINFOHEADER + bitfields and for V4/V5 headers alike.
        masks = struct.unpack_from("<III", data, 54)
        if masks != (0x00FF0000, 0x0000FF00, 0x000000FF):
            raise ValueError(f"unsupported BMP channel masks {masks!r}")
    elif compression != 0:
        raise ValueError(f"compressed BMP (compression={compression}) not supported")

    row_stride = ((width * bpp + 31) // 32) * 4
    if len(data) < pixel_offset + row_stride * height:
        raise ValueError("truncated BMP pixel data")

    r_tab = g_tab = b_tab = None
    if bpp == 8:
        n_colors = colors_used or 256
        pal_off = 14 + header_size
        pal = data[pal_off:pal_off + 4 * n_colors]
        if len(pal) < 4 * n_colors:
            raise ValueError("truncated BMP palette")
        pad = bytes(256 - n_colors)
        b_tab = bytes(pal[0::4]) + pad
        g_tab = bytes(pal[1::4]) + pad
        r_tab = bytes(pal[2::4]) + pad

    row_bytes = width * bpp // 8
    raw = bytearray()
    for y in range(height):
        src_row = y if top_down else height - 1 - y
        start = pixel_offset + src_row * row_stride
        row = data[start:start + row_bytes]
        out = bytearray(width * 3)
        if bpp == 24:
            out[0::3] = row[2::3]
            out[1::3] = row[1::3]
            out[2::3] = row[0::3]
        elif bpp == 32:
            out[0::3] = row[2::4]
            out[1::3] = row[1::4]
            out[2::3] = row[0::4]
        else:
            out[0::3] = row.translate(r_tab)
            out[1::3] = row.translate(g_tab)
            out[2::3] = row.translate(b_tab)
        raw += b"\x00"  # PNG filter type 0 (None) per scanline
        raw += out

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        PNG_MAGIC
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(tag: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(tag + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

"""RIGOL MSO5000-series oscilloscope driver (MSO5072 … MSO5354).

SCPI dialect reference: RIGOL MSO5000 Programming Guide, PGA25100-1110
(Oct 2018). Same raw-TCP port 5555 transport and IEEE-488.2 binary
blocks as the DS1000Z, and most of the instruction set is identical, so
this driver subclasses :class:`RigolDs1000z` and overrides only what the
guide documents differently:

- **Screenshot**: ``:DISPlay:DATA?`` takes no arguments and returns a
  BMP (≈1.8 MB at 1024 × 600). The driver converts it to PNG locally when
  PNG is requested, and widens the socket timeout for the transfer.
- **Channel labels**: the guide has no ``:CHANnel<n>:LABel`` command, so
  label requests are reported as a caveat instead of being sent.
- **Cursors**: manual cursor pixel space is 0–999 × 0–479 over a
  10 × 8 division grid (DS1000Z: 600 × 400 over 12 × 8), and an explicit
  ``:CURSor:MANual:TYPE`` selects time vs amplitude cursors.
- **Waveform time base**: ``:WAVeform:XORigin?`` is already the start time
  relative to the trigger and ``XREFerence`` is always 0; there is no
  ``:TRIGger:POSition?`` (an unanswered query stalls for the full socket
  timeout × retries), so t0 comes straight from the preamble.
- **Memory depth**: ``:ACQuire:MDEPth`` takes tokens (``10k``, ``1M`` …)
  rather than integers. The guide says the query returns tokens too, but
  firmware 00.01.03.02.02 answers in scientific notation (``1.0000E+03``);
  both are parsed.

Verified on an MSO5104 (firmware 00.01.03.02.02) over USB: identification,
bare ``:DISP:DATA?`` → 1.8 MB BMP, channel / timebase / acquire / trigger
reads, NORMal waveform reads and ``:MEAS:ITEM?``. Setters, cursors, RAW
reads and statistics are not yet hardware-verified.
- **Statistics**: an item must be enabled with
  ``:MEASure:STATistic:ITEM <item>,<src>`` before its statistics can be
  queried; there is no COUNt stat type.

Per-model differences (bandwidth, BW-limit choices, timebase minimum)
live in the model profile YAML, the instruction set in
``profiles/_mso5000_family.yaml``.
"""

from __future__ import annotations

from oscilloscope_mcp.helpers import acquisition as acq_helper
from oscilloscope_mcp.helpers import measure as meas_helper
from oscilloscope_mcp.helpers.image import bmp_to_png, sniff_image_format
from oscilloscope_mcp.instruments._base import (
    CursorPair,
    MeasureStatResult,
    ScreenshotPlan,
    ScreenshotResult,
)
from oscilloscope_mcp.instruments.rigol_ds1000z import RigolDs1000z, _parse_float
from oscilloscope_mcp.transport.scpi_lan import ScpiError


# Manual-cursor pixel geometry (guide §:CURSor:MANual:CAX/CAY): X 0..999
# across 10 divisions, Y 0..479 across 8 divisions, origin top-left.
_X_PX_MAX = 999
_Y_PX_MAX = 479
_PX_PER_TIME_DIV = 100
_PX_PER_VOLT_DIV = 60
_TIME_CENTER_PX = 500
_VOLT_CENTER_PX = 240

# NORMal-mode (screen) waveform reads return at most this many points.
_NORMAL_MODE_POINTS = 1000

# A BMP screenshot is rendered on the instrument then streamed as ~1.8 MB;
# that outlasts the 3 s default query timeout, so widen it for this call.
_SCREENSHOT_MIN_TIMEOUT_S = 15.0
_SCREENSHOT_RECV_MAX = 16 * 1024 * 1024

# :ACQuire:MDEPth tokens (guide: {AUTO|1k|10k|100k|1M|10M|25M|50M|100M|200M}).
_MDEPTH_TOKENS: dict[int, str] = {
    1_000: "1k",
    10_000: "10k",
    100_000: "100k",
    1_000_000: "1M",
    10_000_000: "10M",
    25_000_000: "25M",
    50_000_000: "50M",
    100_000_000: "100M",
    200_000_000: "200M",
}

# Profile stat-type keyword → MeasureStatResult field.
_STAT_FIELD = {
    "CURRent": "current",
    "MAXimum": "maximum",
    "MINimum": "minimum",
    "AVERages": "average",
    "DEViation": "deviation",
}

_LABEL_CAVEAT = (
    "channel labels were not applied: the MSO5000 programming guide has no "
    ":CHANnel<n>:LABel command. Describe the channels in your reply instead."
)


class RigolMso5000(RigolDs1000z):
    """Driver for any MSO5000-series scope (MSO5072/5074/5102/5104/5204/5354)."""

    # ------------------------------------------------------------------
    # Acquisition — memory-depth tokens
    # ------------------------------------------------------------------

    def _parse_memory_depth(self, resp: str) -> str | int | None:
        raw = (resp or "").strip()
        if not raw:
            return None
        if raw.upper() == "AUTO":
            return "AUTO"
        try:
            return acq_helper.parse_point_count(raw)
        except ValueError:
            return None

    def _memory_depth_arg(self, validated: str | int) -> str:
        if isinstance(validated, str):
            return validated  # "AUTO"
        return _MDEPTH_TOKENS.get(int(validated), str(int(validated)))

    # ------------------------------------------------------------------
    # Measurement statistics
    # ------------------------------------------------------------------

    def measure_statistics(
        self,
        items: list[str],
        source: str = "CHAN1",
        stat_types: list[str] | None = None,
    ) -> list[MeasureStatResult]:
        # Validate everything before any SCPI is sent.
        resolved = meas_helper.resolve_items(self.profile, items)
        src = meas_helper.normalize_source(self.profile, source)
        for name, spec in resolved:
            if meas_helper.is_dual_source(spec):
                # The query needs both sources; with one it is "invalid"
                # per the guide and the instrument never answers.
                raise meas_helper.MeasureValidationError(
                    f"statistics for two-source item {name!r} are not "
                    "supported; use scope_measure with source2 instead"
                )
        if stat_types is None:
            types = meas_helper.stat_types_list(self.profile)
        else:
            types = [meas_helper.validate_stat_type(self.profile, st) for st in stat_types]

        # Enable statistics and register each item on its source; without
        # the registration the instrument has no statistics to report.
        cmds = [":MEAS:STAT:DISP ON"]
        cmds.extend(f":MEAS:STAT:ITEM {spec['scpi']},{src}" for _, spec in resolved)
        self.transport.write_many(cmds)

        results: list[MeasureStatResult] = []
        for name, spec in resolved:
            result = MeasureStatResult(item=name, source=src)
            for st in types:
                field_name = _STAT_FIELD.get(st)
                if field_name is None:
                    continue
                try:
                    raw = self.transport.query(
                        f":MEAS:STAT:ITEM? {st},{spec['scpi']},{src}"
                    )
                except ScpiError:
                    continue
                setattr(result, field_name, meas_helper.parse_measure_value(raw))
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Waveform hooks
    # ------------------------------------------------------------------

    def _waveform_setup_cmds(self, src: str, mode: str) -> list[str]:
        cmds = super()._waveform_setup_cmds(src, mode)
        if mode.upper() == "NORMAL":
            # STARt/STOP persist across reads, and a previous chunked RAW
            # read leaves them far outside the 1000-point screen buffer.
            cmds += [":WAV:STARt 1", f":WAV:STOP {_NORMAL_MODE_POINTS}"]
        return cmds

    def _waveform_t0_s(self, pre: dict[str, float]) -> float:
        # XORigin = start time of the data relative to the trigger;
        # XREFerence is documented as always 0 but honoured anyway.
        return pre["xorigin"] - pre["xreference"] * pre["xincrement"]

    # ------------------------------------------------------------------
    # Screenshot + annotation
    # ------------------------------------------------------------------

    def screenshot(self, plan: ScreenshotPlan) -> ScreenshotResult:
        fmt = (plan.image_format or "PNG").upper()
        allowed = [
            str(f).upper()
            for f in self.profile.get("capability", {}).get("screenshot_formats", ["PNG", "BMP"])
        ]
        if fmt not in allowed:
            raise ValueError(
                f"unsupported screenshot format {fmt!r}; valid: {', '.join(allowed)}"
            )

        caveats: list[str] = []
        if plan.channel_labels or plan.display_labels:
            caveats.append(_LABEL_CAVEAT)

        cursors_applied: list[CursorPair] = []
        if plan.cursor_pairs:
            self.transport.write(":CURS:MODE MAN")
            for pair in plan.cursor_pairs:
                cursors_applied.append(self._apply_cursor_pair(pair))
            if len(plan.cursor_pairs) > 1:
                caveats.append(
                    "the scope shows a single manual cursor pair; only the "
                    f"last one ({plan.cursor_pairs[-1].label!r}) is on screen"
                )

        payload = self._read_screen_image()
        actual = sniff_image_format(payload)
        if actual is None:
            raise ScpiError(
                f"screenshot data is not a recognized image (first bytes "
                f"{payload[:8]!r})"
            )
        if fmt == "PNG" and actual == "BMP":
            payload = bmp_to_png(payload)
            actual = "PNG"
        elif fmt != actual:
            caveats.append(
                f"requested {fmt} but the instrument returned {actual}; "
                f"the file contains {actual} data"
            )
        return ScreenshotResult(
            image_bytes=payload,
            image_format=actual,
            cursors_set=cursors_applied,
            channel_labels_applied={},
            caveats=caveats,
        )

    def _read_screen_image(self) -> bytes:
        old_timeout = getattr(self.transport, "timeout_s", None)
        if old_timeout is not None and old_timeout < _SCREENSHOT_MIN_TIMEOUT_S:
            self.transport.timeout_s = _SCREENSHOT_MIN_TIMEOUT_S
        try:
            return self.transport.query_binary(
                ":DISP:DATA?", recv_max=_SCREENSHOT_RECV_MAX
            )
        finally:
            if old_timeout is not None:
                self.transport.timeout_s = old_timeout

    def _apply_cursor_pair(self, pair: CursorPair) -> CursorPair:
        """Translate seconds/volts into MSO5000 cursor pixels and apply.

        A pair with time coordinates becomes TIME (vertical-line) cursors;
        a pair with only voltage coordinates becomes AMPLitude cursors.
        Positions off-screen are clamped to the screen edge.
        """
        n = acq_helper.validate_channel_number(self.profile, pair.source_channel)
        has_t = pair.ax_t_s is not None or pair.bx_t_s is not None
        has_v = pair.ay_v is not None or pair.by_v is not None
        cmds = [
            f":CURS:MAN:SOUR CHANnel{n}",
            f":CURS:MAN:TYPE {'TIME' if has_t or not has_v else 'AMPLitude'}",
        ]
        if has_t:
            s_per_div = self.timebase_s_per_div()
            offset_s = _parse_float(self.transport.query(":TIM:MAIN:OFFS?")) or 0.0
            if pair.ax_t_s is not None:
                cmds.append(f":CURS:MAN:CAX {_time_to_px(pair.ax_t_s, s_per_div, offset_s)}")
            if pair.bx_t_s is not None:
                cmds.append(f":CURS:MAN:CBX {_time_to_px(pair.bx_t_s, s_per_div, offset_s)}")
        if has_v:
            v_per_div = self.channel_scale_v_per_div(n)
            offset_v = self.channel_offset_v(n)
            if pair.ay_v is not None:
                cmds.append(f":CURS:MAN:CAY {_volt_to_px(pair.ay_v, v_per_div, offset_v)}")
            if pair.by_v is not None:
                cmds.append(f":CURS:MAN:CBY {_volt_to_px(pair.by_v, v_per_div, offset_v)}")
        self.transport.write_many(cmds)
        return pair


# ---------------------------------------------------------------------------
# Coordinate translation
# ---------------------------------------------------------------------------


def _time_to_px(t_s: float, s_per_div: float, offset_s: float = 0.0) -> int:
    """Time (s from trigger) → cursor X pixel. The screen centre shows
    t = timebase offset; clamped to 0..999.
    """
    if s_per_div <= 0:
        raise ValueError(f"timebase must be positive, got {s_per_div}")
    px = round(_TIME_CENTER_PX + (t_s - offset_s) / s_per_div * _PX_PER_TIME_DIV)
    return min(max(px, 0), _X_PX_MAX)


def _volt_to_px(v: float, v_per_div: float, offset_v: float) -> int:
    """Voltage → cursor Y pixel (origin top-left, so up = smaller y).

    The screen centre shows ``-offset``: the guide gives YORigin =
    offset / YINCrement and volts = (raw - YORigin - YREFerence) ×
    YINCrement, with raw = YREFerence at mid-screen. Clamped to 0..479.
    """
    if v_per_div <= 0:
        raise ValueError(f"voltage scale must be positive, got {v_per_div}")
    py = round(_VOLT_CENTER_PX - (v + offset_v) / v_per_div * _PX_PER_VOLT_DIV)
    return min(max(py, 0), _Y_PX_MAX)

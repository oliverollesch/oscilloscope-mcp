"""Unit tests for the RIGOL MSO5000 driver against a recording transport.

Covers the places where the MSO5000 dialect (Programming Guide
PGA25100-1110) differs from the DS1000Z: screenshot command + BMP→PNG,
cursor pixel geometry, waveform time base (no :TRIG:POS?), memory-depth
tokens, statistics registration, digital trigger sources and the
SETup/SHOLd trigger split — plus dispatch and profile-driven validation.
"""

from __future__ import annotations

from typing import Any

import pytest

from oscilloscope_mcp import instruments
from oscilloscope_mcp.helpers.acquisition import (
    AcquisitionValidationError,
    parse_point_count,
)
from oscilloscope_mcp.helpers.caveat_calc import compute_caveats
from oscilloscope_mcp.helpers.image import PNG_MAGIC
from oscilloscope_mcp.helpers.measure import MeasureValidationError
from oscilloscope_mcp.helpers.trigger import TriggerValidationError
from oscilloscope_mcp.instruments import _load_profile, open_scope
from oscilloscope_mcp.instruments._base import (
    AcquireSetup,
    ChannelSetup,
    CursorPair,
    ScreenshotPlan,
    TriggerSetup,
)
from oscilloscope_mcp.instruments.rigol_mso5000 import (
    RigolMso5000,
    _time_to_px,
    _volt_to_px,
)

from tests.test_image import _ROWS, _decode_png, _make_bmp


_PROFILE = _load_profile("rigol_mso5104.yaml")


class _RecordingTransport:
    """Records writes/queries, serves a response table, and pops binary
    payloads from a FIFO (the last one repeats). Also records the socket
    timeout in force for each binary query.
    """

    def __init__(
        self,
        response_map: dict[str, str] | None = None,
        binary: list[bytes] | None = None,
    ) -> None:
        self.writes: list[str] = []
        self.queries: list[str] = []
        self.binary_calls: list[dict[str, Any]] = []
        self.timeout_s = 3.0
        self._responses = response_map or {}
        self._binary = list(binary or [b"BM"])

    def write(self, cmd: str) -> None:
        self.writes.append(cmd)

    def write_many(self, cmds: list[str]) -> None:
        self.writes.extend(cmds)

    def query(self, cmd: str) -> str:
        self.queries.append(cmd)
        if cmd in self._responses:
            return self._responses[cmd]
        if cmd == ":TIM:MAIN:SCAL?":
            return "1.000000E-06"
        if cmd == ":TIM:MAIN:OFFS?":
            return "0.000000E+00"
        if cmd.endswith(":DISP?"):
            return "1"
        if cmd.endswith(":SCAL?"):
            return "1.000000E+00"
        if cmd.endswith(":OFFS?"):
            return "0.000000E+00"
        return ""

    def query_binary(self, cmd: str, recv_max: int = 4 * 1024 * 1024) -> bytes:
        self.queries.append(cmd)
        self.binary_calls.append(
            {"cmd": cmd, "recv_max": recv_max, "timeout_s": self.timeout_s}
        )
        return self._binary.pop(0) if len(self._binary) > 1 else self._binary[0]


def _make_driver(**kw: Any) -> tuple[RigolMso5000, _RecordingTransport]:
    t = _RecordingTransport(**kw)
    drv = RigolMso5000(transport=t, profile=_PROFILE)
    drv.run_status_settle_s = 0
    return drv, t


# ---------------------------------------------------------------------------
# Dispatch + profile
# ---------------------------------------------------------------------------


def test_registry_and_auto_detect_pick_mso5000(monkeypatch: pytest.MonkeyPatch) -> None:
    assert instruments.MODEL_REGISTRY["RIGOL_MSO5104"][0] is RigolMso5000
    monkeypatch.setenv("SCOPE_MCP_HOST", "127.0.0.1")
    monkeypatch.delenv("SCOPE_MCP_MODEL", raising=False)

    class _Stub:
        host, port, timeout_s = "127.0.0.1", 5555, 1.0

        def query(self, cmd: str) -> str:
            return "RIGOL TECHNOLOGIES,MSO5104,MS5A123456789,00.01.03.00.03"

    monkeypatch.setattr(instruments, "ScpiLan", lambda **kw: _Stub())
    scope = open_scope()
    assert isinstance(scope, RigolMso5000)
    assert scope.profile["model"] == "RIGOL_MSO5104"


def test_ds1000z_idn_does_not_match_mso_profile() -> None:
    import re
    assert not re.search(_PROFILE["idn_match"], "RIGOL TECHNOLOGIES,DS1104Z,SN,FW")


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------


def test_screenshot_converts_bmp_to_png_with_bare_disp_data() -> None:
    drv, t = _make_driver(binary=[_make_bmp(_ROWS)])
    result = drv.screenshot(ScreenshotPlan(image_format="PNG"))

    assert result.image_format == "PNG"
    assert result.image_bytes.startswith(PNG_MAGIC)
    assert _decode_png(result.image_bytes)[2] == _ROWS
    # The documented form takes no arguments (DS1000Z sends ON,0,PNG).
    assert t.binary_calls[0]["cmd"] == ":DISP:DATA?"
    # Socket timeout widened for the transfer, then restored.
    assert t.binary_calls[0]["timeout_s"] >= 15.0
    assert t.binary_calls[0]["recv_max"] >= 4 * 1024 * 1024
    assert t.timeout_s == 3.0
    assert result.caveats == []


def test_screenshot_bmp_request_returns_raw_bmp() -> None:
    bmp = _make_bmp(_ROWS)
    drv, _ = _make_driver(binary=[bmp])
    result = drv.screenshot(ScreenshotPlan(image_format="BMP"))
    assert result.image_format == "BMP"
    assert result.image_bytes == bmp


def test_screenshot_passes_through_png_from_newer_firmware() -> None:
    png = PNG_MAGIC + b"payload"
    drv, _ = _make_driver(binary=[png])
    result = drv.screenshot(ScreenshotPlan(image_format="PNG"))
    assert result.image_bytes == png and result.image_format == "PNG"


def test_screenshot_reports_labels_as_caveat_without_sending_them() -> None:
    drv, t = _make_driver(binary=[_make_bmp(_ROWS)])
    result = drv.screenshot(ScreenshotPlan(channel_labels={1: "CLK"}))
    assert result.channel_labels_applied == {}
    assert any("labels were not applied" in c for c in result.caveats)
    assert not any("LAB" in w for w in t.writes)


def test_screenshot_rejects_unknown_format_before_scpi() -> None:
    drv, t = _make_driver()
    with pytest.raises(ValueError, match="unsupported screenshot format"):
        drv.screenshot(ScreenshotPlan(image_format="JPEG"))
    assert t.writes == [] and t.queries == []


def test_screenshot_rejects_unrecognized_payload() -> None:
    from oscilloscope_mcp.transport.scpi_lan import ScpiError
    drv, _ = _make_driver(binary=[b"garbage!"])
    with pytest.raises(ScpiError, match="not a recognized image"):
        drv.screenshot(ScreenshotPlan())


# ---------------------------------------------------------------------------
# Cursors — 1000 × 480 px over 10 × 8 divisions
# ---------------------------------------------------------------------------


def test_time_cursor_pixels_and_type() -> None:
    drv, t = _make_driver(binary=[_make_bmp(_ROWS)])
    plan = ScreenshotPlan(cursor_pairs=[
        CursorPair(label="span", source_channel=2, ax_t_s=-1e-6, bx_t_s=1e-6),
    ])
    result = drv.screenshot(plan)
    assert ":CURS:MODE MAN" in t.writes
    assert ":CURS:MAN:SOUR CHANnel2" in t.writes
    assert ":CURS:MAN:TYPE TIME" in t.writes
    assert ":CURS:MAN:CAX 400" in t.writes   # 1 µs/div → 100 px/div
    assert ":CURS:MAN:CBX 600" in t.writes
    assert len(result.cursors_set) == 1


def test_amplitude_cursor_pixels_and_type() -> None:
    drv, t = _make_driver(binary=[_make_bmp(_ROWS)])
    drv.screenshot(ScreenshotPlan(cursor_pairs=[
        CursorPair(label="v", source_channel=1, ay_v=1.0, by_v=-1.0),
    ]))
    assert ":CURS:MAN:TYPE AMPLitude" in t.writes
    assert ":CURS:MAN:CAY 180" in t.writes   # 1 V/div → 60 px/div, up = smaller y
    assert ":CURS:MAN:CBY 300" in t.writes


def test_multiple_cursor_pairs_emit_caveat() -> None:
    drv, _ = _make_driver(binary=[_make_bmp(_ROWS)])
    result = drv.screenshot(ScreenshotPlan(cursor_pairs=[
        CursorPair(label="a", source_channel=1, ax_t_s=0.0, bx_t_s=1e-7),
        CursorPair(label="b", source_channel=1, ax_t_s=0.0, bx_t_s=2e-7),
    ]))
    assert any("single manual cursor pair" in c for c in result.caveats)


def test_cursor_rejects_out_of_range_channel() -> None:
    drv, _ = _make_driver()
    with pytest.raises(AcquisitionValidationError):
        drv.screenshot(ScreenshotPlan(cursor_pairs=[
            CursorPair(label="x", source_channel=5, ax_t_s=0.0),
        ]))


@pytest.mark.parametrize("t_s, offset_s, expected", [
    (0.0, 0.0, 500),
    (2e-6, 2e-6, 500),     # screen centre tracks the timebase offset
    (-5e-6, 0.0, 0),       # left edge
    (1.0, 0.0, 999),       # clamped right
    (-1.0, 0.0, 0),        # clamped left
])
def test_time_to_px(t_s: float, offset_s: float, expected: int) -> None:
    assert _time_to_px(t_s, 1e-6, offset_s) == expected


@pytest.mark.parametrize("v, offset_v, expected", [
    (0.0, 0.0, 240),
    (-1.0, 1.0, 240),      # +1 V offset puts -1 V at mid-screen
    (4.0, 0.0, 0),         # top edge
    (100.0, 0.0, 0),       # clamped
    (-100.0, 0.0, 479),    # clamped
])
def test_volt_to_px(v: float, offset_v: float, expected: int) -> None:
    assert _volt_to_px(v, 1.0, offset_v) == expected


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------

# 10 fields: format,type,points,count,xinc,xorigin,xref,yinc,yorigin,yref
_PRE_NORMAL = "0,0,4,1,1.000000E-08,-5.000000E-06,0,4.000000E-02,0,128"


def test_normal_read_resets_window_and_uses_xorigin_as_t0() -> None:
    drv, t = _make_driver(
        response_map={":WAV:PREamble?": _PRE_NORMAL},
        binary=[bytes([128, 153, 103, 128])],
    )
    wf = drv.read_waveform("CHAN2")

    assert t.writes == [
        ":WAV:SOURce CHAN2", ":WAV:MODE NORMal", ":WAV:FORMat BYTE",
        ":WAV:STARt 1", ":WAV:STOP 1000",
    ]
    assert ":TRIG:POS?" not in t.queries
    assert wf.t0_s == pytest.approx(-5e-6)
    assert wf.dt_s == pytest.approx(1e-8)
    assert wf.volts == pytest.approx([0.0, 1.0, -1.0, 0.0])


def test_raw_read_chunks_and_skips_trig_pos() -> None:
    pre = "0,2,300000,1,1.000000E-09,-1.500000E-04,0,4.000000E-02,0,128"
    drv, t = _make_driver(
        response_map={":TRIG:STAT?": "STOP", ":WAV:PREamble?": pre},
        binary=[bytes([128]) * 250_000, bytes([128]) * 50_000],
    )
    wf = drv.read_waveform("CHAN1", mode="RAW")

    assert wf.n_samples == 300_000
    assert wf.t0_s == pytest.approx(-1.5e-4)
    assert ":WAV:MODE RAW" in t.writes
    assert ":WAV:STOP 1000" not in t.writes   # NORMal-only window reset
    assert [w for w in t.writes if w.startswith(":WAV:STAR")] == [
        ":WAV:STARt 1", ":WAV:STARt 250001",
    ]
    assert ":TRIG:POS?" not in t.queries


def test_raw_read_requires_stopped_scope() -> None:
    drv, _ = _make_driver(response_map={":TRIG:STAT?": "RUN"})
    with pytest.raises(RuntimeError, match="must be stopped"):
        drv.read_waveform("CHAN1", mode="RAW")


# ---------------------------------------------------------------------------
# Acquisition — memory-depth tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resp, expected", [
    ("AUTO", "AUTO"), ("1M", 1_000_000), ("10k", 10_000), ("25M", 25_000_000),
    ("1.0000E+03", 1_000),  # what firmware 00.01.03.02.02 actually returns
    ("200M", 200_000_000), ("", None), ("junk", None),
])
def test_get_acquire_parses_memory_depth_tokens(resp: str, expected: Any) -> None:
    drv, _ = _make_driver(response_map={
        ":ACQ:TYPE?": "NORM", ":ACQ:AVER?": "2", ":ACQ:MDEP?": resp,
        ":ACQ:SRAT?": "2.500000E+9",
    })
    state = drv.get_acquire()
    assert state.memory_depth == expected
    assert state.sample_rate == pytest.approx(2.5e9)


@pytest.mark.parametrize("requested, token", [
    ("10k", "10k"), (10_000_000, "10M"), ("1e6", "1M"), ("auto", "AUTO"),
])
def test_set_acquire_writes_memory_depth_token(requested: Any, token: str) -> None:
    drv, t = _make_driver()
    drv.set_acquire(AcquireSetup(memory_depth=requested))
    assert f":ACQ:MDEP {token}" in t.writes


def test_set_acquire_rejects_depth_over_channel_limit_before_scpi() -> None:
    drv, t = _make_driver()  # default: all 4 channels displayed → 50M max
    with pytest.raises(AcquisitionValidationError, match="invalid for 4"):
        drv.set_acquire(AcquireSetup(memory_depth="200M"))
    assert t.writes == []


def test_set_acquire_rejects_hres_type() -> None:
    drv, _ = _make_driver()
    with pytest.raises(AcquisitionValidationError):
        drv.set_acquire(AcquireSetup(type="HRESolution"))


@pytest.mark.parametrize("value, expected", [
    ("10k", 10_000), ("1M", 1_000_000), ("25Mpts", 25_000_000),
    ("1.2e4", 12_000), (12000, 12_000), ("  100K ", 100_000),
])
def test_parse_point_count(value: Any, expected: int) -> None:
    assert parse_point_count(value) == expected


def test_parse_point_count_rejects_fraction() -> None:
    with pytest.raises(ValueError):
        parse_point_count("1.5")


# ---------------------------------------------------------------------------
# Channel — profile-driven validation
# ---------------------------------------------------------------------------


def test_channel_rejects_gnd_coupling_and_100m_bwl() -> None:
    drv, t = _make_driver()
    with pytest.raises(AcquisitionValidationError):
        drv.set_channel(ChannelSetup(channel=1, coupling="GND"))
    with pytest.raises(AcquisitionValidationError):
        drv.set_channel(ChannelSetup(channel=1, bw_limit="100M"))
    assert t.writes == []


def test_channel_accepts_high_ratio_probe() -> None:
    drv, t = _make_driver()
    drv.set_channel(ChannelSetup(channel=3, probe=5000, coupling="AC"))
    assert t.writes[:2] == [":CHAN3:PROB 5000", ":CHAN3:COUP AC"]


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


def test_setup_hold_mode_uses_shold_subsystem_and_digital_source() -> None:
    drv, t = _make_driver(response_map={":TRIG:MODE?": "SET"})
    state = drv.set_trigger(TriggerSetup(
        mode="SETup",
        params={"data_source": "D3", "clock_source": "CHAN1", "type": "SETH"},
    ))
    assert t.writes[0] == ":TRIG:MODE SETup"
    assert ":TRIGger:SHOLd:DSRC D3" in t.writes
    assert ":TRIGger:SHOLd:CSRC CHAN1" in t.writes
    assert ":TRIGger:SHOLd:TYPE SETHold" in t.writes
    assert state.mode == "SETup"


def test_edge_trigger_source_is_acline_not_ac() -> None:
    drv, t = _make_driver(response_map={":TRIG:MODE?": "EDGE"})
    drv.set_trigger(TriggerSetup(mode="EDGE", params={"source": "ACLine"}))
    assert ":TRIGger:EDGE:SOURce ACLINE" in t.writes
    with pytest.raises(TriggerValidationError):
        drv.set_trigger(TriggerSetup(mode="EDGE", params={"source": "AC"}))


def test_slope_trigger_rejects_digital_source() -> None:
    drv, _ = _make_driver()
    with pytest.raises(TriggerValidationError):
        drv.set_trigger(TriggerSetup(mode="SLOPe", params={"source": "D0"}))


def test_pulse_trigger_rejects_ds1000z_width_param() -> None:
    drv, _ = _make_driver()
    with pytest.raises(TriggerValidationError, match="width_s"):
        drv.set_trigger(TriggerSetup(mode="PULSe", params={"width_s": 1e-6}))


def test_pattern_trigger_accepts_digital_symbols() -> None:
    drv, t = _make_driver(response_map={":TRIG:MODE?": "PATT"})
    drv.set_trigger(TriggerSetup(mode="PATT", params={"pattern": "H,R,L,X,H,L"}))
    assert ":TRIGger:PATTern:PATTern H,R,L,X,H,L" in t.writes


def test_get_trigger_reads_nth_edge_params() -> None:
    drv, _ = _make_driver(response_map={
        ":TRIG:MODE?": "NEDG",
        ":TRIG:STAT?": "WAIT",
        ":TRIGger:NEDGe:EDGE?": "3",
        ":TRIGger:NEDGe:SOURce?": "D5",
    })
    state = drv.get_trigger()
    assert state.mode == "NEDGe"
    assert state.status == "WAIT"
    assert state.params["nth_edge"] == 3
    assert state.params["source"] == "D5"


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def test_measure_maps_ds1000z_delay_alias_to_rrdelay() -> None:
    drv, t = _make_driver(response_map={
        ":MEAS:ITEM? RRDelay,CHANnel1,CHANnel2": "1.250000E-07",
    })
    out = drv.measure(["RDELAY"], source="CHAN1", source2="CHAN2")
    assert out == {"RDELAY": pytest.approx(1.25e-7)}


def test_measure_supports_mso_only_items() -> None:
    drv, _ = _make_driver(response_map={":MEAS:ITEM? PEDGes,CHANnel1": "4.200000E+01"})
    assert drv.measure(["PEDGES"]) == {"PEDGES": pytest.approx(42)}


def test_measure_statistics_registers_items_then_queries() -> None:
    drv, t = _make_driver(response_map={
        ":MEAS:STAT:ITEM? MAXimum,VPP,CHANnel2": "9.120000E-01",
        ":MEAS:STAT:ITEM? CURRent,VPP,CHANnel2": "9.000000E-01",
    })
    [res] = drv.measure_statistics(["VPP"], source="CHAN2")

    assert t.writes == [":MEAS:STAT:DISP ON", ":MEAS:STAT:ITEM VPP,CHANnel2"]
    stat_queries = [q for q in t.queries if q.startswith(":MEAS:STAT:ITEM?")]
    assert len(stat_queries) == 5            # no COUNt on MSO5000
    assert res.maximum == pytest.approx(0.912)
    assert res.current == pytest.approx(0.9)
    assert res.count is None


def test_measure_statistics_rejects_count_and_dual_source() -> None:
    drv, t = _make_driver()
    with pytest.raises(MeasureValidationError):
        drv.measure_statistics(["VPP"], stat_types=["COUNt"])
    with pytest.raises(MeasureValidationError, match="two-source"):
        drv.measure_statistics(["RRDELAY"])
    assert t.writes == []


# ---------------------------------------------------------------------------
# Caveats use the 10-division screen
# ---------------------------------------------------------------------------


def test_memory_caveat_uses_ten_divisions() -> None:
    # 4 ch → 2 GSa/s, 50 Mpts per channel. 2.3 ms/div × 10 div needs 46 Mpts
    # (fits); × 12 div (DS1000Z geometry) would need 55.2 Mpts (would not).
    fits = compute_caveats(_PROFILE, active_channel_count=4, timebase_s_per_div=2.3e-3)
    assert not any("per-channel memory" in c for c in fits)
    over = compute_caveats(_PROFILE, active_channel_count=4, timebase_s_per_div=3e-3)
    assert any("per-channel memory" in c for c in over)

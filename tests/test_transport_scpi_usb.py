"""Unit tests for the USB (USBTMC / PyVISA) transport and USB dispatch.

A fake VISA layer stands in for PyVISA: resource managers per backend,
resources that record writes, answer queries from a table, and buffer
binary responses so both block-read strategies (NI-VISA whole-message vs
pyvisa-py exact byte count) are exercised.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from oscilloscope_mcp import instruments
from oscilloscope_mcp.instruments import (
    ScopeDispatchError,
    open_scope,
    resolve_usb_selector_from_env,
    resolve_visa_backends_from_env,
)
from oscilloscope_mcp.transport import scpi_usb
from oscilloscope_mcp.transport.scpi_lan import ScpiError, ScpiLan
from oscilloscope_mcp.transport.scpi_usb import ScpiUsb, UsbDiscoveryError

_TMO = -1073807339
_IDN = "RIGOL TECHNOLOGIES,MSO5104,MS5A000000001,00.01.03.00.03"
_NI_NAME = "USB0::0x1AB1::0x0515::MS5A000000001::INSTR"
_PY_NAME = "USB0::6833::1301::MS5A000000001::0::INSTR"


class _VisaError(Exception):
    def __init__(self, code: int, msg: str = "visa error") -> None:
        super().__init__(msg)
        self.error_code = code


class _Resource:
    def __init__(self, responses: dict[str, str] | None = None,
                 binary: dict[str, bytes] | None = None,
                 idn_fails: bool = False) -> None:
        self.responses = {"*IDN?": _IDN, "*OPC?": "1", **(responses or {})}
        self.binary = binary or {}
        self.idn_fails = idn_fails
        self.timeout: int | None = None
        self.write_termination: str | None = None
        self.read_termination: str | None = "\n"
        self.chunk_size: int | None = None
        self.log: list[str] = []
        self.timeouts_seen: list[int | None] = []
        self.closed = False
        self.fail_next: Exception | None = None
        self._out = bytearray()

    def _check(self) -> None:
        if self.closed:
            raise _VisaError(-1073807346, "invalid session")
        if self.fail_next is not None:
            e, self.fail_next = self.fail_next, None
            raise e

    def write(self, cmd: str) -> None:
        self._check()
        self.log.append(cmd)
        if cmd in self.binary:
            self._out += self.binary[cmd]

    def query(self, cmd: str) -> str:
        self._check()
        self.log.append(cmd)
        if cmd == "*IDN?" and self.idn_fails:
            raise _VisaError(-1073807343, "resource busy")
        if cmd not in self.responses:
            raise _VisaError(_TMO, "timeout")
        return self.responses[cmd] + "\n"

    def read_raw(self) -> bytes:
        self._check()
        assert self.read_termination is None, "termination must be off for binary"
        self.timeouts_seen.append(self.timeout)
        data = bytes(self._out)
        self._out.clear()
        return data

    def read_bytes(self, n: int) -> bytes:
        self._check()
        self.timeouts_seen.append(self.timeout)
        if len(self._out) < n:
            self._out.clear()
            raise _VisaError(_TMO, "timeout")
        data = bytes(self._out[:n])
        del self._out[:n]
        return data

    def close(self) -> None:
        self.closed = True


class _RM:
    def __init__(self, resources: dict[str, _Resource], extra: tuple[str, ...] = ()) -> None:
        self.resources = resources
        self.extra = extra
        self.opened: list[str] = []
        self.closed = False

    def list_resources(self) -> tuple[str, ...]:
        return tuple(self.resources) + self.extra

    def open_resource(self, name: str) -> _Resource:
        self.opened.append(name)
        if name not in self.resources:
            raise _VisaError(-1073807343, f"cannot open {name}")
        res = self.resources[name]
        res.closed = False
        return res

    def close(self) -> None:
        self.closed = True


def _factory(**rms: Any):
    """``ivi=_RM(...)`` / ``py=_RM(...)``; an Exception value = unavailable."""
    def make(backend: str) -> _RM:
        rm = rms.get(backend.lstrip("@"))
        if rm is None:
            raise OSError(f"{backend} not installed")
        if isinstance(rm, Exception):
            raise rm
        return rm
    return make


@pytest.fixture(autouse=True)
def _clean_sessions(monkeypatch: pytest.MonkeyPatch):
    scpi_usb.close_all_sessions()
    monkeypatch.setattr(scpi_usb, "_last_good_backend", None)
    for var in ("SCOPE_MCP_HOST", "SCOPE_MCP_PORT", "SCOPE_MCP_MODEL",
                "SCOPE_MCP_USB", "SCOPE_MCP_VISA_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    yield
    scpi_usb.close_all_sessions()


def _usb(rm_factory, selector: str = "auto", **kw: Any) -> ScpiUsb:
    return ScpiUsb(selector=selector, rm_factory=rm_factory, **kw)


# ---------------------------------------------------------------------------
# Discovery + backend selection
# ---------------------------------------------------------------------------


def test_auto_finds_rigol_by_vendor_id_and_configures_session() -> None:
    res = _Resource()
    rm = _RM({_NI_NAME: res}, extra=("ASRL3::INSTR", "USB0::0x0483::0x5740::OTHER::INSTR"))
    t = _usb(_factory(ivi=rm), timeout_s=2.5)

    assert t.query("*IDN?") == _IDN
    assert t.resource_name == _NI_NAME and t.backend == "@ivi"
    assert rm.opened == [_NI_NAME]
    assert (res.timeout, res.write_termination, res.read_termination) == (2500, "\n", "\n")


def test_auto_parses_pyvisa_py_decimal_ids() -> None:
    t = _usb(_factory(py=_RM({_PY_NAME: _Resource()})), backends=("@py",))
    assert t.resource_name == _PY_NAME


def test_falls_back_from_ivi_to_py() -> None:
    t = _usb(_factory(ivi=_RM({}, extra=("ASRL1::INSTR",)), py=_RM({_PY_NAME: _Resource()})))
    assert t.backend == "@py"


def test_falls_back_when_ivi_not_installed() -> None:
    t = _usb(_factory(ivi=OSError("visa32.dll not found"), py=_RM({_PY_NAME: _Resource()})))
    assert t.backend == "@py"


def test_serial_selector_picks_one_of_several() -> None:
    other = "USB0::0x1AB1::0x04CE::DS1ZA000000002::INSTR"
    rm = _RM({_NI_NAME: _Resource(), other: _Resource()})
    assert _usb(_factory(ivi=rm), selector="ms5a000000001").resource_name == _NI_NAME


def test_auto_with_several_scopes_asks_for_serial() -> None:
    other = "USB0::0x1AB1::0x04CE::DS1ZA000000002::INSTR"
    rm = _RM({_NI_NAME: _Resource(), other: _Resource()})
    with pytest.raises(UsbDiscoveryError, match="several RIGOL USB scopes"):
        _usb(_factory(ivi=rm)).query("*IDN?")


def test_no_scope_error_lists_every_backend() -> None:
    t = _usb(_factory(ivi=_RM({}), py=OSError("no libusb")))
    with pytest.raises(UsbDiscoveryError) as ei:
        t.query("*IDN?")
    msg = str(ei.value)
    assert "@ivi: no RIGOL USB scope" in msg and "@py: backend unavailable" in msg


def test_explicit_resource_string_is_opened_as_given() -> None:
    rm = _RM({_NI_NAME: _Resource()})
    assert _usb(_factory(ivi=rm), selector=_NI_NAME).resource_name == _NI_NAME


def test_busy_scope_reports_another_program() -> None:
    rm = _RM({_NI_NAME: _Resource(idn_fails=True)})
    with pytest.raises(UsbDiscoveryError, match="another program may hold it"):
        _usb(_factory(ivi=rm), backends=("@ivi",)).query("*IDN?")


def test_missing_pyvisa_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyvisa", None)  # import → ImportError
    with pytest.raises(ScpiError, match=r"pip install"):
        ScpiUsb().query("*IDN?")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def test_write_many_syncs_with_opc() -> None:
    res = _Resource()
    t = _usb(_factory(ivi=_RM({_NI_NAME: res})))
    t.write_many([":CHAN1:SCAL 1", ":CHAN1:OFFS 0"])
    assert res.log[-3:] == [":CHAN1:SCAL 1", ":CHAN1:OFFS 0", "*OPC?"]


def test_query_requires_question_mark() -> None:
    t = _usb(_factory(ivi=_RM({_NI_NAME: _Resource()})))
    with pytest.raises(ScpiError, match="requires a '\\?'"):
        t.query(":RUN")


_PAYLOAD = b"BM\x00\n\x0a\xffdata\n"  # contains 0x0A bytes on purpose


def _block(payload: bytes) -> bytes:
    n = str(len(payload)).encode()
    return b"#" + str(len(n)).encode() + n + payload + b"\n"


def test_binary_on_ivi_reads_whole_message_and_restores_termination() -> None:
    res = _Resource(binary={":DISP:DATA?": _block(_PAYLOAD)})
    t = _usb(_factory(ivi=_RM({_NI_NAME: res})))
    assert t.query_binary(":DISP:DATA?") == _PAYLOAD
    assert res.read_termination == "\n"


def test_binary_on_py_reads_exact_byte_count() -> None:
    res = _Resource(binary={":WAV:DATA?": _block(_PAYLOAD)})
    t = _usb(_factory(py=_RM({_PY_NAME: res})), backends=("@py",))
    assert t.query_binary(":WAV:DATA?") == _PAYLOAD


def test_binary_on_py_requires_trailing_newline_and_drops_session() -> None:
    res = _Resource(binary={":WAV:DATA?": _block(_PAYLOAD)[:-1] + b"X"})
    rm = _RM({_PY_NAME: res})
    t = _usb(_factory(py=rm), backends=("@py",))
    with pytest.raises(ScpiError, match="not followed by a newline"):
        t.query_binary(":WAV:DATA?")
    assert res.closed and rm.closed  # stream state unknown → session dropped


def test_binary_over_recv_max_rejected() -> None:
    res = _Resource(binary={":DISP:DATA?": _block(_PAYLOAD)})
    t = _usb(_factory(ivi=_RM({_NI_NAME: res})))
    with pytest.raises(ScpiError, match="exceeds recv_max"):
        t.query_binary(":DISP:DATA?", recv_max=4)


def test_timeout_is_reported_not_retried() -> None:
    res = _Resource()
    t = _usb(_factory(ivi=_RM({_NI_NAME: res})))
    t.query("*IDN?")  # open
    with pytest.raises(ScpiError, match="timed out after 3 s"):
        t.query(":TRIG:POS?")
    assert res.log.count(":TRIG:POS?") == 1
    assert res.closed


def test_lost_connection_on_reused_session_reopens_once() -> None:
    res = _Resource(responses={":TRIG:STAT?": "STOP"})
    rm = _RM({_NI_NAME: res})
    t = _usb(_factory(ivi=rm))
    t.query("*IDN?")
    res.fail_next = _VisaError(-1073807194, "connection lost")
    assert t.query(":TRIG:STAT?") == "STOP"
    assert rm.opened == [_NI_NAME, _NI_NAME]


def test_session_is_shared_across_transport_instances() -> None:
    rm = _RM({_NI_NAME: _Resource()})
    factory = _factory(ivi=rm)
    _usb(factory).query("*IDN?")
    _usb(factory).query("*IDN?")
    assert rm.opened == [_NI_NAME]


def test_timeout_setter_updates_open_session() -> None:
    res = _Resource()
    t = _usb(_factory(ivi=_RM({_NI_NAME: res})))
    t.query("*IDN?")
    t.timeout_s = 15.0
    assert res.timeout == 15000


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("", None), ("0", None), ("off", None), ("1", "auto"), ("AUTO", "auto"),
    ("yes", "auto"), ("MS5A000000001", "MS5A000000001"), (_NI_NAME, _NI_NAME),
])
def test_usb_selector_from_env(monkeypatch: pytest.MonkeyPatch, raw: str, expected: Any) -> None:
    monkeypatch.setenv("SCOPE_MCP_USB", raw)
    assert resolve_usb_selector_from_env() == expected


def test_visa_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_visa_backends_from_env() == ("@ivi", "@py")
    monkeypatch.setenv("SCOPE_MCP_VISA_BACKEND", "@py, @ivi")
    assert resolve_visa_backends_from_env() == ("@py", "@ivi")
    monkeypatch.setenv("SCOPE_MCP_VISA_BACKEND", "@ni")
    with pytest.raises(ScopeDispatchError, match="invalid"):
        resolve_visa_backends_from_env()


def _capture_transports(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    made: list[Any] = []

    class _Stub:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw
            made.append(self)

        def query(self, cmd: str) -> str:
            return _IDN

    monkeypatch.setattr(instruments, "ScpiUsb", lambda **kw: _Stub(kind="usb", **kw))
    monkeypatch.setattr(instruments, "ScpiLan", lambda **kw: _Stub(kind="lan", **kw))
    return made


def test_open_scope_uses_usb_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    made = _capture_transports(monkeypatch)
    monkeypatch.setenv("SCOPE_MCP_USB", "auto")
    monkeypatch.setenv("SCOPE_MCP_HOST", "10.0.0.5")  # USB wins over env host
    scope = open_scope(timeout_s=7.0)
    assert made[0].kw["kind"] == "usb"
    assert made[0].kw["selector"] == "auto" and made[0].kw["timeout_s"] == 7.0
    assert scope.profile["model"] == "RIGOL_MSO5104"


def test_explicit_host_forces_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    made = _capture_transports(monkeypatch)
    monkeypatch.setenv("SCOPE_MCP_USB", "auto")
    open_scope(host="10.0.0.9")
    assert made[0].kw["kind"] == "lan" and made[0].kw["host"] == "10.0.0.9"


def test_no_transport_configured_mentions_both(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ScopeDispatchError, match="SCOPE_MCP_USB=auto"):
        open_scope()


def test_real_usb_transport_class_is_wired() -> None:
    assert instruments.ScpiUsb is ScpiUsb and instruments.ScpiLan is ScpiLan


# ---------------------------------------------------------------------------
# Driver over USB
# ---------------------------------------------------------------------------


def test_mso5000_screenshot_over_usb_widens_timeout() -> None:
    from oscilloscope_mcp.instruments import _load_profile
    from oscilloscope_mcp.instruments._base import ScreenshotPlan
    from oscilloscope_mcp.instruments.rigol_mso5000 import RigolMso5000
    from tests.test_image import _ROWS, _decode_png, _make_bmp

    res = _Resource(binary={":DISP:DATA?": _block(_make_bmp(_ROWS))})
    t = _usb(_factory(ivi=_RM({_NI_NAME: res})))
    drv = RigolMso5000(transport=t, profile=_load_profile("rigol_mso5104.yaml"))
    result = drv.screenshot(ScreenshotPlan())

    assert _decode_png(result.image_bytes)[2] == _ROWS
    assert res.timeouts_seen == [15000]   # widened for the transfer
    assert res.timeout == 3000            # restored afterwards

"""USB (USBTMC) SCPI transport via PyVISA.

Same interface as :class:`~oscilloscope_mcp.transport.scpi_lan.ScpiLan`
(``write`` / ``write_many`` / ``query`` / ``query_binary`` / ``timeout_s``)
so every driver works unchanged over either link.

Install the optional extra first: ``pip install -e ".[usb]"``.

Two VISA backends can reach a USB scope, depending on which Windows driver
is bound to its USB interface. Each backend only sees devices bound to its
kind of driver, so both are tried (last-good first):

- ``@ivi`` — NI-VISA / IVI VISA (``visa32.dll``): the scope is on the
  native USBTMC driver (installed with NI-VISA or RIGOL UltraSigma).
- ``@py`` — pyvisa-py + pyusb + libusb: the scope is bound to WinUSB
  (e.g. via Zadig).

Binary blocks must be read differently on each (learned on DS1000Z / DHO
hardware): NI-VISA frames reads as USBTMC messages, and reading a byte
count that does not land on a message boundary can wedge the scope, so
``@ivi`` reads the whole message. pyvisa-py times out on large
message-based reads, so ``@py`` reads exactly the declared byte count.

Connection model: unlike the LAN transport (fresh socket per call), USB
keeps one open session per scope for the life of the server process,
because opening a VISA session costs far more than a TCP connect. While
the server runs, other programs (UltraSigma, another MCP server) cannot
reliably use the scope over USB.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from oscilloscope_mcp.transport.scpi_lan import ScpiError

RIGOL_USB_VID = 0x1AB1
DEFAULT_BACKENDS: tuple[str, ...] = ("@ivi", "@py")

# VI_ERROR_TMO (0xBFFF0015 as a signed 32-bit int): the instrument did not
# answer in time — usually an unsupported query on RIGOL firmware.
_VI_ERROR_TMO = -1073807339

_INSTALL_HINT = (
    "USB transport needs PyVISA. Install the extra: "
    "pip install -e \".[usb]\"  (or: pip install pyvisa pyvisa-py pyusb libusb-package)"
)


class UsbDiscoveryError(ScpiError):
    """No usable USB scope matched the selector on any backend."""


@dataclass
class _Session:
    rm: Any
    resource: Any
    backend: str
    resource_name: str
    idn: str
    lock: threading.RLock


_SESSIONS: dict[tuple[str, tuple[str, ...]], _Session] = {}
_SESSIONS_LOCK = threading.Lock()
_last_good_backend: str | None = None


class ScpiUsb:
    """SCPI client over USBTMC.

    Parameters
    ----------
    selector : str
        ``"auto"`` (the single RIGOL USB scope present), a VISA resource
        string (``USB0::0x1AB1::0x0515::MS5A...::INSTR``), or a serial
        number to pick one of several RIGOL scopes.
    timeout_s : float
        I/O timeout for every operation (adjustable via :attr:`timeout_s`).
    backends : tuple[str, ...]
        VISA backends to try, in order.
    rm_factory : callable, optional
        ``backend -> ResourceManager``; defaults to ``pyvisa.ResourceManager``.
        Injected by tests.
    """

    def __init__(
        self,
        selector: str = "auto",
        timeout_s: float = 3.0,
        backends: tuple[str, ...] | None = None,
        rm_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.selector = (selector or "auto").strip()
        self.backends = tuple(backends or DEFAULT_BACKENDS)
        self._timeout_s = float(timeout_s)
        self._rm_factory = rm_factory

    # -- public interface (mirrors ScpiLan) ---------------------------------

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    @timeout_s.setter
    def timeout_s(self, value: float) -> None:
        self._timeout_s = float(value)
        session = _SESSIONS.get(self._key)
        if session is not None:
            session.resource.timeout = self._timeout_ms

    @property
    def resource_name(self) -> str:
        """VISA resource string of the connected scope (opens if needed)."""
        return self._session().resource_name

    @property
    def backend(self) -> str:
        """VISA backend in use (``@ivi`` or ``@py``; opens if needed)."""
        return self._session().backend

    def write(self, cmd: str) -> None:
        self._run(cmd, lambda res, _s: res.write(cmd))

    def write_many(self, cmds: list[str]) -> None:
        """Send commands in order, then block on ``*OPC?`` until applied
        (same contract as :meth:`ScpiLan.write_many`).
        """
        if not cmds:
            return

        def op(res: Any, _s: _Session) -> None:
            for c in cmds:
                res.write(c)
            res.query("*OPC?")

        self._run("; ".join(cmds), op)

    def query(self, cmd: str, recv_max: int = 65536) -> str:
        if "?" not in cmd:
            raise ScpiError(f"query() requires a '?' in command, got: {cmd!r}")
        return self._run(cmd, lambda res, _s: str(res.query(cmd)).rstrip("\r\n"))

    def query_binary(self, cmd: str, recv_max: int = 4 * 1024 * 1024) -> bytes:
        """Send a query whose response is an IEEE-488.2 definite-length
        block (``#<n><len><payload>\\n``) and return the payload.
        """
        if "?" not in cmd:
            raise ScpiError(f"query_binary() requires a '?' in command, got: {cmd!r}")

        def op(res: Any, session: _Session) -> bytes:
            res.write(cmd)
            if session.backend == "@ivi":
                return _read_block_message(res, recv_max)
            return _read_block_bytecount(res, recv_max)

        return self._run(cmd, op)

    def close(self) -> None:
        """Release this scope's USB session (the next call reopens it)."""
        _drop_session(self._key)

    # -- internals -----------------------------------------------------------

    @property
    def _key(self) -> tuple[str, tuple[str, ...]]:
        return (self.selector, self.backends)

    @property
    def _timeout_ms(self) -> int:
        return max(1, int(round(self._timeout_s * 1000)))

    def _session(self) -> _Session:
        with _SESSIONS_LOCK:
            session = _SESSIONS.get(self._key)
            if session is None:
                session = self._open()
                _SESSIONS[self._key] = session
        session.resource.timeout = self._timeout_ms
        return session

    def _run(self, cmd: str, op: Callable[[Any, _Session], Any]) -> Any:
        """Run ``op`` on the session. A timeout is reported, not retried
        (the instrument ignored the command). Any other I/O failure on a
        reused session (scope power-cycled, cable re-plugged) reopens the
        session and retries once — commands are idempotent setpoints.
        """
        for attempt in (1, 2):
            reused = self._key in _SESSIONS
            session = self._session()
            try:
                with session.lock:
                    return op(session.resource, session)
            except ScpiError:
                _drop_session(self._key)  # framing error: stream state unknown
                raise
            except Exception as e:  # pyvisa / pyusb / OS errors
                _drop_session(self._key)
                if _is_timeout(e):
                    raise ScpiError(
                        f"USB SCPI {cmd!r} timed out after {self._timeout_s:g} s "
                        "(command unsupported, or the scope is busy)"
                    ) from e
                if attempt == 1 and reused:
                    continue
                raise ScpiError(f"USB SCPI {cmd!r} failed: {type(e).__name__}: {e}") from e
        raise AssertionError("unreachable")  # pragma: no cover

    def _open(self) -> _Session:
        global _last_good_backend
        factory = self._rm_factory or _pyvisa_rm_factory
        order = list(dict.fromkeys(
            ([_last_good_backend] if _last_good_backend in self.backends else [])
            + list(self.backends)
        ))
        problems: list[str] = []
        for backend in order:
            try:
                rm = factory(backend)
            except ScpiError:
                raise  # PyVISA missing: no backend can work
            except Exception as e:
                problems.append(f"{backend}: backend unavailable ({type(e).__name__}: {e})")
                continue
            try:
                name = _select_resource(rm.list_resources(), self.selector)
            except UsbDiscoveryError as e:
                problems.append(f"{backend}: {e}")
                _close_quietly(rm)
                continue
            except Exception as e:
                problems.append(f"{backend}: enumeration failed ({type(e).__name__}: {e})")
                _close_quietly(rm)
                continue
            res = None
            try:
                res = rm.open_resource(name)
                res.timeout = self._timeout_ms
                res.write_termination = "\n"
                res.read_termination = "\n"
                res.chunk_size = 1024 * 1024
                # No res.clear(): DS1000Z firmware mishandles USBTMC
                # INITIATE_CLEAR and stops answering until power-cycled.
                idn = str(res.query("*IDN?")).strip()
            except Exception as e:
                problems.append(
                    f"{backend}: found {name} but could not talk to it "
                    f"({type(e).__name__}: {e}) — another program may hold it"
                )
                _close_quietly(res)
                _close_quietly(rm)
                continue
            _last_good_backend = backend
            return _Session(rm=rm, resource=res, backend=backend,
                            resource_name=name, idn=idn, lock=threading.RLock())

        raise UsbDiscoveryError(
            f"no USB scope matching {self.selector!r} could be opened:\n  "
            + "\n  ".join(problems or ["no backends configured"])
            + "\nCheck the USB-B cable and power, close other programs using the "
            "scope (UltraSigma, other MCP servers), and make sure its USB "
            "interface has a driver: native USBTMC needs NI-VISA (@ivi); "
            "WinUSB via Zadig uses pyvisa-py (@py)."
        )


# ---------------------------------------------------------------------------
# Resource discovery
# ---------------------------------------------------------------------------


def _select_resource(resources: Any, selector: str) -> str:
    """Pick the VISA resource for ``selector`` from ``list_resources()``.

    Resource formats differ by backend — pyvisa-py:
    ``USB0::6833::1305::<serial>::0::INSTR`` (decimal), NI-VISA:
    ``USB0::0x1AB1::0x0515::<serial>::INSTR`` (hex). Field 1 is the vendor
    id and field 3 the serial in both.
    """
    usb = [str(r) for r in resources if str(r).upper().startswith("USB")]
    if selector.upper().startswith("USB"):
        want = selector.upper()
        for r in usb:
            if r.upper() == want:
                return r
        # Not enumerated under that exact spelling (e.g. hex vs decimal);
        # let the backend try to open it as given.
        return selector

    rigol: list[tuple[str, str]] = []
    for r in usb:
        parts = r.split("::")
        if len(parts) < 5:
            continue
        try:
            vid = int(parts[1], 0)  # base 0 parses "6833" and "0x1AB1"
        except ValueError:
            continue
        if vid == RIGOL_USB_VID:
            rigol.append((r, parts[3]))

    if selector.lower() != "auto":
        for r, serial in rigol:
            if serial.upper() == selector.upper():
                return r
        found = ", ".join(s for _, s in rigol) or "none"
        raise UsbDiscoveryError(
            f"no RIGOL USB scope with serial {selector!r} (found: {found})"
        )
    if not rigol:
        raise UsbDiscoveryError(
            "no RIGOL USB scope (vendor 0x1AB1) found; USB resources: "
            + (", ".join(usb) or "none")
        )
    if len(rigol) > 1:
        serials = ", ".join(s for _, s in rigol)
        raise UsbDiscoveryError(
            f"several RIGOL USB scopes found ({serials}); set SCOPE_MCP_USB to "
            "the serial of the one to use"
        )
    return rigol[0][0]


# ---------------------------------------------------------------------------
# Binary block readers
# ---------------------------------------------------------------------------


def _parse_block_header(raw: bytes) -> tuple[int, int]:
    """Return ``(payload_start, payload_length)`` for ``#N<len>...``."""
    if raw[:1] != b"#":
        raise ScpiError(f"binary block missing '#' marker, got {raw[:8]!r}")
    try:
        n = int(raw[1:2])
    except ValueError as e:
        raise ScpiError(f"binary block length-of-length not a digit: {raw[:8]!r}") from e
    if not 1 <= n <= 9 or len(raw) < 2 + n:
        raise ScpiError(f"invalid binary block header {raw[:12]!r}")
    try:
        length = int(raw[2:2 + n].decode("ascii"))
    except ValueError as e:
        raise ScpiError(f"binary block payload length not numeric: {raw[:12]!r}") from e
    return 2 + n, length


def _read_block_message(res: Any, recv_max: int) -> bytes:
    """NI-VISA: read the complete USBTMC message (stops on EOM, never mid-
    transaction), with termination matching off since payloads contain 0x0A.
    """
    saved = res.read_termination
    res.read_termination = None
    try:
        raw = bytes(res.read_raw())
    finally:
        res.read_termination = saved
    start, length = _parse_block_header(raw)
    if length > recv_max:
        raise ScpiError(f"binary block payload {length} bytes exceeds recv_max {recv_max}")
    payload = raw[start:start + length]
    if len(payload) != length:
        raise ScpiError(f"truncated binary block: header {length} bytes, got {len(payload)}")
    return payload


def _read_block_bytecount(res: Any, recv_max: int) -> bytes:
    """pyvisa-py: read exactly the declared byte count plus the trailing
    newline (message-based reads time out on large blocks here).
    """
    head = bytes(res.read_bytes(2))
    if head[:1] != b"#":
        raise ScpiError(f"binary block missing '#' marker, got {head!r}")
    try:
        n = int(head[1:2])
    except ValueError as e:
        raise ScpiError(f"binary block length-of-length not a digit: {head!r}") from e
    if not 1 <= n <= 9:
        raise ScpiError(f"invalid binary block header {head!r}")
    _, length = _parse_block_header(head + bytes(res.read_bytes(n)))
    if length > recv_max:
        raise ScpiError(f"binary block payload {length} bytes exceeds recv_max {recv_max}")
    raw = bytes(res.read_bytes(length + 1))
    if len(raw) != length + 1 or raw[-1:] != b"\n":
        raise ScpiError(
            f"binary block of {length} bytes not followed by a newline "
            f"(got {len(raw)} bytes, last {raw[-1:]!r})"
        )
    return raw[:-1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pyvisa_rm_factory(backend: str) -> Any:
    try:
        import pyvisa
    except ImportError as e:
        raise ScpiError(_INSTALL_HINT) from e
    if backend == "@py":
        _prefer_bundled_libusb1()
    return pyvisa.ResourceManager(backend)


def _prefer_bundled_libusb1() -> None:
    """Point pyusb at the libusb-1.0 bundled with ``libusb-package``, so a
    stray system ``libusb0`` doesn't hide a WinUSB-bound scope from @py.
    No-op when either package is missing.
    """
    try:
        import libusb_package
        import usb.backend.libusb1 as libusb1
    except ImportError:
        return
    backend = libusb_package.get_libusb1_backend()
    if backend is not None:
        libusb1.get_backend = lambda *a, **k: backend


def _is_timeout(exc: BaseException) -> bool:
    code = getattr(exc, "error_code", None)
    try:
        return code is not None and int(code) == _VI_ERROR_TMO
    except (TypeError, ValueError):
        return False


def _close_quietly(obj: Any) -> None:
    if obj is None:
        return
    try:
        obj.close()
    except Exception:
        pass


def _drop_session(key: tuple[str, tuple[str, ...]]) -> None:
    with _SESSIONS_LOCK:
        session = _SESSIONS.pop(key, None)
    if session is not None:
        _close_quietly(session.resource)
        _close_quietly(session.rm)


def close_all_sessions() -> None:
    """Close every cached USB session (tests / clean shutdown)."""
    for key in list(_SESSIONS):
        _drop_session(key)

import io
import os
import sys
import time
import pathlib
import threading
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYSTRAY_BACKEND", "dummy")

import screenguard as sg  # noqa: E402


# ---------------------------------------------------------------- guard --ipc with a fake camera/engine
class FakeCap:
    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.released = True


class Script:
    """Which faces the fake engine reports; changed by the test while the guard runs."""
    faces = "owner"


def fake_engine(*_a, **_k):
    eng = types.SimpleNamespace()

    def detect(frame):
        row = np.array([0, 0, 200, 200, 60, 80, 140, 80, 100, 125, 70, 160, 130, 160, 0.9])
        if Script.faces == "none":
            return []
        return [sg.Face((0, 0, 200, 200), row, True, 200 / 640)]

    def embed(frame, face):
        v = np.zeros(128, np.float32)
        v[0 if Script.faces == "owner" else 1] = 1
        return v

    eng.detect, eng.embed = detect, embed
    return eng


class Pipe(io.TextIOBase):
    """A blocking line stream the test can write commands into."""
    def __init__(self):
        self.lines, self.cv, self.closed_ = [], threading.Condition(), False

    def put(self, s):
        with self.cv:
            self.lines.append(s + "\n"); self.cv.notify_all()

    def close_(self):
        with self.cv:
            self.closed_ = True; self.cv.notify_all()

    def __iter__(self):
        while True:
            with self.cv:
                while not self.lines and not self.closed_:
                    self.cv.wait()
                if self.lines:
                    yield self.lines.pop(0)
                elif self.closed_:
                    return


def test_guard_ipc_protocol(monkeypatch, tmp_path):
    caps = []
    monkeypatch.setattr(sg, "ensure_models", lambda: {"detector": "d", "recognizer": "r"})
    monkeypatch.setattr(sg, "FaceEngine", fake_engine)
    monkeypatch.setattr(sg, "load_owner", lambda p: np.eye(1, 128, 0, dtype=np.float32))
    monkeypatch.setattr(sg, "open_camera", lambda i: caps.append(FakeCap()) or caps[-1])
    locks = []
    monkeypatch.setattr(sg, "os_lock", lambda: locks.append(time.time()) or True)

    args = sg.build_parser().parse_args(["run", "--mode", "lock", "--fps", "50", "--stranger-delay", "0.2",
                                         "--owner-grace", "0.2"])
    inp, out = Pipe(), io.StringIO()
    ipc = sg.IPC(instream=inp, outstream=out)
    t = threading.Thread(target=sg.cmd_run, args=(args, ipc), daemon=True)
    Script.faces = "owner"
    t.start()
    time.sleep(0.4)
    Script.faces = "stranger"
    time.sleep(0.8)
    assert locks, "stranger alone should trigger the lock"
    inp.put("pause"); time.sleep(0.4)
    assert getattr(caps[0], "released", False), "camera must be released while paused"
    Script.faces = "owner"
    inp.put("resume"); time.sleep(0.4)
    inp.close_()  # supervisor gone -> guard must exit
    t.join(3)
    assert not t.is_alive()

    events = [l for l in out.getvalue().splitlines() if l.startswith("@")]
    import json
    states = [json.loads(e[1:]).get("state") for e in events if '"state"' in e and '"event": "state"' in e]
    assert json.loads(events[0][1:])["event"] == "ready"
    assert states[:3] == ["blank", "ok", "paused"], states  # blank -> (paused un-blanks) -> paused
    assert "ok" in states[3:]


# ---------------------------------------------------------------- tray
@pytest.fixture
def tray(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENGUARD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    import importlib
    import tray as tr
    tr = importlib.reload(tr)
    return tr


def test_icon_and_status(tray):
    for st in tray.COLORS:
        img = tray.make_icon(st)
        assert img.size == (64, 64)
    ctl = tray.Controller()
    ctl.handle_line('@{"event": "state", "state": "blank", "reason": "stranger looking"}\n')
    assert ctl.state == "blank" and "stranger" in ctl.status_text()
    ctl.handle_line("plain log line\n")
    assert "plain log line" in tray.LOG_FILE.read_text(encoding="utf-8")


def test_guard_cmd_uses_config(tray):
    tray.save_config({**tray.DEFAULTS, "mode": "lock", "pin": "1234", "threshold": 0.45})
    ctl = tray.Controller()
    cmd = ctl.guard_cmd()
    assert "--ipc" in cmd and cmd[cmd.index("--mode") + 1] == "lock"
    assert cmd[cmd.index("--pin") + 1] == "1234" and "0.45" in cmd


@pytest.mark.skipif(sys.platform != "linux", reason="XDG autostart is Linux only")
def test_autostart_linux(tray):
    assert not tray.Autostart.is_enabled()
    tray.Autostart.enable()
    text = tray.Autostart._linux_desktop().read_text()
    assert "tray.py" in text and "X-GNOME-Autostart-enabled=true" in text
    assert tray.Autostart.is_enabled()
    tray.Autostart.disable()
    assert not tray.Autostart.is_enabled()


def test_supervisor_with_fake_guard(tray, tmp_path, monkeypatch):
    """Supervisor launches the guard, parses its events, pauses it and restarts it after a crash."""
    fake = tmp_path / "fake_guard.py"
    fake.write_text(
        "import sys, json\n"
        "print('@' + json.dumps({'event': 'ready'}), flush=True)\n"
        "print('@' + json.dumps({'event': 'state', 'state': 'blank', 'reason': 'stranger looking'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    if line.strip() == 'pause': print('@' + json.dumps({'event': 'state', 'state': 'paused'}), flush=True)\n"
        "    if line.strip() == 'crash': sys.exit(3)\n"
        "    if line.strip() == 'quit': break\n"
    )
    monkeypatch.setattr(tray, "GUARD", fake)
    tray.OWNER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tray.OWNER_FILE.write_bytes(b"x")
    ctl = tray.Controller()
    ctl.start_threads()

    def wait(cond, t=5):
        end = time.time() + t
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.05)
        return False

    assert wait(lambda: ctl.state == "blank")
    ctl.pause()
    assert wait(lambda: ctl.state == "paused")
    ctl.resume()
    first = ctl.proc
    ctl.send("crash")
    assert wait(lambda: ctl.proc is not None and ctl.proc is not first, 8), "guard should be restarted"
    ctl.quit()
    assert wait(lambda: ctl.proc is None)

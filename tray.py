#!/usr/bin/env python3
"""
faceCheck tray app: a system-tray icon (Windows taskbar / macOS menu bar / Linux panel)
that supervises the guard process and starts automatically when you log in.

    python tray.py                     # start the tray app (enables autostart on first run)
    python tray.py --enable-autostart  # register to start at login, then exit
    python tray.py --disable-autostart # remove the login entry, then exit

Icon colors:  green = protected   red = screen blanked   yellow = paused
              blue = enrolling    grey = starting / error
"""
from __future__ import annotations

import json
import os
import platform
import plistlib
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARD = HERE / "screenguard.py"
APP_DIR = Path(os.environ.get("SCREENGUARD_HOME", Path.home() / ".screenguard"))
CONFIG_FILE = APP_DIR / "config.json"
LOG_FILE = APP_DIR / "facecheck.log"
OWNER_FILE = APP_DIR / "owner.npz"
SYSTEM = platform.system()
APP_ID = "com.sundist.facecheck"
LOCK_PORT = 47391  # localhost port used as a single-instance lock

DEFAULTS = {
    "mode": "overlay",        # overlay | lock | off
    "threshold": 0.40,
    "stranger_delay": 1.5,
    "absent_delay": 0,
    "camera": 0,
    "pin": "",
    "preview": False,
    "first_run": True,
}

COLORS = {
    "ok": (46, 160, 67),
    "blank": (218, 54, 51),
    "paused": (210, 153, 34),
    "enroll": (56, 132, 255),
    "starting": (140, 140, 140),
    "error": (140, 140, 140),
}
LABELS = {
    "ok": "Protected",
    "blank": "Screen blanked",
    "paused": "Paused",
    "enroll": "Enrolling face…",
    "starting": "Starting…",
    "error": "Error",
}


# --------------------------------------------------------------------------- config
def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: dict) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def gui_python() -> str:
    """Interpreter for background use (pythonw on Windows so no console window appears)."""
    exe = Path(sys.executable)
    if SYSTEM == "Windows" and exe.name.lower() == "python.exe":
        w = exe.with_name("pythonw.exe")
        if w.exists():
            return str(w)
    return str(exe)


def open_path(path: Path) -> None:
    if SYSTEM == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif SYSTEM == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# --------------------------------------------------------------------------- autostart
class Autostart:
    """Start at login: Windows Run key, macOS LaunchAgent, Linux XDG autostart."""

    @staticmethod
    def command() -> list[str]:
        return [gui_python(), str(Path(__file__).resolve())]

    @staticmethod
    def _mac_plist() -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{APP_ID}.plist"

    @staticmethod
    def _linux_desktop() -> Path:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return base / "autostart" / "facecheck.desktop"

    @classmethod
    def is_enabled(cls) -> bool:
        if SYSTEM == "Windows":
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as k:
                    winreg.QueryValueEx(k, "faceCheck")
                return True
            except OSError:
                return False
        if SYSTEM == "Darwin":
            return cls._mac_plist().exists()
        return cls._linux_desktop().exists()

    @classmethod
    def enable(cls) -> None:
        cmd = cls.command()
        if SYSTEM == "Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run",
                                0, winreg.KEY_SET_VALUE) as k:
                winreg.SetValueEx(k, "faceCheck", 0, winreg.REG_SZ, " ".join(f'"{c}"' for c in cmd))
        elif SYSTEM == "Darwin":
            p = cls._mac_plist()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                plistlib.dump({
                    "Label": APP_ID,
                    "ProgramArguments": cmd,
                    "RunAtLoad": True,
                    "ProcessType": "Interactive",
                    "WorkingDirectory": str(HERE),
                    "StandardOutPath": str(LOG_FILE),
                    "StandardErrorPath": str(LOG_FILE),
                }, f)
        else:
            p = cls._linux_desktop()
            p.parent.mkdir(parents=True, exist_ok=True)
            exec_line = " ".join(f'"{c}"' for c in cmd)
            p.write_text(
                "[Desktop Entry]\n"
                "Type=Application\n"
                "Name=faceCheck\n"
                "Comment=Blank the screen when a stranger looks at it\n"
                f"Exec={exec_line}\n"
                f"Path={HERE}\n"
                "Icon=security-high\n"
                "Terminal=false\n"
                "X-GNOME-Autostart-enabled=true\n"
                "X-GNOME-Autostart-Delay=5\n",
                encoding="utf-8",
            )

    @classmethod
    def disable(cls) -> None:
        if SYSTEM == "Windows":
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run",
                                    0, winreg.KEY_SET_VALUE) as k:
                    winreg.DeleteValue(k, "faceCheck")
            except OSError:
                pass
        elif SYSTEM == "Darwin":
            cls._mac_plist().unlink(missing_ok=True)
        else:
            cls._linux_desktop().unlink(missing_ok=True)


# --------------------------------------------------------------------------- icon
def make_icon(state: str, size: int = 64):
    from PIL import Image, ImageDraw
    color = COLORS.get(state, COLORS["error"])
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = size // 16
    d.ellipse((m, m, size - m, size - m), fill=color + (255,))
    # an "eye": white almond + dark pupil
    cx, cy = size / 2, size / 2
    ew, eh = size * 0.34, size * 0.20
    d.ellipse((cx - ew, cy - eh, cx + ew, cy + eh), fill=(255, 255, 255, 255))
    r = size * 0.11
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(30, 30, 30, 255))
    if state == "paused":
        d.line((size * 0.2, size * 0.8, size * 0.8, size * 0.2), fill=(30, 30, 30, 255), width=max(2, size // 12))
    return img


# --------------------------------------------------------------------------- controller
class Controller:
    def __init__(self, icon_factory=None):
        APP_DIR.mkdir(parents=True, exist_ok=True)
        self.cfg = load_config()
        self.state = "starting"
        self.detail = ""
        self.paused = False
        self.pause_until = 0.0
        self.enrolling = False
        self.stopping = False
        self.restart_requested = False
        self.proc: subprocess.Popen | None = None
        self.proc_lock = threading.Lock()
        self.log = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
        self.icon = icon_factory(self) if icon_factory else None

    # ---------------- helpers
    def write_log(self, text: str) -> None:
        self.log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text.rstrip() + "\n")

    def status_text(self) -> str:
        s = f"faceCheck: {LABELS.get(self.state, self.state)}"
        if self.state == "paused" and self.pause_until:
            s += f" (until {time.strftime('%H:%M', time.localtime(self.pause_until))})"
        elif self.detail and self.state in ("blank", "error"):
            s += f" ({self.detail})"
        return s

    def set_state(self, state: str, detail: str = "") -> None:
        if self.paused and state in ("ok", "blank") and not self.enrolling:
            state = "paused"
        self.state, self.detail = state, detail
        if self.icon is not None:
            try:
                self.icon.icon = make_icon(state)
                self.icon.title = self.status_text()
                self.icon.update_menu()
            except Exception:
                pass

    def notify(self, msg: str) -> None:
        self.write_log(msg)
        if self.icon is not None:
            try:
                self.icon.notify(msg, "faceCheck")
            except Exception:
                pass

    def guard_cmd(self) -> list[str]:
        c = self.cfg
        cmd = [sys.executable, "-u", str(GUARD), "--camera", str(c["camera"]), "run", "--ipc",
               "--mode", c["mode"], "--threshold", str(c["threshold"]),
               "--stranger-delay", str(c["stranger_delay"]), "--absent-delay", str(c["absent_delay"])]
        if c.get("pin"):
            cmd += ["--pin", str(c["pin"])]
        if c.get("preview"):
            cmd.append("--preview")
        return cmd

    @staticmethod
    def popen(cmd: list[str]) -> subprocess.Popen:
        kw = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                  text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=str(HERE),
                  env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        if SYSTEM == "Windows":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        return subprocess.Popen(cmd, **kw)

    def send(self, command: str) -> None:
        with self.proc_lock:
            p = self.proc
        if p and p.poll() is None and p.stdin:
            try:
                p.stdin.write(command + "\n")
                p.stdin.flush()
            except OSError:
                pass

    def handle_line(self, line: str) -> None:
        if not line.startswith("@"):
            if line.strip():
                self.write_log(line)
            return
        try:
            ev = json.loads(line[1:])
        except ValueError:
            return
        kind = ev.get("event")
        if kind == "ready":
            self.set_state("paused" if self.paused else "ok")
        elif kind in ("state", "hb"):
            st = ev.get("state", "ok")
            if kind == "hb" and st == self.state:
                return
            self.set_state(st, ev.get("reason", ""))
        elif kind == "error":
            self.write_log("guard error: " + ev.get("message", ""))

    def stop_child(self, timeout: float = 4.0) -> None:
        with self.proc_lock:
            p = self.proc
        if not p or p.poll() is not None:
            return
        self.restart_requested = True
        self.send("quit")
        try:
            p.wait(timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(2)
            except subprocess.TimeoutExpired:
                p.kill()

    def restart(self) -> None:
        self.cfg = load_config()
        self.set_state("starting")
        self.stop_child()

    # ---------------- background threads
    def supervisor(self) -> None:
        backoff = 2.0
        while not self.stopping:
            if self.enrolling:
                time.sleep(0.5)
                continue
            if not OWNER_FILE.exists():
                self.set_state("error", "no face enrolled - use 'Enroll my face'")
                time.sleep(1)
                continue
            started = time.time()
            self.restart_requested = False
            try:
                p = self.popen(self.guard_cmd())
            except OSError as e:
                self.set_state("error", str(e))
                time.sleep(5)
                continue
            with self.proc_lock:
                self.proc = p
            self.write_log("guard started: " + " ".join(self.guard_cmd()[2:]))
            if self.paused:
                self.send("pause")
            assert p.stdout is not None
            for line in p.stdout:
                self.handle_line(line)
            rc = p.wait()
            with self.proc_lock:
                self.proc = None
            self.write_log(f"guard exited with code {rc}")
            if self.stopping:
                break
            if self.restart_requested:
                backoff = 2.0
                continue
            self.set_state("error", f"guard stopped (code {rc}), restarting")
            backoff = 2.0 if time.time() - started > 60 else min(backoff * 2, 60.0)
            time.sleep(backoff)

    def ticker(self) -> None:
        while not self.stopping:
            if self.paused and self.pause_until and time.time() >= self.pause_until:
                self.resume()
            time.sleep(2)

    def start_threads(self) -> None:
        threading.Thread(target=self.supervisor, daemon=True).start()
        threading.Thread(target=self.ticker, daemon=True).start()

    # ---------------- menu actions
    def pause(self, minutes: float | None = None) -> None:
        self.paused = True
        self.pause_until = time.time() + minutes * 60 if minutes else 0.0
        self.send("pause")
        self.set_state("paused")

    def resume(self) -> None:
        self.paused = False
        self.pause_until = 0.0
        self.send("resume")
        self.set_state("ok")

    def toggle_pause(self) -> None:
        self.resume() if self.paused else self.pause()

    def set_mode(self, mode: str) -> None:
        self.cfg["mode"] = mode
        save_config(self.cfg)
        self.restart()

    def toggle_preview(self) -> None:
        self.cfg["preview"] = not self.cfg.get("preview", False)
        save_config(self.cfg)
        self.restart()

    def toggle_autostart(self) -> None:
        if Autostart.is_enabled():
            Autostart.disable()
            self.notify("Start at login disabled")
        else:
            Autostart.enable()
            self.notify("faceCheck will start when you log in")
        if self.icon is not None:
            self.icon.update_menu()

    def enroll(self, append: bool = False) -> None:
        if self.enrolling:
            return
        self.enrolling = True
        self.set_state("enroll")
        self.stop_child()  # the camera can only be used by one process

        def work():
            try:
                cmd = [sys.executable, "-u", str(GUARD), "--camera", str(self.cfg["camera"]), "enroll"]
                if append:
                    cmd.append("--append")
                p = self.popen(cmd)
                assert p.stdout is not None
                for line in p.stdout:
                    self.write_log("[enroll] " + line)
                ok = p.wait() == 0
                self.notify("Face enrolled" if ok else "Enrollment failed - see the log")
            finally:
                self.enrolling = False
                self.set_state("starting")

        threading.Thread(target=work, daemon=True).start()

    def quit(self) -> None:
        self.stopping = True
        self.stop_child()
        if self.icon is not None:
            self.icon.stop()


# --------------------------------------------------------------------------- pystray glue
def build_icon(ctl: Controller):
    import pystray
    from pystray import Menu, MenuItem as Item

    def mode_item(mode: str, label: str):
        return Item(label, lambda: ctl.set_mode(mode), checked=lambda _: ctl.cfg["mode"] == mode, radio=True)

    menu = Menu(
        Item(lambda _: ctl.status_text(), None, enabled=False),
        Menu.SEPARATOR,
        Item(lambda _: "Resume" if ctl.paused else "Pause", lambda: ctl.toggle_pause(), default=True),
        Item("Pause for 15 minutes", lambda: ctl.pause(15)),
        Item("Pause for 1 hour", lambda: ctl.pause(60)),
        Menu.SEPARATOR,
        Item("When a stranger looks", Menu(
            mode_item("overlay", "Black overlay (auto-restore)"),
            mode_item("lock", "Lock the computer"),
            mode_item("off", "Turn the display off"),
        )),
        Item("Camera preview", lambda: ctl.toggle_preview(), checked=lambda _: bool(ctl.cfg.get("preview"))),
        Item("Enroll my face", lambda: ctl.enroll()),
        Item("Add more face samples", lambda: ctl.enroll(append=True)),
        Menu.SEPARATOR,
        Item("Start at login", lambda: ctl.toggle_autostart(), checked=lambda _: Autostart.is_enabled()),
        Item("Edit settings…", lambda: (save_config(ctl.cfg), open_path(CONFIG_FILE))),
        Item("Reload settings", lambda: ctl.restart()),
        Item("Open log", lambda: open_path(LOG_FILE)),
        Menu.SEPARATOR,
        Item("Quit faceCheck", lambda: ctl.quit()),
    )
    return pystray.Icon("faceCheck", make_icon("starting"), "faceCheck", menu)


def single_instance_lock():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", LOCK_PORT))
        s.listen(1)
        return s
    except OSError:
        s.close()
        return None


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--enable-autostart" in argv:
        Autostart.enable()
        print("Autostart enabled:", " ".join(Autostart.command()))
        return
    if "--disable-autostart" in argv:
        Autostart.disable()
        print("Autostart disabled")
        return

    try:
        import pystray  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        sys.exit("Missing dependencies:  pip install -r requirements.txt")

    lock = single_instance_lock()
    if lock is None:
        print("faceCheck is already running.")
        return

    ctl = Controller(icon_factory=build_icon)
    ctl.write_log("tray started")

    if ctl.cfg.get("first_run", True):
        ctl.cfg["first_run"] = False
        save_config(ctl.cfg)
        try:
            Autostart.enable()
            ctl.write_log("autostart enabled on first run")
        except Exception as e:  # pragma: no cover
            ctl.write_log(f"could not enable autostart: {e}")

    def setup(icon):
        icon.visible = True
        if not OWNER_FILE.exists():
            ctl.notify("Welcome! Look at the camera to enroll your face.")
            ctl.enroll()
        ctl.start_threads()

    try:
        ctl.icon.run(setup=setup)  # must run on the main thread (macOS)
    finally:
        ctl.stopping = True
        ctl.stop_child()
        lock.close()


if __name__ == "__main__":
    main()

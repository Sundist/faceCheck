#!/usr/bin/env python3
"""
faceCheck / ScreenGuard - cross-platform (Windows / macOS / Linux) owner-aware screen guard.

Watches the webcam:
  * Owner in frame            -> screen stays ON (even if other people are around).
  * Only stranger(s) looking  -> screen is blanked after a short delay.
  * Nobody in front (optional) -> screen is blanked after a configurable delay.
  * Owner comes back          -> screen is restored automatically (overlay / off modes).

Usage:
  python screenguard.py enroll            # register the owner's face
  python screenguard.py run               # start guarding
  python screenguard.py run --preview     # with a camera preview window (for tuning)
  python screenguard.py run --mode lock   # use the OS lock screen instead of an overlay

For background use with a system-tray icon and autostart, run tray.py instead.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("OpenCV is required:  pip install opencv-python numpy")

APP_DIR = Path(os.environ.get("SCREENGUARD_HOME", Path.home() / ".screenguard"))
OWNER_FILE = APP_DIR / "owner.npz"

MODELS = {
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
    ),
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx",
    ),
}

SYSTEM = platform.system()  # "Windows" | "Darwin" | "Linux"


# --------------------------------------------------------------------------- models
def ensure_models() -> dict[str, Path]:
    model_dir = APP_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, url) in MODELS.items():
        p = model_dir / name
        if not p.exists() or p.stat().st_size < 10_000:
            print(f"[model] downloading {name}", flush=True)
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(p)
        paths[key] = p
    return paths


# --------------------------------------------------------------------------- faces
@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h
    row: np.ndarray                 # raw YuNet row (15 values)
    frontal: bool                   # roughly looking at the screen
    size_ratio: float               # face width / frame width
    score: float = 0.0              # similarity to owner (cosine)
    is_owner: bool = False


class FaceEngine:
    def __init__(self, det_path: Path, rec_path: Path, det_threshold: float = 0.8):
        self.det = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), det_threshold, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(rec_path), "")

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, rows = self.det.detect(frame)
        faces = []
        if rows is None:
            return faces
        for r in rows:
            x, y, fw, fh = (int(v) for v in r[:4])
            faces.append(Face((x, y, fw, fh), r, is_frontal(r), fw / float(w)))
        return faces

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        aligned = self.rec.alignCrop(frame, face.row)
        feat = self.rec.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)


def is_frontal(row: np.ndarray, max_yaw: float = 0.35, max_pitch: float = 0.9) -> bool:
    """Rough gaze check from YuNet landmarks: nose close to the eye midpoint => facing the screen."""
    rex, rey, lex, ley, nx, ny = row[4:10]
    eye_dist = abs(lex - rex)
    if eye_dist < 1:
        return False
    mid_x, mid_y = (rex + lex) / 2, (rey + ley) / 2
    yaw = abs(nx - mid_x) / eye_dist
    pitch = (ny - mid_y) / eye_dist  # ~0.5-0.7 when frontal
    return yaw < max_yaw and 0.15 < pitch < max_pitch


def load_owner(path: Path) -> np.ndarray:
    if not path.exists():
        sys.exit(f"Owner profile not found: {path}\nRun first:  python screenguard.py enroll")
    return np.load(path)["features"]


def owner_score(feat: np.ndarray, owner_feats: np.ndarray) -> float:
    return float(np.max(owner_feats @ feat))


# --------------------------------------------------------------------------- camera
def open_camera(index: int, width: int = 640, height: int = 480):
    backend = cv2.CAP_DSHOW if SYSTEM == "Windows" else (cv2.CAP_AVFOUNDATION if SYSTEM == "Darwin" else cv2.CAP_ANY)
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera (index={index}). Try another one with --camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# --------------------------------------------------------------------------- enroll
def cmd_enroll(args) -> None:
    paths = ensure_models()
    eng = FaceEngine(paths["detector"], paths["recognizer"])
    feats: list[np.ndarray] = []

    if args.images:
        for img_path in sorted(Path(args.images).iterdir()):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = eng.detect(img)
            if len(faces) != 1:
                print(f"  skipped ({len(faces)} faces): {img_path.name}")
                continue
            feats.append(eng.embed(img, faces[0]))
            print(f"  added: {img_path.name}")
    else:
        cap = open_camera(args.camera)
        prompts = ["Look straight", "Turn slightly left", "Turn slightly right", "Look slightly up",
                   "Look slightly down", "Smile", "Look straight"]
        print("Look at the camera. Samples are taken automatically when exactly one face is visible. Quit: q")
        last = 0.0
        while len(feats) < args.samples:
            ok, frame = cap.read()
            if not ok:
                continue
            faces = eng.detect(frame)
            prompt = prompts[min(len(feats) * len(prompts) // args.samples, len(prompts) - 1)]
            status = f"{len(feats)}/{args.samples}  {prompt}"
            if len(faces) == 1 and time.time() - last > 0.35:
                f = eng.embed(frame, faces[0])
                if not feats or max(float(x @ f) for x in feats) < 0.97:  # skip near-duplicates
                    feats.append(f)
                    last = time.time()
            elif len(faces) > 1:
                status = "Only you should be in the frame!"
            for fc in faces:
                x, y, w, h = fc.box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            try:
                cv2.imshow("faceCheck - enroll", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass  # headless OpenCV build: no window
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if len(feats) < 3:
        sys.exit("Not enough samples captured (at least 3 required).")
    feats_arr = np.stack(feats)
    out = Path(args.owner)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.append and out.exists():
        feats_arr = np.concatenate([np.load(out)["features"], feats_arr])
    np.savez(out, features=feats_arr)
    print(f"Saved: {out}  ({len(feats_arr)} samples)")


# --------------------------------------------------------------------------- screen actions
def _run(cmd: list[str]) -> bool:
    if not shutil.which(cmd[0]):
        return False
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except Exception:
        return False


def os_lock() -> bool:
    if SYSTEM == "Windows":
        import ctypes
        return bool(ctypes.windll.user32.LockWorkStation())
    if SYSTEM == "Darwin":
        return _run(["pmset", "displaysleepnow"])  # locks if "require password immediately" is on
    for cmd in (["loginctl", "lock-session"], ["xdg-screensaver", "lock"],
                ["gnome-screensaver-command", "-l"], ["dm-tool", "lock"]):
        if _run(cmd):
            return True
    return False


def display_off() -> bool:
    if SYSTEM == "Windows":
        import ctypes
        ctypes.windll.user32.PostMessageW(0xFFFF, 0x0112, 0xF170, 2)  # SC_MONITORPOWER -> off
        return True
    if SYSTEM == "Darwin":
        return _run(["pmset", "displaysleepnow"])
    return _run(["xset", "dpms", "force", "off"])


def display_on() -> bool:
    if SYSTEM == "Windows":
        import ctypes
        ctypes.windll.user32.PostMessageW(0xFFFF, 0x0112, 0xF170, -1)
        ctypes.windll.user32.mouse_event(0x0001, 0, 1, 0, 0)  # tiny mouse move wakes the display
        ctypes.windll.user32.mouse_event(0x0001, 0, -1, 0, 0)
        return True
    if SYSTEM == "Darwin":
        return _run(["caffeinate", "-u", "-t", "1"])
    return _run(["xset", "dpms", "force", "on"])


class Overlay:
    """Full-screen, always-on-top black curtain (tkinter ships with Python on every OS)."""

    def __init__(self, pin: str | None = None):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("faceCheck")
        self.root.configure(bg="black", cursor="none")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.pin = pin
        self.typed = ""
        self.unlocked_until = 0.0
        self.visible = False
        self.label = tk.Label(self.root, text="", fg="#333333", bg="black", font=("Helvetica", 14))
        self.label.place(relx=0.5, rely=0.97, anchor="s")
        self.root.bind("<Key>", self._on_key)
        x, y, w, h = self._virtual_screen()
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _virtual_screen(self):
        if SYSTEM == "Windows":
            import ctypes
            m = ctypes.windll.user32.GetSystemMetrics
            w, h = m(78), m(79)
            if w and h:
                return m(76), m(77), w, h  # covers all monitors
        return 0, 0, self.root.winfo_screenwidth(), self.root.winfo_screenheight()

    def _on_key(self, ev):
        if not self.pin:
            return
        if ev.keysym == "Return":
            if self.typed == self.pin:
                self.unlocked_until = time.time() + 60
                self.hide()
            self.typed = ""
        elif ev.keysym == "BackSpace":
            self.typed = self.typed[:-1]
        elif ev.char and ev.char.isprintable():
            self.typed = (self.typed + ev.char)[-64:]
        self.label.config(text="•" * len(self.typed))

    def show(self):
        if time.time() < self.unlocked_until or self.visible:
            return
        self.visible = True
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()

    def hide(self):
        if not self.visible:
            return
        self.visible = False
        self.root.withdraw()

    def pump(self):
        if self.visible:
            self.root.lift()
        self.root.update()


# --------------------------------------------------------------------------- decision logic
class Guard:
    """Pure decision logic (camera independent, unit-testable)."""

    def __init__(self, stranger_delay=1.5, absent_delay=0.0, owner_grace=2.0, owner_confirm=2):
        self.stranger_delay = stranger_delay
        self.absent_delay = absent_delay
        self.owner_grace = owner_grace
        self.owner_confirm = owner_confirm
        self.last_owner = time.time()     # assume the owner is there (they just started it)
        self.owner_streak = 0
        self.stranger_since: float | None = None
        self.empty_since: float | None = None
        self.blanked = False
        self.reason = ""

    def update(self, now: float, owner_seen: bool, stranger_looking: bool, any_face: bool) -> bool:
        """Returns True if the screen should be blanked."""
        self.owner_streak = self.owner_streak + 1 if owner_seen else 0
        if owner_seen and (not self.blanked or self.owner_streak >= self.owner_confirm):
            self.last_owner = now
        owner_present = (now - self.last_owner) <= self.owner_grace

        self.stranger_since = (self.stranger_since or now) if stranger_looking else None
        self.empty_since = (self.empty_since or now) if not any_face else None

        if owner_present:
            self.blanked, self.reason = False, "owner"
        elif self.stranger_since is not None and now - self.stranger_since >= self.stranger_delay:
            self.blanked, self.reason = True, "stranger looking"
        elif (self.absent_delay > 0 and self.empty_since is not None
              and now - max(self.empty_since, self.last_owner) >= self.absent_delay):
            self.blanked, self.reason = True, "nobody there"
        return self.blanked


# --------------------------------------------------------------------------- IPC (used by tray.py)
class IPC:
    """Line protocol with the tray supervisor.

    stdout: lines starting with '@' carry JSON events, e.g. @{"event": "state", "state": "blank"}
    stdin : commands, one per line: pause | resume | quit
    If stdin closes (supervisor died) the guard exits, so it can never be left orphaned.
    """

    def __init__(self, instream=None, outstream=None):
        self.instream = instream or sys.stdin
        self.out = outstream or sys.stdout
        self.paused = False
        self.quit = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.instream:
            cmd = line.strip().lower()
            if cmd == "pause":
                self.paused = True
            elif cmd == "resume":
                self.paused = False
            elif cmd == "quit":
                break
        self.quit = True

    def emit(self, event: str, **kw):
        kw["event"] = event
        try:
            self.out.write("@" + json.dumps(kw, ensure_ascii=False) + "\n")
            self.out.flush()
        except Exception:
            self.quit = True


# --------------------------------------------------------------------------- run
def cmd_run(args, ipc: IPC | None = None) -> None:
    paths = ensure_models()
    eng = FaceEngine(paths["detector"], paths["recognizer"])
    owner = load_owner(Path(args.owner))
    new_guard = lambda: Guard(args.stranger_delay, args.absent_delay, args.owner_grace)  # noqa: E731
    guard = new_guard()
    if ipc is None and getattr(args, "ipc", False):
        ipc = IPC()

    overlay = None
    if args.mode == "overlay":
        try:
            overlay = Overlay(pin=args.pin)
        except Exception as e:
            sys.exit(f"Could not create overlay ({e}). Try --mode lock or --mode off.")

    cap = None
    period = 1.0 / args.fps
    blanked = False
    paused = False
    fail_since = None
    last_hb = 0.0
    print(f"faceCheck running  [mode={args.mode}, threshold={args.threshold}]  Stop with Ctrl+C", flush=True)
    if ipc:
        ipc.emit("ready", mode=args.mode)

    def set_blank(value: bool, reason: str):
        nonlocal blanked
        if value == blanked:
            return
        blanked = value
        print(time.strftime("%H:%M:%S"), "BLANK" if blanked else "SHOW", f"({reason})", flush=True)
        if args.mode == "overlay":
            overlay.show() if blanked else overlay.hide()
        elif args.mode == "lock":
            if blanked and not os_lock():
                print("[warn] OS lock failed", flush=True)
        elif args.mode == "off":
            display_off() if blanked else display_on()
        if ipc:
            ipc.emit("state", state="blank" if blanked else "ok", reason=reason)

    try:
        while True:
            t0 = time.time()

            # ---- commands from the tray
            if ipc and ipc.quit:
                break
            if ipc and ipc.paused != paused:
                paused = ipc.paused
                if paused:
                    set_blank(False, "paused")
                    if cap is not None:
                        cap.release()  # turn the camera (and its LED) off while paused
                        cap = None
                    ipc.emit("state", state="paused", reason="paused")
                else:
                    guard = new_guard()
                    ipc.emit("state", state="ok", reason="resumed")
            if paused:
                if overlay:
                    overlay.pump()
                time.sleep(0.2)
                continue

            if cap is None:
                cap = open_camera(args.camera)

            ok, frame = cap.read()
            if not ok:
                fail_since = fail_since or t0
                if t0 - fail_since > 3 and blanked:  # fail-open: never trap the owner
                    set_blank(False, "camera unavailable")
                    if ipc:
                        ipc.emit("error", message="camera unavailable")
                if overlay:
                    overlay.pump()
                time.sleep(0.1)
                continue
            fail_since = None

            faces = eng.detect(frame)
            owner_seen = stranger_looking = False
            for f in faces:
                f.score = owner_score(eng.embed(frame, f), owner)
                f.is_owner = f.score >= args.threshold
                if f.is_owner:
                    owner_seen = True
                elif f.frontal and f.size_ratio >= args.min_face:
                    stranger_looking = True

            want_blank = guard.update(t0, owner_seen, stranger_looking, bool(faces))
            set_blank(want_blank, guard.reason)

            if args.mode == "off" and blanked and int(t0) % 5 == 0:
                display_off()  # keep it off if something woke it up

            if overlay:
                if blanked and not overlay.visible and time.time() >= overlay.unlocked_until:
                    overlay.show()
                overlay.pump()

            if ipc and t0 - last_hb > 5:
                last_hb = t0
                ipc.emit("hb", state="blank" if blanked else "ok", faces=len(faces), owner=owner_seen)

            if args.preview:
                for f in faces:
                    x, y, w, h = f.box
                    color = (0, 200, 0) if f.is_owner else ((0, 0, 255) if f.frontal else (0, 165, 255))
                    label = f"{'OWNER' if f.is_owner else 'stranger'} {f.score:.2f}{'' if f.frontal else ' (not looking)'}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, "BLANK" if blanked else "OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255) if blanked else (0, 200, 0), 2)
                cv2.imshow("faceCheck - preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        if overlay:
            overlay.hide()
            overlay.root.destroy()
        if args.mode == "off" and blanked:
            display_on()
        if args.preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# --------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="faceCheck: blanks the screen when a stranger looks at it without you")
    p.add_argument("--owner", default=str(OWNER_FILE), help="owner face profile (.npz)")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="register the owner's face")
    e.add_argument("--samples", type=int, default=20)
    e.add_argument("--images", help="enroll from photos in this folder instead of the camera")
    e.add_argument("--append", action="store_true", help="add to the existing profile (other lighting, glasses...)")
    e.set_defaults(func=cmd_enroll)

    r = sub.add_parser("run", help="start guarding")
    r.add_argument("--mode", choices=["overlay", "lock", "off"], default="overlay",
                   help="overlay: black curtain (auto-restores) | lock: OS lock screen | off: turn the display off")
    r.add_argument("--threshold", type=float, default=0.40, help="owner match threshold (SFace cosine, ~0.36-0.5)")
    r.add_argument("--stranger-delay", type=float, default=1.5, help="seconds a lone stranger may look before blanking")
    r.add_argument("--absent-delay", type=float, default=0.0, help="blank after N seconds with nobody there (0 = off)")
    r.add_argument("--owner-grace", type=float, default=2.0, help="seconds the owner still counts as present after leaving the frame")
    r.add_argument("--min-face", type=float, default=0.08, help="ignore distant faces (face width / frame width)")
    r.add_argument("--fps", type=float, default=8.0)
    r.add_argument("--pin", help="overlay mode: type this PIN + Enter to lift the curtain for 60 s")
    r.add_argument("--preview", action="store_true", help="show a camera preview window")
    r.add_argument("--ipc", action="store_true", help="machine-readable status on stdout, commands on stdin (used by tray.py)")
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

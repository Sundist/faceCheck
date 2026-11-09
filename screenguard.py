#!/usr/bin/env python3
"""
faceCheck / ScreenGuard - cross-platform (Windows / macOS / Linux) owner-aware screen guard.

Webcam'den yüzleri izler:
  * Sahip (owner) kadrajdaysa ekran AÇIK kalır (yanında başkaları olsa bile).
  * Ekrana yalnızca yabancı(lar) bakıyorsa ekran kısa bir gecikmeyle KAPATILIR.
  * (İsteğe bağlı) Kimse yoksa belirli bir süre sonra ekran kapatılır.
  * Sahip geri gelince ekran otomatik açılır (overlay / off modlarında).

Kullanım:
  python screenguard.py enroll            # sahibin yüzünü kaydet
  python screenguard.py run               # korumayı başlat
  python screenguard.py run --preview     # kamera önizlemesiyle (ayar/test için)
  python screenguard.py run --mode lock   # yabancıda OS kilit ekranı
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("OpenCV gerekli:  pip install opencv-python numpy")

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
            print(f"[model] indiriliyor: {name}")
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
    """YuNet landmark'larıyla kaba bakış yönü: burun iki gözün ortasına yakınsa ekrana bakıyor."""
    rex, rey, lex, ley, nx, ny = row[4:10]
    eye_dist = abs(lex - rex)
    if eye_dist < 1:
        return False
    mid_x, mid_y = (rex + lex) / 2, (rey + ley) / 2
    yaw = abs(nx - mid_x) / eye_dist
    pitch = (ny - mid_y) / eye_dist  # ~0.5-0.7 when frontal; small/negative when looking up/down a lot
    return yaw < max_yaw and 0.15 < pitch < max_pitch


def load_owner(path: Path) -> np.ndarray:
    if not path.exists():
        sys.exit(f"Sahip kaydı bulunamadı: {path}\nÖnce çalıştırın:  python screenguard.py enroll")
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
        sys.exit(f"Kamera açılamadı (index={index}). --camera ile başka bir index deneyin.")
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
                print(f"  atlandı ({len(faces)} yüz): {img_path.name}")
                continue
            feats.append(eng.embed(img, faces[0]))
            print(f"  eklendi: {img_path.name}")
    else:
        cap = open_camera(args.camera)
        prompts = ["Düz bakın", "Hafif sola dönün", "Hafif sağa dönün", "Hafif yukarı", "Hafif aşağı",
                   "Gülümseyin", "Düz bakın"]
        print("Kameraya bakın. Tek bir yüz göründüğünde örnekler otomatik alınır. Çıkış: q")
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
                status = "Kadrajda yalnızca siz olun!"
            for fc in faces:
                x, y, w, h = fc.box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            try:
                cv2.imshow("faceCheck - kayit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass  # headless OpenCV: no window
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if len(feats) < 3:
        sys.exit("Yeterli örnek alınamadı (en az 3 gerekli).")
    feats_arr = np.stack(feats)
    out = Path(args.owner)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.append and out.exists():
        feats_arr = np.concatenate([np.load(out)["features"], feats_arr])
    np.savez(out, features=feats_arr)
    print(f"Kaydedildi: {out}  ({len(feats_arr)} örnek)")


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
    """Tam ekran, en üstte duran siyah perde (tkinter: her OS'ta var)."""

    def __init__(self, pin: str | None = None):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.withdraw()
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
        try:
            self.root.grab_release()
        except Exception:
            pass
        self.root.withdraw()

    def pump(self):
        if self.visible:
            self.root.lift()
        self.root.update()


# --------------------------------------------------------------------------- decision logic
class Guard:
    """Saf karar mantığı (kameradan bağımsız, test edilebilir)."""

    def __init__(self, stranger_delay=1.5, absent_delay=0.0, owner_grace=2.0, owner_confirm=2):
        self.stranger_delay = stranger_delay
        self.absent_delay = absent_delay
        self.owner_grace = owner_grace
        self.owner_confirm = owner_confirm
        self.last_owner = time.time()     # start assuming owner is there (they just launched it)
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
            self.blanked, self.reason = False, "sahip"
        elif self.stranger_since is not None and now - self.stranger_since >= self.stranger_delay:
            self.blanked, self.reason = True, "yabancı bakıyor"
        elif (self.absent_delay > 0 and self.empty_since is not None
              and now - max(self.empty_since, self.last_owner) >= self.absent_delay):
            self.blanked, self.reason = True, "kimse yok"
        return self.blanked


# --------------------------------------------------------------------------- run
def cmd_run(args) -> None:
    paths = ensure_models()
    eng = FaceEngine(paths["detector"], paths["recognizer"])
    owner = load_owner(Path(args.owner))
    guard = Guard(args.stranger_delay, args.absent_delay, args.owner_grace)

    overlay = None
    if args.mode == "overlay":
        try:
            overlay = Overlay(pin=args.pin)
        except Exception as e:
            sys.exit(f"Overlay açılamadı ({e}). --mode lock veya --mode off deneyin.")

    cap = open_camera(args.camera)
    period = 1.0 / args.fps
    blanked = False
    fail_since = None
    print(f"faceCheck çalışıyor  [mod={args.mode}, eşik={args.threshold}]  Durdurmak için Ctrl+C")

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                fail_since = fail_since or t0
                if t0 - fail_since > 3 and blanked and overlay:  # fail-open: never trap the owner
                    overlay.hide()
                    blanked = False
                    print("[uyarı] kamera okunamıyor, perde kaldırıldı")
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

            if want_blank != blanked:
                blanked = want_blank
                print(time.strftime("%H:%M:%S"), "KAPAT" if blanked else "AÇ", f"({guard.reason})")
                if args.mode == "overlay":
                    overlay.show() if blanked else overlay.hide()
                elif args.mode == "lock":
                    if blanked and not os_lock():
                        print("[uyarı] OS kilidi başarısız")
                elif args.mode == "off":
                    display_off() if blanked else display_on()

            if args.mode == "off" and blanked and int(t0) % 5 == 0:
                display_off()  # keep it off if something woke it

            if overlay:
                if blanked and not overlay.visible and time.time() >= overlay.unlocked_until:
                    overlay.show()
                overlay.pump()

            if args.preview:
                for f in faces:
                    x, y, w, h = f.box
                    color = (0, 200, 0) if f.is_owner else ((0, 0, 255) if f.frontal else (0, 165, 255))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{'SAHIP' if f.is_owner else 'yabanci'} {f.score:.2f}{'' if f.frontal else ' (bakmiyor)'}"
                    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, "BLANK" if blanked else "OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255) if blanked else (0, 200, 0), 2)
                cv2.imshow("faceCheck - onizleme", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if overlay:
            overlay.hide()
            overlay.root.destroy()
        if args.mode == "off":
            display_on()
        if args.preview:
            cv2.destroyAllWindows()


# --------------------------------------------------------------------------- cli
def main(argv=None):
    p = argparse.ArgumentParser(description="faceCheck: sahip yokken yabancıya ekranı kapatan koruma")
    p.add_argument("--owner", default=str(OWNER_FILE), help="sahip yüz kaydı (.npz)")
    p.add_argument("--camera", type=int, default=0, help="kamera index (varsayılan 0)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="sahibin yüzünü kaydet")
    e.add_argument("--samples", type=int, default=20)
    e.add_argument("--images", help="kamera yerine bu klasördeki fotoğraflardan kaydet")
    e.add_argument("--append", action="store_true", help="mevcut kayda ekle (farklı ışık/gözlük vb.)")
    e.set_defaults(func=cmd_enroll)

    r = sub.add_parser("run", help="korumayı başlat")
    r.add_argument("--mode", choices=["overlay", "lock", "off"], default="overlay",
                   help="overlay: siyah perde (otomatik geri açılır) | lock: OS kilidi | off: monitörü kapat")
    r.add_argument("--threshold", type=float, default=0.40, help="sahip eşleşme eşiği (SFace kosinüs, ~0.36-0.5)")
    r.add_argument("--stranger-delay", type=float, default=1.5, help="yabancı tek başına bakarsa kaç sn sonra kapansın")
    r.add_argument("--absent-delay", type=float, default=0.0, help="kimse yoksa kaç sn sonra kapansın (0=kapalı)")
    r.add_argument("--owner-grace", type=float, default=2.0, help="sahip kadrajdan çıkınca kaç sn 'var' sayılsın")
    r.add_argument("--min-face", type=float, default=0.08, help="uzaktaki yüzleri yok say (yüz genişliği/kare oranı)")
    r.add_argument("--fps", type=float, default=8.0)
    r.add_argument("--pin", help="overlay modunda perdeyi 60 sn kaldırmak için yazılacak PIN (+Enter)")
    r.add_argument("--preview", action="store_true", help="kamera önizleme penceresi")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

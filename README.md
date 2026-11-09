# faceCheck

An owner-aware screen guard for laptops. faceCheck watches the webcam and **blanks the screen when a stranger looks at it while you are not there.** It runs on Windows, macOS and Linux from the same code. It sits in the system tray and starts automatically when you log in.

| Situation | Screen |
|---|---|
| You (the owner) are in front of it, even with other people around | **On** |
| Only stranger(s) are looking at it | **Blanked after ~1.5 s** |
| A stranger is around but not looking, or is far away | On |
| Nobody is there | On (optional: `--absent-delay 30`) |
| You come back | Restored automatically (`overlay` / `off` modes) |

Face detection uses OpenCV **YuNet** and face recognition uses **SFace** (ONNX models). No dlib and no GPU needed. The models are downloaded on first run to `~/.screenguard/models`. Everything runs locally; no image ever leaves your computer.

## Install

```bash
git clone https://github.com/Sundist/faceCheck.git
cd faceCheck
pip install -r requirements.txt
```

Linux extras: `sudo apt install python3-tk`. For a tray icon on GNOME you also need the *AppIndicator* extension (`sudo apt install gir1.2-ayatanaappindicator3-0.1 python3-gi`, then `pip install PyGObject`).

## Quick start (tray app, recommended)

```bash
python tray.py
```

On first launch faceCheck:

1. Registers itself to **start at login** (Windows Run key, macOS LaunchAgent, Linux `~/.config/autostart`).
2. Opens the camera to **enroll your face**. Look at the camera and follow the prompts; 20 samples are taken.
3. Shows an icon in the tray (Windows taskbar, macOS menu bar, Linux panel) and starts guarding.

From then on it starts by itself every time you log in and stops when you shut down.

### Tray icon

| Color | Meaning |
|---|---|
| 🟢 green | Protected (you are recognized or nobody suspicious) |
| 🔴 red | Screen blanked: a stranger is looking |
| 🟡 yellow | Paused (camera off) |
| 🔵 blue | Enrolling your face |
| ⚪ grey | Starting or error (see *Open log*) |

The menu (right-click; on Windows a left-click toggles pause) offers:

- **Pause / Resume**, **Pause for 15 minutes**, **Pause for 1 hour**. The camera is released while paused.
- **When a stranger looks**: *Black overlay*, *Lock the computer*, or *Turn the display off*.
- **Camera preview**: shows boxes and match scores, useful for tuning.
- **Enroll my face** / **Add more face samples** (e.g. with glasses or in different lighting).
- **Start at login** (checkbox).
- **Edit settings…**, **Reload settings**, **Open log**, **Quit faceCheck**.

The tray app supervises the guard in a separate process and restarts it if it crashes. If the tray app is killed, the guard exits too and the screen is never left blanked.

### Autostart from the command line

```bash
python tray.py --enable-autostart
python tray.py --disable-autostart
```

The autostart entry stores the absolute path of the current Python interpreter and of `tray.py`. If you move the folder or change your Python/virtualenv, run `--enable-autostart` again. On Windows it uses `pythonw.exe` so no console window appears.

## Settings

`~/.screenguard/config.json` (menu: *Edit settings…*, then *Reload settings*):

| Key | Default | Description |
|---|---|---|
| `mode` | `overlay` | `overlay` = black curtain, auto-restores · `lock` = OS lock screen (most secure) · `off` = display power off |
| `threshold` | `0.40` | Owner match threshold. Lower it (0.36) if you are not recognized; raise it (0.45–0.5) if strangers are accepted. |
| `stranger_delay` | `1.5` | Seconds a lone stranger may look before blanking |
| `absent_delay` | `0` | Blank after N seconds with nobody there (0 = disabled) |
| `camera` | `0` | Camera index |
| `pin` | `""` | Overlay mode: type this PIN + Enter to lift the curtain for 60 s |
| `preview` | `false` | Show the camera preview window |

Files in `~/.screenguard/`: `owner.npz` (your face embeddings, no images), `config.json`, `facecheck.log`, `models/`.

## Command line (without the tray)

```bash
python screenguard.py enroll                 # register your face
python screenguard.py enroll --append        # add more samples
python screenguard.py run --preview          # guard with a preview window
python screenguard.py run --mode lock        # lock the OS instead of an overlay
python screenguard.py run --help             # all options
```

Preview colors: green = owner, red = stranger looking at the screen, orange = stranger not looking.

## How it works

1. Every frame (~8 fps), YuNet finds faces and their landmarks.
2. Each face is embedded with SFace and compared (cosine similarity) with your enrolled samples.
3. A face counts as *looking at the screen* when its nose is near the midpoint between the eyes and the face is large enough (not in the background).
4. A small state machine decides:
   - owner seen within the last `owner_grace` seconds → show,
   - a non-owner looking for `stranger_delay` seconds → blank,
   - nobody for `absent_delay` seconds → blank (if enabled).
   Restoring requires two consecutive owner matches to avoid flicker.

## Limitations

- No liveness check: a printed photo or a video of you could fool it. Use `mode: lock` if you need real security.
- `overlay` is a visual curtain, not an input lock; the keyboard still works behind it.
- `off` mode on Linux needs X11 (`xset`); on Wayland use `overlay` or `lock`.
- On macOS, grant camera access to the Python/Terminal app on first run. The `lock` / `off` modes use `pmset displaysleepnow`; enable *Require password immediately* for a real lock.
- Detection gets worse in the dark.

## Development

```bash
pip install pytest
python -m pytest -q tests
```

The tests cover the decision logic, the guard's IPC protocol (with a fake camera), and the tray supervisor (launch, pause, crash restart, autostart).

## Project layout

```
screenguard.py   face detection/recognition, decision logic, screen actions, CLI
tray.py          tray icon + menu, guard supervisor, start-at-login integration
tests/           pytest suite
```

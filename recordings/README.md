# Recordings

Raw hand-tracking recordings by the author, so every number in the main README can be reproduced
with `tools/replay.py`. They contain MediaPipe landmarks only: no images or video.

| File | Recorded | Role |
|---|---|---|
| `session_20260924_192624.csv` | 2026-09-24 | development session (three phases, before the jumps phase existed) |
| `session_20260924_214146.csv` | 2026-09-24 | tuning session: the final configurations were chosen on it |
| `session_20260927_190639.csv` | 2026-09-27 | test session: recorded after the configurations were fixed in commit `8528057` |
| `depth_probe_20260924_185624.csv` | 2026-09-24 | depth-proxy comparison (`tools/depth_probe.py`); phases were labeled by time because the key presses did not register |

## Session columns (`tools/record_session.py`)

One row per camera frame with a detected hand, about 30 per second.

- `t_ms`: capture time, monotonic clock in ms.
- `rep`, `phase`: repetition (1 to 3) and phase (1 still, 2 circle, 3 depth, 4 jumps).
- `rec`: 1 inside the measured 10 s window, 0 during the preparation before it.
- `t_phase_ms`: time since the measured window started (negative during preparation).
- `gu`, `gv`, `gz`: on-screen guide position (normalized image coordinates of the mirrored image)
  and guide distance in cm.
- `lat_ms`: hand-tracking inference latency of that frame.
- `w`, `h`: camera frame size in pixels.
- `x0` ... `z20`: the 21 normalized image landmarks.
- `wx0` ... `wz20`: the 21 world landmarks in meters.

The depth probe has the same landmark columns after `t_ms`, `phase`, `w` and `h`.

## Reproduce the results

```bash
python tools/replay.py recordings/session_20260927_190639.csv --final   # test table
python tools/replay.py recordings/session_20260924_214146.csv --final   # tuning table
```

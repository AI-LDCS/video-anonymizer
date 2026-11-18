#!/usr/bin/env python3
"""Pick the sharpest video frame in every N metres walked (default 3 m).

Input : a video and its Field Capture gps.json (points with videoTimeS, latitude, longitude, accuracyM).
Output: lossless PNG frames, manifest.csv and frames.json. This tool does NOT anonymize: the frames are raw,
        so run anonymize.py on the output folder (or use pipeline.py / worker.py) before sharing or CVAT.

How it works
  1. Drop inaccurate GPS points, project lat/lng to metres, smooth, then add up the distance walked.
     Tiny movements (standing still, GPS jitter) count as zero distance.
  2. Cut the path into bins of --step metres and map each bin to a time range in the video.
  3. Decode the video once, score every frame (variance of the Laplacian, higher = sharper) and keep
     the best frame of each bin. A bin where everything is blurry still gets its least-blurry frame.

Several recordings of one assignment are one continuous walk (the worker is told to carry on from where
the last recording stopped): pass --state-out of recording N as --state-in of recording N+1. The distance
counter and the frame numbers then carry on, so the spacing is the same across the join. If the next
recording starts farther than --gap-max from where the last one stopped, something is missing (an unrecorded
stretch, or a GPS error): the count still carries on, but the jump is not added to the distance and the first
frame is marked afterGap, with gapM, so nobody takes the hole for a continuous stretch.

Frame positions are the smoothed track, not the raw GPS fix, because raw fixes jitter by metres.
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

EARTH_R = 6371000.0

CSV_FIELDS = ["frame", "bin", "file", "timestamp", "video_time_s", "walked_start_m", "walked_end_m", "latitude",
              "longitude", "accuracy_m", "heading_deg", "sharpness", "low_sharpness", "time_uncertain", "after_gap", "gap_m"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path, help="original video file")
    p.add_argument("--gps", required=True, type=Path, help="gps.json of that recording")
    p.add_argument("--output", default=Path("frames_raw"), type=Path, help="output directory (default: ./frames_raw)")
    p.add_argument("--step", type=float, default=3.0, help="metres of path per frame (default 3)")
    p.add_argument("--max-accuracy", type=float, default=15.0, help="drop GPS points worse than this, in metres (default 15)")
    p.add_argument("--smooth", type=float, default=2.0, help="GPS smoothing, Gaussian sigma in seconds; 0 = off (default 2)")
    p.add_argument("--min-speed", type=float, default=0.25,
                   help="slower than this (m/s) counts as standing still, 0 distance (default 0.25)")
    p.add_argument("--time-offset", type=float, default=0.0,
                   help="seconds added to every GPS video time, to fix camera/GPS delay (default 0)")
    p.add_argument("--score-size", type=int, default=960,
                   help="frames are scored at this long-side size, in px; saved frames are always full size (default 960)")
    p.add_argument("--state-in", type=Path, help="state.json written by the previous recording of the same assignment")
    p.add_argument("--state-out", type=Path, help="write the state for the next recording here (default: <output>/state.json)")
    p.add_argument("--gap-max", type=float, default=15.0,
                   help="restarts farther than this from where the previous recording stopped are flagged afterGap, and the jump is not counted as distance walked (default 15)")
    p.add_argument("--prefix", help="file name prefix (default: the video's file name)")
    p.add_argument("--time-uncertain", action="store_true", help="mark every frame timeUncertain (the video and GPS clocks disagree)")
    a = p.parse_args()
    if a.step <= 0:
        p.error("--step must be > 0")
    return a


# ---------------------------------------------------------------- geometry

def haversine_m(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(h))


def track_from_points(points, max_accuracy):
    """Arrays t (video s), lat, lng, acc, wall-clock ms, sorted by time, without bad points. Also the number dropped."""
    good = [q for q in points if q.get("accuracyM") is not None and q["accuracyM"] <= max_accuracy]
    good.sort(key=lambda q: q["videoTimeS"])
    if len(good) < 2:
        raise ValueError(f"fewer than 2 usable GPS points (max accuracy {max_accuracy} m)")
    col = lambda k: np.array([q[k] for q in good], dtype=float)
    return col("videoTimeS"), col("latitude"), col("longitude"), col("accuracyM"), col("timestampMs"), len(points) - len(good)


def smooth(t, v, sigma):
    """Gaussian-weighted local straight-line fit over time (uneven spacing is fine).
    Unlike a plain moving average it does not pull the ends of the track inwards, so a recording
    keeps its full length and two recordings join without a fake gap."""
    if sigma <= 0:
        return v
    dt = t[None, :] - t[:, None]
    w = np.exp(-0.5 * (dt / sigma) ** 2)
    s0, s1, s2 = w.sum(1), (w * dt).sum(1), (w * dt * dt).sum(1)
    t0, t1 = (w * v[None, :]).sum(1), (w * dt * v[None, :]).sum(1)
    den = s0 * s2 - s1 * s1
    ok = den > 1e-9 * np.maximum(s0 * s2, 1e-12)
    return np.where(ok, (s2 * t0 - s1 * t1) / np.where(ok, den, 1.0), t0 / s0)


def sharpness(frame, size):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = size / max(h, w)
    if s < 1:
        g = cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- sampling

def sample(video: Path, points, out_dir: Path, *, step=3.0, max_accuracy=15.0, smooth_s=2.0, min_speed=0.25,
           time_offset=0.0, score_size=960, state_in=None, gap_max_m=15.0, time_uncertain=False, prefix=None,
           log=print):
    """Save the best frame of every `step` metres to out_dir. Returns (frames, state_out).

    frames: list of dicts (camelCase keys, see CSV_FIELDS). state_out: dict for the next recording of the
    same assignment: {"cumM", "nextFrame", "lastLat", "lastLng"}.
    """
    t, lat, lng, acc, tms, dropped = track_from_points(points, max_accuracy)
    t = t + time_offset
    lat0, lng0 = float(lat.mean()), float(lng.mean())
    kx = math.radians(1) * EARTH_R * math.cos(math.radians(lat0))
    ky = math.radians(1) * EARTH_R
    xs = smooth(t, (lng - lng0) * kx, smooth_s)
    ys = smooth(t, (lat - lat0) * ky, smooth_s)
    lat_s, lng_s = lat0 + ys / ky, lng0 + xs / kx  # the smoothed track, back in degrees

    d = np.hypot(np.diff(xs), np.diff(ys))
    dt = np.maximum(np.diff(t), 1e-6)
    d = np.where(d / dt < min_speed, 0.0, d)
    cum_local = np.concatenate([[0.0], np.cumsum(d)])

    # where this recording sits on the assignment-wide distance counter
    after_gap, gap_m, frame0 = False, 0.0, 0
    if state_in:
        gap_m = haversine_m(state_in["lastLat"], state_in["lastLng"], float(lat_s[0]), float(lng_s[0]))
        frame0 = int(state_in["nextFrame"])
        cum0 = float(state_in["cumM"])  # every later recording continues the walk
        if gap_m <= gap_max_m:
            cum0 += gap_m  # a short stretch walked while not recording still counts
        else:
            after_gap = True  # a hole: flagged, and the jump is not counted as walking
    else:
        cum0 = 0.0
    cum = cum0 + cum_local
    total = float(cum[-1])

    def time_at(dist):
        """First moment the walker has covered `dist` metres (cum is non-decreasing, may have flat parts)."""
        i = int(np.searchsorted(cum, dist, side="left"))
        if i <= 0:
            return float(t[0])
        if i >= len(t):
            return float(t[-1])
        return float(t[i - 1] + (dist - cum[i - 1]) / (cum[i] - cum[i - 1]) * (t[i] - t[i - 1]))

    # bins of the assignment-wide counter; a bin cut short by the start or end of the recording must still be
    # longer than half a step, so the two recordings around a join do not both take a frame from one bin
    bins = []
    for k in range(int(cum0 // step), int(total // step) + 1):
        d0, d1 = max(k * step, cum0), min((k + 1) * step, total)
        if d1 - d0 > 0.5 * step:
            t0, t1 = time_at(d0), time_at(d1)
            if t1 > t0:
                bins.append((k, t0, t1, d0, d1))
    log(f"gps: {len(t)} points used ({dropped} dropped), walked {cum_local[-1]:.1f} m, {len(bins)} bins of {step:g} m"
        + (f", after a {gap_m:.1f} m gap" if after_gap else ""))
    if not bins:
        raise ValueError("no bins: the path is shorter than the step, or no movement was detected")

    vx, vy = np.gradient(xs, t), np.gradient(ys, t)  # m/s east, north

    def at(ts):
        la, lo = float(np.interp(ts, t, lat_s)), float(np.interp(ts, t, lng_s))
        ac, wall = float(np.interp(ts, t, acc)), float(np.interp(ts, t, tms))
        ex, ny = float(np.interp(ts, t, vx)), float(np.interp(ts, t, vy))
        heading = round(math.degrees(math.atan2(ex, ny)) % 360, 1) if math.hypot(ex, ny) >= min_speed else None
        return la, lo, ac, wall, heading

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or video.stem
    frames, scores = [], []
    cur, best, n_read, last_ts = 0, None, 0, 0.0

    def flush(i):
        nonlocal best
        k, t0, t1, d0, d1 = bins[i]
        if best is None:
            log(f"warning: bin {k} ({t0:.1f}-{t1:.1f}s) has no frame, skipped")
            return
        score, ts, img = best
        best = None
        number = frame0 + len(frames)
        name = f"{prefix}_{number:04d}_{ts:08.3f}s.png"
        if not cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"cannot write {out_dir / name}")
        la, lo, ac, wall, heading = at(ts)
        first = not frames
        frames.append({
            "frame": number, "bin": k, "file": name, "timestamp": iso(wall), "videoTimeS": round(ts, 3),
            "walkedStartM": round(d0, 2), "walkedEndM": round(d1, 2), "latitude": round(la, 7),
            "longitude": round(lo, 7), "accuracyM": round(ac, 2), "headingDeg": heading, "sharpness": round(score, 2),
            "lowSharpness": False, "timeUncertain": bool(time_uncertain),
            "afterGap": bool(after_gap and first), "gapM": round(gap_m, 1) if after_gap and first else None,
        })
        scores.append(score)

    while cur < len(bins):
        ok, img = cap.read()
        if not ok:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        last_ts = ts
        n_read += 1
        while cur < len(bins) and ts >= bins[cur][2]:
            flush(cur)
            cur += 1
        if cur >= len(bins):
            break
        if ts < bins[cur][1]:
            continue
        s = sharpness(img, score_size)
        if best is None or s > best[0]:
            best = (s, ts, img)
    if cur < len(bins):  # the video ended inside a bin
        flush(cur)
        cur += 1
        if cur < len(bins):
            log(f"warning: video ended at {last_ts:.1f}s; {len(bins) - cur} later bin(s) skipped")
    cap.release()

    median = float(np.median(scores)) if scores else 0.0
    for f in frames:
        f["lowSharpness"] = bool(f["sharpness"] < 0.5 * median)  # a flag for review, nothing is removed
    log(f"read {n_read} frames, saved {len(frames)} to {out_dir}/ ({sum(f['lowSharpness'] for f in frames)} flagged lowSharpness)")

    end_ts = min(float(t[-1]), last_ts) if last_ts else float(t[-1])
    la_end, lo_end, *_ = at(end_ts)
    state_out = {"cumM": round(float(np.interp(end_ts, t, cum)), 3), "nextFrame": frame0 + len(frames),
                 "lastLat": round(la_end, 7), "lastLng": round(lo_end, 7)}
    return frames, state_out


def write_outputs(out_dir: Path, frames, state):
    (out_dir / "frames.json").write_text(json.dumps({"frames": frames, "state": state}, indent=2))
    camel = {"frame": "frame", "bin": "bin", "file": "file", "timestamp": "timestamp", "video_time_s": "videoTimeS",
             "walked_start_m": "walkedStartM", "walked_end_m": "walkedEndM", "latitude": "latitude",
             "longitude": "longitude", "accuracy_m": "accuracyM", "heading_deg": "headingDeg", "sharpness": "sharpness",
             "low_sharpness": "lowSharpness", "time_uncertain": "timeUncertain", "after_gap": "afterGap", "gap_m": "gapM"}
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for fr in frames:
            w.writerow(["" if fr[camel[c]] is None else int(fr[camel[c]]) if isinstance(fr[camel[c]], bool) else fr[camel[c]]
                        for c in CSV_FIELDS])


def main():
    a = parse_args()
    if not a.video.is_file():
        sys.exit(f"error: video not found: {a.video}")
    points = json.loads(a.gps.read_text())["points"]
    state_in = json.loads(a.state_in.read_text()) if a.state_in else None
    try:
        frames, state = sample(a.video, points, a.output, step=a.step, max_accuracy=a.max_accuracy, smooth_s=a.smooth,
                               min_speed=a.min_speed, time_offset=a.time_offset, score_size=a.score_size,
                               state_in=state_in, gap_max_m=a.gap_max, time_uncertain=a.time_uncertain, prefix=a.prefix)
    except ValueError as e:
        sys.exit(f"error: {e}")
    write_outputs(a.output, frames, state)
    (a.state_out or a.output / "state.json").write_text(json.dumps(state))
    print("note: these frames are NOT anonymized. Run anonymize.py on this folder before sharing or CVAT.")


if __name__ == "__main__":
    main()

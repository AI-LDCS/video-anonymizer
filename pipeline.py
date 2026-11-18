#!/usr/bin/env python3
"""Sample the sharpest frames from a video, anonymize them, ready for CVAT.

  --stage both        (default) sample frames, then anonymize them
  --stage sample      only sample: raw (NOT anonymized) frames in <output>/frames_raw
  --stage anonymize   only anonymize --input (image/video file or folder; default <output>/frames_raw)

With --purpose inference (the default here) the anonymized images also get GPS and the app's IDs written into
them (tag_gps.py). With --purpose training the images stay free of metadata; frames.json keeps it.

Layout under --output:
  frames_raw/    sampled frames + manifest.csv        (stage sample/both; NOT anonymized, keep private)
  anonymized/    anonymized copies                    (stage anonymize/both; this is what goes to CVAT)

This only runs sample_frames.py and anonymize.py; both also work on their own.
"""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["both", "sample", "anonymize"], default="both")
    p.add_argument("--output", default=Path("output"), type=Path, help="output root (default: ./output)")
    g = p.add_argument_group("sampling (stage sample/both)")
    g.add_argument("--video", type=Path, help="original video")
    g.add_argument("--gps", type=Path, help="gps.json of that recording")
    g.add_argument("--step", type=float, default=3.0, help="metres per frame (default 3)")
    g.add_argument("--assignment-json", type=Path,
                   help="assignment.json of the export: gives the route and assignment IDs for the image metadata")
    g.add_argument("--sample-extra", default="", help="more options for sample_frames.py, one quoted string")
    g = p.add_argument_group("anonymizing (stage anonymize/both)")
    g.add_argument("--input", type=Path, help="stage anonymize: what to anonymize (default <output>/frames_raw)")
    g.add_argument("--mode", choices=["pixelgauss", "gaussian", "pixelate", "solid"], default="pixelgauss")
    g.add_argument("--face-conf", type=float, default=0.15,
                   help="face threshold; lower than anonymize.py's 0.222, because single images have no neighbour frames (default 0.15)")
    g.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    g.add_argument("--purpose", choices=["inference", "training"], default="inference",
                   help="inference: write GPS and IDs into the images; training: leave the images clean (default inference)")
    g.add_argument("--anonymize-extra", default="", help="more options for anonymize.py, one quoted string, e.g. '--padding 0.3'")
    p.add_argument("--delete-raw", action="store_true", help="stage both: delete frames_raw after anonymizing (manifest.csv is kept)")
    a = p.parse_args()
    if a.stage in ("both", "sample") and not (a.video and a.gps):
        p.error("--video and --gps are required for --stage sample/both")
    if a.delete_raw and a.stage != "both":
        p.error("--delete-raw only works with --stage both")
    return a


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("+", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(f"error: {Path(cmd[1]).name} failed with exit code {r.returncode}")


def build_context(a):
    """The IDs shared by every frame of this recording, from gps.json and (if given) assignment.json."""
    ctx = {"purpose": a.purpose, "stepM": a.step}
    if a.gps:
        g = json.loads(a.gps.read_text())
        ctx.update({k: g[k] for k in ("assignmentId", "recordingId") if k in g})
    if a.assignment_json:
        e = json.loads(a.assignment_json.read_text())
        ctx["assignmentId"] = e["assignment"]["id"]
        ctx["routeId"] = e["route"]["id"]
        ctx["routeVersionId"] = e["route"]["versionId"]
        for r in e.get("recordings", []):
            if r.get("recordingId") == ctx.get("recordingId"):
                ctx["recordingIndex"] = r["index"]
    return ctx


def main():
    a = parse_args()
    raw, anon = a.output / "frames_raw", a.output / "anonymized"

    if a.stage in ("both", "sample"):
        run([sys.executable, HERE / "sample_frames.py", "--video", a.video, "--gps", a.gps,
             "--output", raw, "--step", a.step, *shlex.split(a.sample_extra)])

    if a.stage in ("both", "anonymize"):
        src = raw if a.stage == "both" else (a.input or raw)
        if not src.exists():
            sys.exit(f"error: nothing to anonymize, not found: {src}")
        run([sys.executable, HERE / "anonymize.py", "--input", src, "--output", anon, "--mode", a.mode,
             "--face-conf", a.face_conf, "--device", a.device, *shlex.split(a.anonymize_extra)])
        frames_json = src / "frames.json" if src.is_dir() else None
        if a.purpose == "inference" and frames_json and frames_json.is_file():
            ctx_file = a.output / "context.json"
            ctx_file.write_text(json.dumps(build_context(a)))
            run([sys.executable, HERE / "tag_gps.py", "--images", anon, "--frames", frames_json, "--context", ctx_file])
        manifest = raw / "manifest.csv"
        if a.stage == "both" and a.purpose == "inference":
            for name in ("manifest.csv", "frames.json"):  # the sidecar travels with the images
                if (raw / name).is_file():
                    shutil.copy2(raw / name, anon / name)
        if a.delete_raw:
            for f in raw.glob("*.png"):
                f.unlink()
            print(f"deleted raw frames in {raw}")

    if a.stage == "sample":
        print(f"\nraw frames in {raw} are NOT anonymized: do not upload them to CVAT.")
    else:
        print(f"\nanonymized output in {anon}")


if __name__ == "__main__":
    main()

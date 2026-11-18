#!/usr/bin/env python3
"""Write GPS and the app's IDs into already anonymized PNG images (EXIF).

anonymize.py strips every piece of metadata on purpose. This runs afterwards and adds back only:
  * the standard EXIF GPS block: latitude, longitude, accuracy, direction, date and time
  * ImageDescription: one JSON string with the IDs the scoring engine needs
    {"routeId","routeVersionId","assignmentId","recordingId","recordingIndex","frame","purpose","stepM"}
Nothing else is written (no worker, device, or camera information). The pixels are not changed: the file is
decoded again and compared before it replaces the original.

    python tag_gps.py --images anonymized/ --frames frames_raw/frames.json --context context.json

context.json holds the IDs shared by all frames: routeId, routeVersionId, assignmentId, recordingId,
recordingIndex, purpose, stepM. Every frame in frames.json whose file is in --images is tagged.
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

EXIF_GPS, IMAGE_DESCRIPTION = 0x8825, 0x010E
ALLOWED_TOP = {EXIF_GPS, IMAGE_DESCRIPTION}
ALLOWED_GPS = {1, 2, 3, 4, 7, 16, 17, 29, 31}  # Ref/Lat/Ref/Lng, TimeStamp, ImgDirectionRef/ImgDirection, DateStamp, HPositioningError
CONTEXT_KEYS = ("routeId", "routeVersionId", "assignmentId", "recordingId", "recordingIndex", "purpose", "stepM")


def dms(value):
    """Degrees (positive) as EXIF degrees/minutes/seconds rationals."""
    d = int(value)
    m = int((value - d) * 60)
    s = round(((value - d) * 60 - m) * 60, 4)
    return (IFDRational(d, 1), IFDRational(m, 1), IFDRational(round(s * 10000), 10000))


def gps_block(frame):
    lat, lng = frame["latitude"], frame["longitude"]
    t = datetime.fromisoformat(frame["timestamp"].replace("Z", "+00:00"))
    g = {
        1: "N" if lat >= 0 else "S", 2: dms(abs(lat)),
        3: "E" if lng >= 0 else "W", 4: dms(abs(lng)),
        7: (IFDRational(t.hour, 1), IFDRational(t.minute, 1), IFDRational(round((t.second + t.microsecond / 1e6) * 1000), 1000)),
        29: t.strftime("%Y:%m:%d"),
        31: IFDRational(round(frame["accuracyM"] * 100), 100),
    }
    if frame.get("headingDeg") is not None:
        g[16] = "T"  # true north
        g[17] = IFDRational(round(frame["headingDeg"] * 10), 10)
    return g


def description(frame, ctx):
    d = {k: ctx[k] for k in CONTEXT_KEYS if k in ctx}
    d["frame"] = frame["frame"]
    return json.dumps(d, separators=(",", ":"))


def check_only_allowed(path: Path):
    """The file may carry GPS and ImageDescription, nothing else."""
    with Image.open(path) as im:
        ex = im.getexif()
        extra = set(ex.keys()) - ALLOWED_TOP
        extra_gps = set(ex.get_ifd(EXIF_GPS).keys()) - ALLOWED_GPS
        other = set(im.info) - {"exif", "dpi", "gamma", "srgb"}  # PNG text/ICC/XMP chunks would show here
        if extra or extra_gps or other:
            raise RuntimeError(f"{path.name}: unexpected metadata {extra or extra_gps or other}")


def tag(path: Path, frame, ctx):
    with Image.open(path) as im:
        im.load()
        img = im.copy()
    before = np.asarray(img)
    ex = Image.Exif()
    ex[IMAGE_DESCRIPTION] = description(frame, ctx)
    ex[EXIF_GPS] = gps_block(frame)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".png")
    os.close(fd)
    try:
        img.save(tmp, format="PNG", exif=ex, compress_level=6)
        with Image.open(tmp) as out:
            out.load()
            if out.size != img.size or not np.array_equal(np.asarray(out), before):
                raise RuntimeError(f"{path.name}: pixels changed while writing metadata")
        check_only_allowed(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def tag_folder(images: Path, frames, ctx, log=print):
    n = 0
    for fr in frames:
        p = images / fr["file"]
        if not p.is_file():
            log(f"warning: {p.name} not found, not tagged")
            continue
        tag(p, fr, ctx)
        n += 1
    log(f"tagged {n} image(s) with GPS and IDs")
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, type=Path, help="folder with the anonymized PNG files")
    p.add_argument("--frames", required=True, type=Path, help="frames.json from sample_frames.py")
    p.add_argument("--context", required=True, type=Path, help="JSON with routeId, routeVersionId, assignmentId, recordingId, ...")
    a = p.parse_args()
    frames = json.loads(a.frames.read_text())["frames"]
    ctx = json.loads(a.context.read_text())
    try:
        tag_folder(a.images, frames, ctx)
    except Exception as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()

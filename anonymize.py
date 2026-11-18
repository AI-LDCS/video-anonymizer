#!/usr/bin/env python3
"""Anonymize faces and license plates in images and videos using the YOLO26 face/plate detector.

Detected regions are obscured with pixelation smoothed by a Gaussian blur (default),
a plain Gaussian blur, pixelation, or solid masks. Files are re-encoded from raw pixels (no EXIF/ICC/XMP/text metadata)
and written to a separate output directory. Source files are never modified.
Video audio is dropped unless --keep-audio is given.
"""
import argparse
import math
import os
import queue
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Ultralytics must never pip-install packages at runtime, and it monkey-patches PIL.Image.open
# (auto-installing pi-heif on any decode failure), so keep Pillow's original opener.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
PIL_OPEN = Image.open

MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolo26m_face_lp.pt"
MODEL_URL = ("https://github.com/H-Held/yolo26-license-plate-face-detection/"
             "releases/download/v3.0.0/yolo26m_face_lp.pt")
FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
SAVE_OPTS = {"JPEG": {"quality": 95}, "PNG": {"compress_level": 6}, "WEBP": {"quality": 95, "method": 4}}
# Video input suffix -> output suffix. Containers that are poor targets for H.264 become .mp4.
VIDEO_OUTPUT = {**{e: e for e in (".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".ts", ".mts", ".m2ts")},
                **{e: ".mp4" for e in (".wmv", ".flv", ".mpg", ".mpeg", ".3gp", ".ogv")}}
X264_OPTS = {"crf": "18", "preset": "medium"}
VP9_OPTS = {"crf": "30", "b": "0", "deadline": "realtime", "cpu-used": "5", "row-mt": "1"}
VP9_CHUNK_FRAMES = 120                         # VP9 is encoded in parallel chunks of this many frames (each starts on a keyframe)
CHUNK_BYTES = 400 << 20                        # ...fewer at high resolution, so 3 chunks in flight stay near 1.2 GiB
VERIFY_EVERY = 30                              # check 1 in N masked frames against the encoded file
# The only tags muxers add on their own. Anything else in an output fails verification.
VIDEO_TAGS_OK = {"major_brand", "minor_version", "compatible_brands", "encoder", "handler_name", "vendor_id",
                 "language", "duration", "service_name", "service_provider", "software"}
FACE, PLATE = "face", "license-plate"          # class labels shipped with the model
TILE_OVERLAP = 0.15                            # matches the model's training tiling (TILE_OVERLAP_PCT=15)
MIN_BLUR_KERNEL = 51
UMASK = os.umask(0); os.umask(UMASK)           # read once so outputs get normal file permissions


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="image/video file or directory")
    p.add_argument("--output", default=Path("output"), type=Path, help="output directory (default: ./output)")
    p.add_argument("--confidence", type=float, help="confidence threshold for both classes")
    p.add_argument("--face-conf", type=float, help="face threshold (default 0.222, overrides --confidence)")
    p.add_argument("--plate-conf", type=float, help="plate threshold (default 0.263, overrides --confidence)")
    p.add_argument("--imgsz", type=int, default=640, help="model input size (default 640, the training size)")
    p.add_argument("--mode", choices=["pixelgauss", "gaussian", "pixelate", "solid"], default="pixelgauss",
                   help="anonymization method (default: pixelgauss = pixelate, then smooth with a Gaussian blur)")
    p.add_argument("--blur-kernel", type=int, default=MIN_BLUR_KERNEL,
                   help=f"minimum Gaussian kernel size, odd, >= {MIN_BLUR_KERNEL}; grows with region size (default 51)")
    p.add_argument("--pixel-size", type=int, default=12,
                   help="minimum pixelation block size in px; large regions get coarser blocks (default 12)")
    p.add_argument("--padding", type=float, default=0.20, help="box expansion ratio per side (default 0.20)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--recursive", action="store_true", help="process sub-directories")
    p.add_argument("--tile", type=int, default=1280,
                   help="tile size in source pixels for large images; 0 = full-image pass only (default 1280)")
    p.add_argument("--batch", type=int, default=8, help="inference batch size for tiles (default 8)")
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)),
                   help="CPU threads for decoding/anonymizing/encoding, overlapped with GPU detection "
                        "(default: number of usable CPUs)")
    p.add_argument("--temporal-window", type=int, default=5,
                   help="video: also mask each frame with the boxes from this many frames before and after it, "
                        "so a one-frame detection miss does not reveal a face (default 5, 0 = off)")
    p.add_argument("--keep-audio", action="store_true",
                   help="video: keep the first audio track, re-encoded (default: drop audio, voices can identify people)")
    p.add_argument("--fill", default="0,0,0", help="solid-mode colour as R,G,B (default 0,0,0 = black)")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help="weights file")
    a = p.parse_args()
    base = a.confidence
    a.thr = {FACE: a.face_conf if a.face_conf is not None else (base if base is not None else 0.222),
             PLATE: a.plate_conf if a.plate_conf is not None else (base if base is not None else 0.263)}
    if not all(0 < t < 1 for t in a.thr.values()):
        p.error("confidence thresholds must be between 0 and 1")
    if a.padding < 0 or a.imgsz < 32 or a.batch < 1 or a.tile < 0 or a.workers < 1 or a.temporal_window < 0:
        p.error("invalid --padding/--imgsz/--batch/--tile/--workers/--temporal-window value")
    if a.blur_kernel < MIN_BLUR_KERNEL:
        p.error(f"--blur-kernel must be >= {MIN_BLUR_KERNEL}")
    if a.pixel_size < 2:
        p.error("--pixel-size must be >= 2")
    try:
        a.fill = tuple(int(v) for v in a.fill.split(","))
        assert len(a.fill) == 3 and all(0 <= v <= 255 for v in a.fill)
    except (ValueError, AssertionError):
        p.error("--fill must be R,G,B with values 0-255")
    return a


def collect(inp: Path, out_root: Path, recursive: bool):
    """Return (jobs, unsupported). jobs = [(src, dst)]. Output dir is excluded from scanning."""
    if inp.is_file():
        files, root = [inp], inp.parent
    elif inp.is_dir():
        root = inp
        files = sorted(f for f in (inp.rglob("*") if recursive else inp.iterdir()) if f.is_file())
        out_res = out_root.resolve()
        files = [f for f in files if out_res not in f.resolve().parents]
        if not recursive and any(d.is_dir() for d in inp.iterdir()):
            print("note: sub-directories ignored (use --recursive to include them)")
    else:
        sys.exit(f"error: input not found: {inp}")
    jobs, unsupported = [], []
    for f in files:
        if f.suffix.lower() in FORMATS:
            jobs.append((f, out_root / f.relative_to(root)))
        elif f.suffix.lower() in VIDEO_OUTPUT:
            jobs.append((f, (out_root / f.relative_to(root)).with_suffix(VIDEO_OUTPUT[f.suffix.lower()])))
        else:
            unsupported.append(f)
    return jobs, unsupported


def load_image(path: Path) -> np.ndarray:
    """Decode to an RGB/RGBA uint8 array with EXIF orientation baked in (EXIF is dropped on save)."""
    with PIL_OPEN(path) as im:
        # Camera JPEGs are often MPO (main image + embedded previews): keep frame 0 only, so the
        # unmasked previews are dropped. Genuinely animated WebP/APNG are rejected.
        if im.format != "MPO" and getattr(im, "n_frames", 1) > 1:
            raise ValueError("animated images are not supported")
        im.load()  # raises on truncated/corrupt data
        im = ImageOps.exif_transpose(im) or im
        alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        mode = "RGBA" if alpha else "RGB"
        return np.array(im if im.mode == mode else im.convert(mode))


def axis_starts(length: int, tile: int):
    if length <= tile:
        return [0]
    n = math.ceil((length - tile) / (tile * (1 - TILE_OVERLAP))) + 1
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


def detect(model, rgb: np.ndarray, a):
    """Full-image pass + overlapping tiles (as recommended by the model authors).
    Returns [(label, conf, x1, y1, x2, y2)] in image coordinates."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR if rgb.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    views = [(0, 0, bgr)]
    if a.tile and max(h, w) > a.tile:
        views += [(x, y, bgr[y:y + a.tile, x:x + a.tile])
                  for y in axis_starts(h, a.tile) for x in axis_starts(w, a.tile)]
    dets = []
    for i in range(0, len(views), a.batch):
        chunk = views[i:i + a.batch]
        results = model.predict([v[2] for v in chunk], imgsz=a.imgsz, conf=min(a.thr.values()),
                                device=a.device, verbose=False)
        for (ox, oy, _), r in zip(chunk, results):
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                label = model.names[int(k)]
                if label in a.thr and c >= a.thr[label]:
                    dets.append((label, c, x1 + ox, y1 + oy, x2 + ox, y2 + oy))
    return dets


def count_unique(dets, label):
    """Count objects, merging duplicates from overlapping tiles/full pass (intersection / smaller area > 0.6).
    Only used for reporting; every raw box is still masked."""
    kept = []
    for _, _, *b in sorted((d for d in dets if d[0] == label), key=lambda d: -d[1]):
        for k in kept:
            iw = min(b[2], k[2]) - max(b[0], k[0])
            ih = min(b[3], k[3]) - max(b[1], k[1])
            small = min((b[2] - b[0]) * (b[3] - b[1]), (k[2] - k[0]) * (k[3] - k[1]))
            if iw > 0 and ih > 0 and iw * ih > 0.6 * small:
                break
        else:
            kept.append(b)
    return len(kept)


def anonymize(arr: np.ndarray, dets, a):
    """Obscure every padded, clipped box in place. Returns the pixel regions that were changed.
    Every raw detection is processed (overlaps included), so tile duplicates never leave gaps."""
    h, w = arr.shape[:2]
    regions = []
    for _, _, x1, y1, x2, y2 in dets:
        px, py = (x2 - x1) * a.padding, (y2 - y1) * a.padding
        l, t = max(0, math.floor(x1 - px)), max(0, math.floor(y1 - py))
        r, b = min(w, math.ceil(x2 + px)), min(h, math.ceil(y2 + py))
        if r <= l or b <= t:
            continue
        rw, rh = r - l, b - t
        roi = arr[t:b, l:r, :3]
        if a.mode == "solid":
            roi[:] = a.fill
        elif a.mode == "gaussian":
            # Adaptive: kernel ~ half the region's longer side (never below --blur-kernel), always odd.
            k = max(a.blur_kernel, max(rw, rh) // 2) | 1
            roi[:] = cv2.GaussianBlur(np.ascontiguousarray(roi), (k, k), 0, borderType=cv2.BORDER_REFLECT)
        else:  # pixelate: average into blocks, then nearest-neighbour upscale (<= ~16 blocks per side)
            block = max(a.pixel_size, math.ceil(max(rw, rh) / 16))
            small = cv2.resize(np.ascontiguousarray(roi), (max(1, round(rw / block)), max(1, round(rh / block))),
                               interpolation=cv2.INTER_AREA)
            roi[:] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            if a.mode == "pixelgauss":
                # Smooth the hard block edges: kernel ~ twice the block size, always odd.
                k = (2 * block) | 1
                roi[:] = cv2.GaussianBlur(np.ascontiguousarray(roi), (k, k), 0, borderType=cv2.BORDER_REFLECT)
        if arr.shape[2] == 4:
            arr[t:b, l:r, 3] = 255  # opaque, so the alpha channel cannot reveal the region's shape
        regions.append((l, t, r, b))
    return regions


def save(arr: np.ndarray, dst: Path, fmt: str, regions):
    """Encode from raw pixels into a temp file, verify, then atomically move into place."""
    img = Image.fromarray(arr)  # fresh image object: carries no EXIF/ICC/XMP/text chunks
    if fmt == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp-", suffix=dst.suffix)
    os.close(fd)
    try:
        os.chmod(tmp, 0o666 & ~UMASK)
        img.save(tmp, format=fmt, **SAVE_OPTS[fmt])
        with PIL_OPEN(tmp) as out:
            meta = set(out.info) & {"exif", "icc_profile", "xmp", "XML:com.adobe.xmp", "comment"}
            if out.size != img.size or len(out.getexif()) or meta:
                raise RuntimeError(f"output verification failed (size/metadata: {meta or 'exif'})")
            got = np.asarray(out.convert("RGB"))
        # Decoded regions must match the anonymized pixels (exact for PNG, small codec error otherwise).
        tol = 0 if fmt == "PNG" else 12
        for l, t, r, b in regions:
            if np.abs(got[t:b, l:r].astype(np.int16) - arr[t:b, l:r, :3]).mean() > tol:
                raise RuntimeError(f"mask verification failed for region {(l, t, r, b)}")
        os.replace(tmp, dst)
    except BaseException:
        os.unlink(tmp)
        raise


def merge_boxes(dets):
    """Replace overlapping boxes with their bounding box, so a region is obscured once per frame
    however many window frames contributed to it. The result always covers every input box."""
    boxes = [list(d[2:]) for d in dets]
    merged = True
    while merged:
        merged = False
        kept = []
        for b in boxes:
            for k in kept:
                if b[0] < k[2] and k[0] < b[2] and b[1] < k[3] and k[1] < b[3]:
                    k[:] = min(b[0], k[0]), min(b[1], k[1]), max(b[2], k[2]), max(b[3], k[3])
                    merged = True
                    break
            else:
                kept.append(b)
        boxes = kept
    return [("", 0.0, *b) for b in boxes]


def video_frame_to_rgb(frame) -> np.ndarray:
    """Decode to an RGB array with the rotation flag baked in (the output carries no rotation flag).
    Sizes are cropped to even numbers, because 4:2:0 encoding requires them."""
    colorspace = frame.colorspace
    if colorspace == 2:  # unspecified: the usual convention is BT.709 for HD, BT.601 below it
        colorspace = 1 if frame.height >= 720 else 5
    arr = frame.reformat(format="rgb24", src_colorspace=colorspace).to_ndarray()
    if frame.rotation:
        arr = np.rot90(arr, round(frame.rotation / 90) % 4)
    h, w = arr.shape[:2]
    return arr[:h & ~1, :w & ~1].copy()


def detect_frames(model, arrs, a):
    """Same result format as detect(), for a list of same-size video frames. Full-frame views and tiles are
    batched separately: same-shape batches run at their true aspect ratio instead of being padded to square."""
    h, w = arrs[0].shape[:2]
    bgrs = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in arrs]
    full = [(i, 0, 0, b) for i, b in enumerate(bgrs)]
    tiles = []
    if a.tile and max(h, w) > a.tile:
        tiles = [(i, x, y, b[y:y + a.tile, x:x + a.tile]) for i, b in enumerate(bgrs)
                 for y in axis_starts(h, a.tile) for x in axis_starts(w, a.tile)]
    dets = [[] for _ in arrs]
    for views in (full, tiles):
        for j in range(0, len(views), a.batch):
            chunk = views[j:j + a.batch]
            results = model.predict([v[3] for v in chunk], imgsz=a.imgsz, conf=min(a.thr.values()),
                                    device=a.device, verbose=False)
            for (i, ox, oy, _), r in zip(chunk, results):
                for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    label = model.names[int(k)]
                    if label in a.thr and c >= a.thr[label]:
                        dets[i].append((label, c, x1 + ox, y1 + oy, x2 + ox, y2 + oy))
    return dets


class Segment:
    """One encoder fed from a queue on its own thread. Its packets are muxed in order by the main thread."""

    def __init__(self, maxsize):
        self.frames = queue.Queue(maxsize)
        self.packets = deque()
        self.future = None


def encode_video(model, src: Path, tmp: str, a):
    """Decode, detect, anonymize and encode one video into tmp. Returns stats plus the samples for verification.

    Pipeline: a reader thread decodes -> the main thread runs GPU detection in batches -> encoder threads
    anonymize and encode -> the main thread muxes. H.264 uses one encoder, which is already fast enough. VP9
    is slower and stops scaling at about 3 cores, so it runs in parallel chunks that are joined in order."""
    import av
    from av.video.reformatter import ColorRange, Colorspace

    suffix = Path(tmp).suffix
    is_webm = suffix == ".webm"
    with av.open(str(src)) as probe:  # the output streams must exist before audio packets can be written
        first = next(probe.decode(video=0), None)
        if first is None:
            raise ValueError("no decodable video frames")
        height, width = video_frame_to_rgb(first).shape[:2]
        del first

    options = {"movflags": "faststart"} if suffix in (".mp4", ".m4v", ".mov") else {}
    stop = threading.Event()
    with av.open(str(src)) as inp, av.open(tmp, "w", options=options) as out:
        video_in = inp.streams.video[0]
        video_in.thread_type = "AUTO"
        fps = video_in.average_rate or 30
        time_base = video_in.time_base
        codec_name, codec_opts = ("libvpx-vp9", VP9_OPTS) if is_webm else ("libx264", X264_OPTS)
        if suffix == ".avi":  # AVI stores no separate display timestamps, so B-frames would come back out of order
            codec_opts = {**codec_opts, "bf": "0"}
        n_encoders = max(1, a.workers // 2) if is_webm else 1
        chunk_frames = (max(24, min(VP9_CHUNK_FRAMES, CHUNK_BYTES // (width * height * 3))) if is_webm else None)
        encoder_threads = 2 if is_webm else a.workers

        def configure(ctx):
            ctx.width, ctx.height, ctx.pix_fmt, ctx.time_base = width, height, "yuv420p", time_base
            ctx.thread_count = encoder_threads
            ctx.colorspace, ctx.color_primaries, ctx.color_trc, ctx.color_range = 1, 1, 1, 1  # BT.709, limited range

        video_out = out.add_stream(codec_name, rate=fps, options=codec_opts)
        configure(video_out.codec_context)

        audio_in = inp.streams.audio[0] if a.keep_audio and inp.streams.audio else None
        audio_out = None
        if audio_in is not None:
            audio_codec = {".webm": "libopus", ".avi": "libmp3lame"}.get(suffix, "aac")  # AVI cannot mux AAC
            audio_out = out.add_stream(audio_codec, rate=48000 if is_webm else (audio_in.rate or 48000), layout="stereo")
        audio_packets = deque()

        decoded = queue.Queue(max(2, a.batch * 2))

        def put(q, item, alive=lambda: None):  # never blocks forever: the consumer may have stopped on an error
            while not stop.is_set():
                alive()
                try:
                    q.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    pass
            return False

        def read():
            """Decode on its own thread, so the GPU never waits for it. Runs until end of file, an error or stop."""
            try:
                frame_step = max(1, round(1 / (fps * time_base)))
                last_pts = None
                streams = [video_in] + ([audio_in] if audio_in is not None else [])
                for packet in inp.demux(*streams):
                    for frame in packet.decode():
                        if packet.stream is audio_in:
                            audio_packets.extend(audio_out.encode(frame))
                            continue
                        arr = video_frame_to_rgb(frame)
                        if arr.shape[:2] != (height, width):
                            raise ValueError("resolution changes mid-stream are not supported")
                        pts = frame.pts if frame.pts is not None else (0 if last_pts is None else last_pts + frame_step)
                        pts = pts if last_pts is None else max(pts, last_pts + 1)
                        last_pts = pts
                        if not put(decoded, (pts, arr)):
                            return
                if audio_out is not None:
                    audio_packets.extend(audio_out.encode(None))
                put(decoded, None)
            except BaseException as e:  # handed to the main thread, which fails this video
                put(decoded, e)

        stats = {"frames": 0, "masked": 0, "regions": 0, "face_frames": 0, "plate_frames": 0}
        samples = {}

        def encode_segment(ctx, seg):
            """Anonymize and encode one segment's frames. Runs on an encoder thread."""
            masked = n_regions = 0
            while True:
                try:
                    item = seg.frames.get(timeout=0.2)
                except queue.Empty:
                    if stop.is_set():
                        return masked, n_regions
                    continue
                if item is None:
                    break
                index, pts, arr, boxes = item
                regions = anonymize(arr, boxes, a)
                if regions:
                    if masked % VERIFY_EVERY == 0:
                        samples[index] = [(reg, cv2.resize(arr[reg[1]:reg[3], reg[0]:reg[2]], (8, 8),
                                                           interpolation=cv2.INTER_AREA)) for reg in regions]
                    masked += 1
                    n_regions += len(regions)
                out_frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(
                    format="yuv420p", dst_colorspace=Colorspace.ITU709, dst_color_range=ColorRange.MPEG)
                out_frame.pts, out_frame.time_base = pts, time_base
                seg.packets.extend(ctx.encode(out_frame))
            seg.packets.extend(ctx.encode(None))
            return masked, n_regions

        segments = deque()
        muxed_seconds = 0.0

        def mux_video(packet):
            nonlocal muxed_seconds
            packet.stream = video_out  # chunk encoders are separate from the output stream's own context
            if packet.pts is not None and packet.time_base:
                muxed_seconds = max(muxed_seconds, float(packet.pts * packet.time_base))
            out.mux(packet)

        def drain(final=False):
            """Mux whatever is ready, in order. Audio is held back to the video position to keep them interleaved."""
            while segments:
                seg = segments[0]
                done = seg.future.done()  # read first: packets added before this point are all taken below
                while seg.packets:
                    mux_video(seg.packets.popleft())
                if not done:
                    break
                masked, n_regions = seg.future.result()
                stats["masked"] += masked
                stats["regions"] += n_regions
                segments.popleft()
            while audio_packets and (final or audio_packets[0].pts is None or
                                     float(audio_packets[0].pts * audio_packets[0].time_base) <= muxed_seconds):
                out.mux(audio_packets.popleft())

        window = a.temporal_window
        recent = deque(maxlen=2 * window + 1)  # (index, detections) of the frames around the one being written
        pending = deque()                      # (index, pts, pixels) detected, waiting for their later neighbours
        current = None
        current_count = 0

        with ThreadPoolExecutor(n_encoders, "encode") as pool:
            def start_segment():
                nonlocal current, current_count
                while len(segments) >= n_encoders:  # at most n_encoders chunks in flight bounds the RAM used
                    segments[0].future.exception()   # waits; drain() below re-raises any error
                    drain()
                if chunk_frames is None:
                    ctx = video_out.codec_context
                else:
                    ctx = av.CodecContext.create(codec_name, "w")
                    configure(ctx)
                    ctx.framerate = fps
                    ctx.options = dict(codec_opts)
                current = Segment(chunk_frames or 32)
                current.future = pool.submit(encode_segment, ctx, current)
                current_count = 0
                segments.append(current)

            def finish_segment():
                nonlocal current
                if current is not None:
                    seg, current = current, None
                    put(seg.frames, None, lambda: seg.future.done() and seg.future.result())

            def write(index, pts, arr):
                nonlocal current_count
                if current is None:
                    start_segment()
                boxes = merge_boxes([d for i, ds in recent if abs(i - index) <= window for d in ds])
                seg = current
                put(seg.frames, (index, pts, arr, boxes), lambda: seg.future.done() and seg.future.result())
                current_count += 1
                if chunk_frames and current_count >= chunk_frames:
                    finish_segment()

            def detect_batch(batch):
                for (pts, arr), dets in zip(batch, detect_frames(model, [b[1] for b in batch], a)):
                    index = stats["frames"]
                    stats["frames"] += 1
                    stats["face_frames"] += any(d[0] == FACE for d in dets)
                    stats["plate_frames"] += any(d[0] == PLATE for d in dets)
                    recent.append((index, dets))
                    pending.append((index, pts, arr))
                    if len(pending) > window:
                        write(*pending.popleft())

            reader = threading.Thread(target=read, name="decode", daemon=True)
            reader.start()
            try:
                batch = []
                last_print = time.perf_counter()
                while True:
                    item = decoded.get()
                    if isinstance(item, BaseException):
                        raise item
                    if item is not None:
                        batch.append(item)
                    if batch and (item is None or len(batch) >= a.batch):
                        detect_batch(batch)
                        batch = []
                        drain()
                        now = time.perf_counter()
                        if now - last_print >= 30:
                            last_print = now
                            print(f"  ...        {src}: {stats['frames']}/{video_in.frames or '?'} frames")
                    if item is None:
                        break
                while pending:
                    write(*pending.popleft())
                finish_segment()
                while segments:
                    segments[0].future.exception()
                    drain()
                drain(final=True)
            finally:
                stop.set()
                reader.join()
        stats["size"] = (width, height)
        stats["source_frames"] = video_in.frames
        stats["audio"] = "none" if not inp.streams.audio else "kept" if audio_out is not None else "dropped"
    return stats, samples


def verify_video(tmp: str, stats, samples):
    """Decode the encoded file and check its streams, tags, frame count and a sample of masked regions."""
    import av

    with av.open(tmp) as c:
        expected = ["video"] + (["audio"] if stats["audio"] == "kept" else [])
        if [s.type for s in c.streams] != expected:
            raise RuntimeError(f"output verification failed (streams: {[s.type for s in c.streams]})")
        tags = {k.lower() for src in (c, *c.streams) for k in src.metadata} - VIDEO_TAGS_OK
        if tags:
            raise RuntimeError(f"output verification failed (metadata: {tags})")
        count = 0
        for frame in c.decode(video=0):
            if count == 0 and (frame.width, frame.height) != stats["size"]:
                raise RuntimeError("output verification failed (size)")
            if count in samples:
                got = frame.reformat(format="rgb24").to_ndarray()
                for (l, t, r, b), thumb in samples[count]:
                    decoded = cv2.resize(got[t:b, l:r], (8, 8), interpolation=cv2.INTER_AREA)
                    if np.abs(decoded.astype(np.int16) - thumb).mean() > 12:
                        raise RuntimeError(f"mask verification failed for frame {count}, region {(l, t, r, b)}")
            count += 1
    if count != stats["frames"]:
        raise RuntimeError(f"output verification failed (frames: wrote {stats['frames']}, decoded {count})")


def process_video(model, src: Path, dst: Path, a):
    """Encode into a temp file, verify, then atomically move into place."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp-", suffix=dst.suffix)
    os.close(fd)
    try:
        os.chmod(tmp, 0o666 & ~UMASK)
        stats, samples = encode_video(model, src, tmp, a)
        verify_video(tmp, stats, samples)
        os.replace(tmp, dst)
    except BaseException:
        os.unlink(tmp)
        raise
    return stats


def main():
    a = parse_args()
    out_root = a.output
    if a.input.resolve() == out_root.resolve():
        sys.exit("error: --output must differ from --input")
    jobs, unsupported = collect(a.input, out_root, a.recursive)
    for f in unsupported:
        print(f"UNSUPPORTED  {f}  (supported: {', '.join([*FORMATS, *VIDEO_OUTPUT])})")
    if not jobs:
        print("no supported images or videos found")
        return 1 if unsupported else 0
    videos = [j for j in jobs if j[0].suffix.lower() in VIDEO_OUTPUT]
    images = [j for j in jobs if j[0].suffix.lower() in FORMATS]
    if videos:
        try:
            import av  # noqa: F401
        except ImportError:
            sys.exit("error: video input needs PyAV: pip install -r requirements.txt")

    import torch
    from ultralytics import YOLO
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("error: --device cuda requested but CUDA is not available")
    a.device = "cpu" if a.device == "cpu" or not torch.cuda.is_available() else 0
    if not a.model.is_file():
        sys.exit(f"error: model not found at {a.model}\ndownload: curl -L -o {a.model} {MODEL_URL}")
    model = YOLO(str(a.model))  # loaded once, reused for every image
    missing = {FACE, PLATE} - set(model.names.values())
    if missing:
        sys.exit(f"error: model lacks expected classes {missing}; has {model.names}")
    print(f"model: {a.model.name}  device: {'cuda' if a.device == 0 else 'cpu'}  imgsz: {a.imgsz}  "
          f"tile: {a.tile or 'off'}  workers: {a.workers}  mode: {a.mode}  conf: face={a.thr[FACE]} plate={a.thr[PLATE]}  padding: {a.padding}")

    # Pipeline: loader threads decode ahead -> main thread runs GPU detection (one model, not shared
    # across threads) -> writer threads anonymize/encode/verify. In-flight images are bounded to cap RAM.
    sources = {s.resolve() for s, _ in jobs}
    ok, failed, faces, plates, frames, t0 = 0, [], 0, 0, 0, time.perf_counter()

    def finish(src, dst, arr, dets):
        regions = anonymize(arr, dets, a)
        save(arr, dst, FORMATS[src.suffix.lower()], regions)
        return len(regions), arr.shape

    def failed_future(e):  # failures travel through the same ordered queue, so output stays in input order
        f = Future()
        f.set_exception(e)
        return f

    def report(src, dst, dets, fut):
        nonlocal ok, faces, plates
        try:
            n_regions, shape = fut.result()
        except Exception as e:  # report and continue; nothing is exported for this image
            failed.append(src)
            print(f"FAILED       {src}: {type(e).__name__}: {e}")
            return
        nf, npl = count_unique(dets, FACE), count_unique(dets, PLATE)
        faces, plates, ok = faces + nf, plates + npl, ok + 1
        print(f"OK           {src} -> {dst}  faces={nf} plates={npl} regions={n_regions} {shape[1]}x{shape[0]}")

    with ThreadPoolExecutor(2, "load") as loader, ThreadPoolExecutor(a.workers, "write") as writer:
        todo, loading, writing = iter(images), deque(), deque()

        def prefetch():
            for src, dst in todo:
                fut = (failed_future(RuntimeError("output path would overwrite a source file"))
                       if dst.resolve() in sources else loader.submit(load_image, src))
                loading.append((src, dst, fut))
                return

        for _ in range(2):
            prefetch()
        while loading:
            src, dst, fut = loading.popleft()
            prefetch()
            try:
                arr = fut.result()
                dets = detect(model, arr, a)
                writing.append((src, dst, dets, writer.submit(finish, src, dst, arr, dets)))
                del arr
            except Exception as e:
                writing.append((src, dst, [], failed_future(e)))
            while len(writing) > a.workers or (writing and writing[0][3].done()):
                report(*writing.popleft())
        while writing:
            report(*writing.popleft())

    # Videos run one at a time, each with its own decode/detect/encode pipeline (see encode_video).
    claimed = set()
    for src, dst in videos:
        try:
            if dst.resolve() in sources or dst.resolve() in claimed:
                raise RuntimeError(f"output path would overwrite another file: {dst}")
            claimed.add(dst.resolve())
            video_t0 = time.perf_counter()
            st = process_video(model, src, dst, a)
        except Exception as e:  # report and continue; nothing is exported for this video
            failed.append(src)
            print(f"FAILED       {src}: {type(e).__name__}: {e}")
            continue
        ok, frames = ok + 1, frames + st["frames"]
        video_dt = time.perf_counter() - video_t0
        print(f"OK           {src} -> {dst}  frames={st['frames']} masked_frames={st['masked']} "
              f"face_frames={st['face_frames']} plate_frames={st['plate_frames']} audio={st['audio']} "
              f"{st['size'][0]}x{st['size'][1]} {st['frames'] / video_dt:.1f} fps")
        if st["source_frames"] and st["frames"] < st["source_frames"]:
            print(f"WARNING      {src}: only {st['frames']} of {st['source_frames']} frames could be decoded "
                  "(truncated source?); the output is shorter")

    dt = time.perf_counter() - t0
    print(f"\nprocessed: {ok}/{len(jobs)}  failed: {len(failed)}  unsupported: {len(unsupported)}  "
          f"faces: {faces}  plates: {plates}  video frames: {frames}  time: {dt:.1f}s ({dt / max(len(jobs), 1):.2f}s/file)")
    print("note: detection is not perfect; review outputs before publishing.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

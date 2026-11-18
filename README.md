# Image & Video Anonymizer (YOLO26 face & license plate)

Detects **faces** and **license plates** with the
[YOLO26 face/plate model](https://github.com/H-Held/yolo26-license-plate-face-detection)
(`yolo26m_face_lp.pt`, v3.0.0, classes `face` / `license-plate`). It then obscures each
detection with a Gaussian blur, pixelation or a solid mask. Outputs are re-encoded with all
metadata removed, and source files are never modified. Video audio is dropped by default.

## Install

Python 3.10 or newer. Install PyTorch for your hardware first (see pytorch.org; the CUDA build for a
GPU), then the rest:

```bash
pip install -r requirements.txt
mkdir -p models
curl -L -o models/yolo26m_face_lp.pt \
  https://github.com/H-Held/yolo26-license-plate-face-detection/releases/download/v3.0.0/yolo26m_face_lp.pt
```

## Usage

```bash
PY=python

# Directory, default mode (pixelgauss: pixelate, then smooth with a Gaussian blur)
$PY anonymize.py --input ./images --output ./output

# Single image
$PY anonymize.py --input ./street.jpg --output ./output

# Single video, keeping the audio (a directory can mix images and videos)
$PY anonymize.py --input ./street.mp4 --output ./output --keep-audio

# Pixelation with 16 px minimum blocks
$PY anonymize.py --input ./images --output ./output --mode pixelate --pixel-size 16

# Solid black masks (irreversible – recommended for strict anonymization)
$PY anonymize.py --input ./images --output ./output --mode solid

# Stronger blur, more padding, lower thresholds, sub-directories, GPU
$PY anonymize.py --input ./images --output ./output --mode gaussian --blur-kernel 101 \
    --padding 0.3 --confidence 0.15 --recursive --device cuda
```

### Modes

| `--mode` | Effect | Tuning |
|---|---|---|
| `pixelgauss` (default) | Pixelate as below, then a Gaussian blur over the region (kernel = 2 × block size, odd) to remove the hard block edges. | `--pixel-size` (default 12) |
| `gaussian` | Gaussian blur. Kernel = max(`--blur-kernel`, ½ × region's longer side), always odd. | `--blur-kernel` (min 51, default 51) |
| `pixelate` | Area-downscale then nearest-neighbour upscale. Block = max(`--pixel-size`, longer side / 16), so a region is never more than ~16 blocks across. | `--pixel-size` (default 12) |
| `solid` | Pixels replaced with a solid colour (black). | `--fill R,G,B` |

In all modes each box is expanded by `--padding` (default **0.20** of the box width/height on
each side) and clipped to the image. For RGBA images the region is also made opaque.

> **Privacy note:** blur and pixelation keep a transformed version of the original pixels.
> Their strength is adaptive, but low-detail content such as plate characters can sometimes be
> partially reconstructed or guessed. `--mode solid` is the only irreversible option.

### All options

| Option | Default | Description |
|---|---|---|
| `--input` | required | image/video file or directory |
| `--output` | `./output` | output directory (must differ from input) |
| `--mode` | `pixelgauss` | `pixelgauss`, `gaussian`, `pixelate` or `solid` |
| `--blur-kernel` | 51 | minimum Gaussian kernel (≥ 51) |
| `--pixel-size` | 12 | minimum pixelation block size (px) |
| `--padding` | 0.20 | box expansion ratio per side |
| `--confidence` | – | threshold for both classes |
| `--face-conf` / `--plate-conf` | 0.222 / 0.263 | per-class thresholds (model authors' recommendation). They override `--confidence`. |
| `--imgsz` | 640 | model input size (the model was trained at 640) |
| `--tile` | 1280 | tile size for large images; `0` disables tiling |
| `--batch` | 8 | tiles per inference batch |
| `--workers` | usable CPUs | CPU threads for decode/anonymize/encode, overlapped with GPU detection |
| `--device` | `auto` | `auto`, `cpu`, `cuda` |
| `--recursive` | off | include sub-directories (the structure is mirrored in the output) |
| `--temporal-window` | 5 | video: also mask each frame with the boxes from this many frames before and after it; `0` disables |
| `--keep-audio` | off | video: keep the first audio track, re-encoded as stereo |

## Still images (sampling)

For labelling (CVAT) and for the scoring engine you rarely want the whole video, but one sharp, anonymized picture
every few metres, with its position. Three small tools, each usable on its own:

| Tool | What it does |
|---|---|
| `sample_frames.py` | video + `gps.json` -> the sharpest frame in every N metres walked (default 3), as lossless PNG, with `frames.json` and `manifest.csv`. **Raw: not anonymized.** |
| `anonymize.py` | hides faces and plates (above). Run it on the folder the sampler wrote. |
| `tag_gps.py` | writes GPS and the app's IDs into the anonymized PNGs (EXIF), after anonymizing. |
| `pipeline.py` | runs them in order. `--stage both` (default), `sample` (stop before anonymizing) or `anonymize` (only hide faces). |

```bash
# everything: sample, anonymize, tag. Result in output/anonymized/
python pipeline.py --video recording.webm --gps gps.json --assignment-json assignment.json --output output --step 3

# training images (CVAT): no metadata inside the images
python pipeline.py --video recording.webm --gps gps.json --output output --purpose training

# only sample (raw frames!), or only anonymize some folder
python pipeline.py --stage sample --video recording.webm --gps gps.json --output output
python pipeline.py --stage anonymize --input output/frames_raw --output output
```

**How the frames are picked.** GPS points worse than 15 m are dropped, the track is smoothed (a local straight-line
fit, so the ends are not pulled inwards), and the distance walked is added up; standing still counts as zero. The path
is cut into bins of `--step` metres, each bin is mapped to a time range in the video, and every frame in it is scored
with the variance of the Laplacian (higher is sharper). The best frame wins; a bin where everything is blurry still gets
its least-blurry one, flagged `lowSharpness`. Frame times come from the video's own timestamps, because phone video has
variable frame rate. Frames are saved at full size.

**Frame position** is the smoothed track, not the raw fix, and `headingDeg` is the walking direction (`null` when standing still).

**Several recordings of one walk.** `sample_frames.py --state-out state.json` writes the distance counter and the next
frame number; pass it as `--state-in` for the next recording. The spacing then carries on across the restart, and a jump
of more than `--gap-max` metres (default 15) is flagged: that frame is marked `afterGap` (with `gapM`) and the jump is not counted
as walking, but the count still carries on, because a later recording is always the continuation of the earlier ones (the
worker is told to restart where they stopped). The worker does this itself.

**What `tag_gps.py` writes** (nothing else; the pixels are checked to be unchanged): the EXIF GPS block (latitude,
longitude, accuracy, direction, date and time) and `ImageDescription` = one JSON string
`{"routeId","routeVersionId","assignmentId","recordingId","recordingIndex","purpose","stepM","frame"}`. No worker or device
information. `anonymize.py` itself still strips all metadata; tagging happens afterwards.

**Face threshold.** For a single image there are no neighbouring frames to catch a missed face, so the pipeline and the
worker use `--face-conf 0.15` (anonymize.py's own default is 0.222). Check the output before sharing: the model still
misses small or turned-away faces.

## Worker (automatic, for the field-capture app)

`worker.py` runs `anonymize.py` automatically on every recording the field-capture app has finished
and verified. It polls the app, so the GPU machine only needs HTTPS access to it (no database or
storage credentials). Standard library only.

```bash
export ANONYMIZER_TOKEN=...   # same value as ANONYMIZER_TOKEN in the app's .env (32+ characters)
$PY worker.py --server https://your-host.your-tailnet.ts.net           # runs until stopped
$PY worker.py --server https://... --once                              # drain the queue, then exit
$PY worker.py --server https://... --mode pixelgauss --device cuda --extra "--padding 0.3"
```

It takes two kinds of job (`--kinds frames,video`, the default; frames first):

- **frames**: downloads the video and GPS, runs `sample_frames.py`, then `anonymize.py` on those frames, then `tag_gps.py`
  if the assignment is for inference, and uploads every image. Recordings of one assignment are done in order, so the
  distance counter carries on (a later recording is the continuation of the earlier ones). The raw frames stay in a temporary folder here and are deleted. The step is the server's
  `SAMPLE_STEP_M`.
- **video**: claims a job, downloads the original, runs `anonymize.py` (default `--mode pixelgauss`; `--mode solid` is the
  irreversible one), checks the output and uploads the whole anonymized video.

For both it checks the output and uploads it. A heartbeat keeps the claim
alive; if the worker dies the app gives the job to the next claim, and after 3 failed attempts it
marks the recording failed for an admin to retry. `Ctrl+C` or `SIGTERM` gives the job back at once.
Several workers can run side by side. Details, failure handling and who may download what are in the
app's `docs/operations/anonymizer.md`.

## Using the GPU and CPU fully

What is already overlapped inside one video (`anonymize.py`): a thread decodes, the main thread runs GPU
detection in batches, and encoder threads mask and encode, so the GPU and CPUs work at the same time.
On the GTX 1660 Ti a 1080p video needs 3 model views per frame at about 26 ms each, and it runs at about
11.4 fps, so **the GPU is already busy about 90% of the time and is the limit**. More CPU cores or a second
process will not make a video much faster on that card. They help where something else is the limit:

| Setting | What it does | When it helps |
|---|---|---|
| `worker.py --jobs N` (`ANON_JOBS`) | N recordings at once; the CPUs are shared between them (`--workers` = CPUs / N) | Fills the GPU's idle gaps (start-up and model load, download, upload, encoding tails), and fast GPUs where the CPU encoders are the limit. Costs about 2.5 GiB of RAM per 1080p WebM and some VRAM per job, so try 2 and watch `free -m` |
| `--extra '--batch 16'` (`ANON_EXTRA`) | bigger GPU batches | More VRAM available (the default 8 is sized for 6 GB) |
| `--extra '--workers 4'` | CPU threads per job for decode and encode | Overrides the automatic split |

Measure before keeping a change: while a video runs, `nvidia-smi dmon -s u` shows GPU use (a value that stays
near 100 means the GPU is the limit) and `htop` shows the CPU. FP16 is deliberately not used: it was slower
and lost detections on GTX 16xx cards (see Performance).

## Docker

One `Dockerfile`, two Compose profiles. The model goes in `models/` first (see [Install](#install)); it is
mounted, not baked into the image. The worker joins the field-capture app's Docker network, so start the app
stack first.

```bash
# GPU: Linux with an NVIDIA GPU, the NVIDIA driver and nvidia-container-toolkit installed
ANONYMIZER_TOKEN=<token> docker compose --profile gpu up -d --build

# CPU: no GPU, much slower (the only option on macOS, where Docker cannot reach the GPU)
ANONYMIZER_TOKEN=<token> docker compose --profile cpu up -d --build
```

The GPU profile starts the worker with `--device cuda`, so it stops with a clear error instead of silently
falling back to the CPU if the container cannot see the GPU. Check with
`docker compose --profile gpu run --rm --entrypoint nvidia-smi worker-gpu`. The image uses the standard PyTorch
wheels, which bundle CUDA (the CUDA 13.0 build by default, the one this tool was tested with); `nvidia-smi` must show a
CUDA version of 13.0 or higher. For an older host driver build with a matching index, for example
`TORCH_INDEX=https://download.pytorch.org/whl/cu126 docker compose --profile gpu up -d --build`.

## How it works

1. **Detection.** The model was trained on 1280 px tiles downscaled to 640. Each image therefore
   gets a full-image pass plus overlapping 1280 px tiles (15% overlap) when it is larger than
   the tile size. All views of one image are batched on the GPU. Per-class thresholds apply.
2. **Overlaps.** Every raw box is anonymized, so duplicates and partial boxes from tile edges
   are all covered. Reported counts merge duplicates.
3. **Output.** A fresh image is built from the pixel array, so no EXIF, GPS, ICC, XMP or PNG
   text chunks are carried over. EXIF orientation is applied to the pixels first. The image is
   then re-encoded (JPEG q95, WebP q95, PNG lossless) to a temp file. The temp file is checked
   for size, absence of metadata, and anonymized regions that match what was written. Only then
   is it renamed into place.
4. **Fail-safe.** If anything fails for an image, nothing is exported for it. The error is
   reported and the batch continues. The exit code is 1 if any image failed.
5. **Formats.** `.jpg/.jpeg/.png/.webp` are supported, plus the video formats listed under
   [Video](#video). Other files are listed as `UNSUPPORTED`.
   Animated WebP/APNG are rejected. For camera MPO-JPEGs only the main image is kept, so the
   embedded previews are dropped.

## Video

| Input | Output | Video codec | Audio codec (`--keep-audio`) |
|---|---|---|---|
| `.mp4 .m4v .mov .mkv .ts .mts .m2ts` | same | H.264 (CRF 18) | AAC |
| `.webm` | `.webm` | VP9 (CRF 30) | Opus |
| `.avi` | `.avi` | H.264 (CRF 18) | MP3 |
| `.wmv .flv .mpg .mpeg .3gp .ogv` | `.mp4` | H.264 (CRF 18) | AAC |

- **Every frame is detected,** using the same full-image and tile passes as images.
- **Temporal window.** A detector that misses a face for one frame would otherwise flash it
  unmasked. Each frame is therefore also masked with the boxes from `--temporal-window` frames
  either side, and overlapping boxes are merged into one region. This fills short gaps only: a
  face the model misses for longer than the window stays visible. A larger window holds a few
  more frames in RAM.
- **Audio is dropped by default,** because voices and background speech can identify people.
  `--keep-audio` keeps the first track and does **not** anonymize it.
- **What is removed.** Output contains only the new video stream and, if requested, the new audio
  stream. Container and stream tags (GPS, device, title, dates), subtitles, chapters, attachments,
  data tracks and extra audio tracks are not carried over.
- **Timing and rotation.** Original frame timestamps are kept, so variable frame rate footage stays
  in sync. The rotation flag from phones is applied to the pixels before detection.
- **Fail-safe.** Encoding goes to a temp file. Before it is renamed into place, the file is decoded
  to check its streams, tags, size and frame count, and 1 in 30 masked frames is compared with
  what was written. Any failure, including corrupt source data, means nothing is exported. A source
  cut off cleanly part-way can still decode: the frames that decode are exported, with a `WARNING`
  if the container says there should be more. This confirms the file matches the frames that were masked. It cannot find faces the model missed.
- **Name clashes.** `clip.wmv` and `clip.3gp` both map to `clip.mp4`. The second one is reported
  as `FAILED` rather than overwriting the first.

**Pipeline.** Decoding runs on its own thread, the main thread runs GPU detection in batches of
`--batch` frames, and encoder threads mask and encode. H.264 uses one encoder with `--workers`
threads. VP9 runs `--workers / 2` encoders with 2 threads each, on chunks of up to 120 frames that
are joined in order. Chunks are limited to about 400 MiB each, so 1080p uses about 67 frames.

**Speed** on the GTX 1660 Ti with 6 CPUs: about 11.4 fps on a 1080×1920 street recording (3 model
views per frame), which is about 2.6× slower than real time for 30 fps footage. One hour of it
would take about 2.6 hours. `--tile 0` is faster but misses small faces and plates.

**Memory.** A 1080p WebM can use about 2.5 GiB of RAM on top of what is already in use. Higher
`--workers` values use more, because each VP9 encoder can hold a chunk.

## Performance

The image path runs as a pipeline (videos run one at a time after the images). Loader threads decode the next images while the main thread runs
GPU detection. `--workers` threads (default: number of usable CPUs, respecting `taskset`/container limits) anonymize, encode and verify. One
model instance is used only from the main thread. Results print in input order, and at most
about `workers + 3` images are held in RAM at once.

Measured on a GTX 1660 Ti (6 GB) with 4 CPU cores. Benchmark: 30 images from 0.7 to 20 MP,
gaussian mode, model already loaded.

| Configuration | Time / 30 images | Per image | GPU util (mean) |
|---|---|---|---|
| Sequential (original loop) | 19.4 s | 646 ms | 24% |
| `--workers 1` | 17.8 s | 590 ms | 20% |
| `--workers 4` | **10.0–11.2 s** | **~350 ms** | 28% |

After the move to 6 CPUs, worker count was re-tested: 30 copies of the `images/` PNGs, and 5
20 MP JPEGs. Times are per image, the median of 2 runs.

| `--workers` | 1 | 2 | 4 | **6 (= CPUs, default)** | 8 |
|---|---|---|---|---|---|
| PNG set (0.9 MP) | 470 ms | 280 ms | 190 ms | **170 ms** | 170 ms |
| 20 MP JPEG set | 1580 ms | 1320 ms | 1320–1370 ms | **1340 ms** | 1350–1370 ms |

Matching the CPU count is fastest for small images and costs nothing for large ones. Large
images are GPU-bound, so extra workers do not help them. Going above the CPU count gains nothing.

Per-image cost, sequential, in ms (4-CPU run):

| Image | Model views | Load | Detect (GPU) | Anonymize | Save + verify |
|---|---|---|---|---|---|
| 1024×740 JPEG | 1 | 13 | 23 | 6 | 15 |
| 1920×1440 JPEG | 5 | 30 | 141 | 4 | 42 |
| 1505×842 PNG | 3 | 45 | 86 | 4 | 485 |
| 5568×3712 JPEG (20 MP) | 21 | 284 | 594 | 532 | 397 |

- **Detection is GPU compute-bound:** about 25–28 ms per 640×640 view in FP32. Large photos cost
  more because they are tiled (a 20 MP image means 21 views). `--tile 0` is much faster but
  misses small faces and plates.
- **The low GPU-utilization average is expected:** the GPU runs flat out during detection and idles
  between images while the 4-core CPU decodes and encodes. More CPU cores raise throughput.
- **Options tested and rejected:**
  - **FP16 (`half`)** is 3.6× slower on GTX 16xx cards and silently lost every detection on 3 of
    5 test images.
  - **`cudnn.benchmark`** is 14% faster per view, but it re-tunes for every new input shape, which
    made mixed-size batches 70% slower and used 4.9 GB of VRAM.
- **PNG and WebP encoding is the main CPU cost.** PNG `compress_level=1` would be about 3× faster
  to encode with files about 10% larger, but overall that saves only about 6%, so the default
  stays at level 6.

## Limitations

- **Detection can miss faces and plates.** On the model's test split, recall is 0.61 for
  faces and 0.70 for plates. Review outputs before publishing, and consider lowering
  `--face-conf` / `--plate-conf` (more false positives, fewer misses).
- Faces that are tiny, side-on, heavily occluded or very large and close-up are the most
  likely to be missed.
- The very first GPU run after installing or upgrading torch or the driver can take about a
  minute, because this torch build (CUDA 13) JIT-compiles kernels for the GTX 1660 Ti. The
  result is cached, so later runs start in seconds (~0.1–2 s per image).
- Ultralytics' runtime auto-install is disabled (`YOLO_AUTOINSTALL=false`), so the tool never
  modifies the Python environment.
- The model is not rotation-invariant. Images whose content is rotated (for example a portrait
  photo saved sideways with no EXIF orientation) may get fewer detections.
- Output colour profiles (ICC) are stripped, so wide-gamut images may shift slightly in colour.
  16-bit PNGs are written as 8-bit.
- Video frames are detected independently and the model has no tracker, so recall on low-quality
  or heavily compressed footage can be much lower. Review video outputs before publishing.
- Video output is always 8-bit 4:2:0 BT.709. HDR or 10-bit sources lose their range and can look
  washed out. Odd frame sizes lose one pixel row or column, and alpha in WebM is dropped.
- Videos whose resolution changes mid-stream are rejected.
- Model license: AGPL-3.0.

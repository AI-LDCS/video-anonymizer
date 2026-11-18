"""Tests for sample_frames.py and tag_gps.py. Needs opencv, numpy and pillow (the same as the tools).

    python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sample_frames as sf  # noqa: E402
import tag_gps  # noqa: E402

BASE_MS = 1_700_000_000_000


def make_video(path: Path, secs: int, fps=10):
    """A moving texture; every 7th frame is blurred, so there is always a sharper frame to choose."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for i in range(secs * fps):
        im = np.roll(base, i * 2, axis=1)
        w.write(cv2.GaussianBlur(im, (0, 0), 4) if i % 7 == 0 else im)
    w.release()


def walk(t0, t1, lat_off=0.0, speed=1.2):
    """GPS points once a second, walking north at `speed` m/s; video time restarts at 0 for each recording."""
    return [
        {"videoTimeS": float(s - t0), "timestampMs": BASE_MS + s * 1000, "accuracyM": 3.0,
         "latitude": 13.0 + s * speed / 111320 + lat_off, "longitude": 100.0}
        for s in range(t0, t1 + 1)
    ]


class SamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        make_video(cls.dir / "a.avi", 60)
        make_video(cls.dir / "b1.avi", 30)
        make_video(cls.dir / "b2.avi", 30)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_sample(self, video, points, name, **kw):
        return sf.sample(self.dir / video, points, self.dir / name, prefix=name, log=lambda *_: None, **kw)

    def test_one_frame_per_step_with_position_and_heading(self):
        frames, state = self.run_sample("a.avi", walk(0, 60), "single")
        self.assertGreater(len(frames), 20)
        self.assertEqual([f["frame"] for f in frames], list(range(len(frames))))
        self.assertTrue(all((self.dir / "single" / f["file"]).is_file() for f in frames))
        self.assertAlmostEqual(frames[3]["headingDeg"], 0.0, delta=1.0)  # walking north
        self.assertAlmostEqual(state["cumM"], 72.0, delta=1.5)  # 60 s at 1.2 m/s, no shrinking at the ends
        # one bin every 3 m
        steps = [b - a for a, b in zip([f["walkedStartM"] for f in frames], [f["walkedStartM"] for f in frames][1:])]
        self.assertTrue(all(abs(s - 3.0) < 0.01 for s in steps[1:-1]), steps)

    def test_picks_a_sharp_frame_not_a_blurred_one(self):
        frames, _ = self.run_sample("a.avi", walk(0, 60), "sharp")
        scores = [f["sharpness"] for f in frames]
        self.assertLess(max(scores) / min(scores), 1.5, "the blurred frames were not chosen")

    def test_two_recordings_carry_on_like_one(self):
        whole, _ = self.run_sample("a.avi", walk(0, 60), "whole")
        f1, s1 = self.run_sample("b1.avi", walk(0, 30), "r1")
        f2, s2 = self.run_sample("b2.avi", walk(30, 60), "r2", state_in=s1)
        self.assertEqual([f["bin"] for f in f1 + f2], [f["bin"] for f in whole])
        self.assertEqual([f["frame"] for f in f1 + f2], list(range(len(whole))), "frame numbers carry on")
        self.assertEqual(s2["nextFrame"], len(whole))
        self.assertAlmostEqual(s2["cumM"], 72.0, delta=1.5)
        self.assertFalse(any(f["afterGap"] for f in f1 + f2))

    def test_a_big_jump_between_recordings_is_flagged_but_the_count_carries_on(self):
        f1, s1 = self.run_sample("b1.avi", walk(0, 30), "g1")
        f2, s2 = self.run_sample("b2.avi", walk(30, 60, lat_off=100 / 111320), "g2", state_in=s1)
        self.assertTrue(f2[0]["afterGap"])
        self.assertAlmostEqual(f2[0]["gapM"], 100.0, delta=5.0)
        self.assertEqual(f2[0]["frame"], s1["nextFrame"], "frame numbers carry on")
        self.assertFalse(any(f["afterGap"] for f in f2[1:]))
        # the 100 m jump is not counted as walking: the counter ends where it would without the jump
        _, whole = self.run_sample("a.avi", walk(0, 60), "g0")
        self.assertAlmostEqual(s2["cumM"], whole["cumM"], delta=1.5)
        self.assertEqual(f2[0]["bin"], f1[-1]["bin"] + 1)

    def test_standing_still_adds_no_frames(self):
        still = [{**p, "latitude": 13.0} for p in walk(0, 30)]
        with self.assertRaises(ValueError):
            self.run_sample("b1.avi", still, "still")

    def test_too_few_good_points_is_an_error(self):
        bad = [{**p, "accuracyM": 50.0} for p in walk(0, 30)]
        with self.assertRaises(ValueError):
            self.run_sample("b1.avi", bad, "bad")


class TagTest(unittest.TestCase):
    def test_writes_only_gps_and_ids_and_keeps_the_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"
            pixels = np.random.default_rng(2).integers(0, 255, (60, 80, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(p)
            frame = {"frame": 7, "latitude": 13.7270722, "longitude": 100.7781985, "accuracyM": 2.5,
                     "headingDeg": 191.4, "timestamp": "2026-11-03T10:10:58.287Z", "file": "f.png"}
            ctx = {"routeId": "r", "routeVersionId": "v", "assignmentId": "a", "recordingId": "c",
                   "recordingIndex": 2, "purpose": "inference", "stepM": 3}
            self.assertEqual(tag_gps.tag_folder(Path(d), [frame], ctx, log=lambda *_: None), 1)
            with Image.open(p) as im:
                self.assertTrue(np.array_equal(np.asarray(im), pixels))
                ex = im.getexif()
                self.assertEqual(set(ex.keys()), {0x010E, 0x8825})
                gps = ex.get_ifd(0x8825)
                self.assertEqual(gps[1], "N")
                self.assertEqual(gps[3], "E")
                deg, mn, sec = (float(x) for x in gps[2])
                self.assertAlmostEqual(deg + mn / 60 + sec / 3600, 13.7270722, places=6)
                self.assertAlmostEqual(float(gps[17]), 191.4, places=1)
                self.assertEqual(json.loads(ex[0x010E]), {**ctx, "frame": 7})
            tag_gps.check_only_allowed(p)

    def test_no_heading_means_no_direction_tag(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"
            Image.new("RGB", (8, 8)).save(p)
            frame = {"frame": 0, "latitude": -33.5, "longitude": -70.6, "accuracyM": 4.0, "headingDeg": None,
                     "timestamp": "2026-11-03T10:10:58Z", "file": "f.png"}
            tag_gps.tag_folder(Path(d), [frame], {}, log=lambda *_: None)
            with Image.open(p) as im:
                gps = im.getexif().get_ifd(0x8825)
                self.assertNotIn(17, gps)
                self.assertEqual((gps[1], gps[3]), ("S", "W"))

    def test_refuses_a_file_with_other_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"
            ex = Image.Exif()
            ex[0x010F] = "SomeCameraMaker"  # Make: device information must never be there
            Image.new("RGB", (8, 8)).save(p, exif=ex)
            with self.assertRaises(RuntimeError):
                tag_gps.check_only_allowed(p)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Anonymizer worker: pulls finished recordings from the field-capture app, hides faces and
licence plates with anonymize.py, and uploads the result.

It only needs HTTPS access to the app and the shared bearer token (ANONYMIZER_TOKEN); it never
touches the database or object storage. Standard library only.

    export ANONYMIZER_TOKEN=...            # same value as the server's .env
    python worker.py --server https://host.tailnet.ts.net

One recording at a time: claim -> download the original -> anonymize.py -> upload -> next.
The claim is a lease the worker keeps alive with a heartbeat; if this process dies, the server
gives the recording to the next claim. Several workers can run at once.
"""
import argparse
import collections
import hashlib
import http.client
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
BLOCK = 1 << 20
STOP = threading.Event()  # set on Ctrl+C / SIGTERM: running jobs are stopped and handed back to the queue


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body}")
        self.status, self.body = status, body


class Api:
    """The few calls the server offers under /api/anonymizer. A fresh connection per call keeps it simple."""

    def __init__(self, base, token, timeout, allow_http=False):
        u = urlsplit(base)
        if u.scheme not in ("http", "https") or not u.hostname:
            sys.exit(f"error: --server must look like https://host, got {base!r}")
        if u.scheme == "http" and not allow_http and u.hostname not in ("localhost", "127.0.0.1", "::1"):
            sys.exit("error: refusing to send the token over plain http to a remote host; use https "
                     "(or --allow-http inside a private Docker network)")
        self.scheme, self.host, self.port, self.prefix = u.scheme, u.hostname, u.port, u.path.rstrip("/")
        self.token, self.timeout = token, timeout

    def _conn(self):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def _headers(self, lease=None, extra=None):
        h = {"Authorization": f"Bearer {self.token}"}
        if lease:
            h["x-lease-id"] = lease
        h.update(extra or {})
        return h

    @staticmethod
    def _fail(resp, raw):
        try:
            detail = json.loads(raw).get("message") or raw.decode("utf8", "replace")
        except Exception:
            detail = raw.decode("utf8", "replace")[:300]
        raise ApiError(resp.status, detail)

    def call(self, method, path, lease=None, body=None):
        """A small JSON request. Returns the parsed body, or None for 204."""
        conn = self._conn()
        try:
            data = json.dumps(body).encode() if body is not None else None
            extra = {"Content-Type": "application/json"} if data is not None else {}
            conn.request(method, self.prefix + path, body=data, headers=self._headers(lease, extra))
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status == 204:
                return None
            if resp.status >= 300:
                self._fail(resp, raw)
            return json.loads(raw) if raw else {}
        finally:
            conn.close()

    def claim(self):
        return self.call("POST", "/api/anonymizer/claim", body={})

    def heartbeat(self, rid, lease):
        return self.call("POST", f"/api/anonymizer/{rid}/heartbeat", lease=lease, body={})

    def fail(self, rid, lease, message):
        return self.call("POST", f"/api/anonymizer/{rid}/fail", lease=lease, body={"error": message[:500]})

    def download(self, rid, lease, dest: Path):
        """Streams the original into `dest`. Returns (bytes, sha256)."""
        conn = self._conn()
        try:
            conn.request("GET", f"{self.prefix}/api/anonymizer/{rid}/source", headers=self._headers(lease))
            resp = conn.getresponse()
            if resp.status != 200:
                self._fail(resp, resp.read())
            h, n = hashlib.sha256(), 0
            with open(dest, "wb") as f:
                while block := resp.read(BLOCK):
                    f.write(block)
                    h.update(block)
                    n += len(block)
            return n, h.hexdigest()
        finally:
            conn.close()

    def upload(self, rid, lease, src: Path, sha256, mode):
        """Streams the anonymized file to the server. Never reads it all into memory."""
        size = src.stat().st_size
        conn = self._conn()
        try:
            conn.putrequest("PUT", f"{self.prefix}/api/anonymizer/{rid}/result")
            for k, v in self._headers(lease, {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
                "x-sha256": sha256,
                "x-anon-mode": mode,
            }).items():
                conn.putheader(k, v)
            conn.endheaders()
            try:
                with open(src, "rb") as f:
                    while block := f.read(BLOCK):
                        conn.send(block)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the server answered early (for example a lost lease); its reply says why
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 300:
                self._fail(resp, raw)
        finally:
            conn.close()


def sha256_of(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(BLOCK):
            h.update(block)
    return h.hexdigest()


class Heartbeat(threading.Thread):
    """Keeps the lease alive while anonymize.py runs. If the server says the lease is gone, stops the job."""

    def __init__(self, api, rid, lease, every, on_lost):
        super().__init__(daemon=True)
        self.api, self.rid, self.lease, self.every, self.on_lost = api, rid, lease, every, on_lost
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.wait(self.every):
            try:
                self.api.heartbeat(self.rid, self.lease)
            except ApiError as e:
                if e.status in (404, 409):
                    log(f"lease lost for {self.rid}: {e}")
                    self.on_lost()
                    return
                log(f"heartbeat failed (will retry): {e}")
            except Exception as e:  # network trouble: the lease is long, so try again next time
                log(f"heartbeat failed (will retry): {e}")


def build_command(args, src: Path, out_dir: Path, out_file: Path):
    if args.anonymize_cmd:  # a stand-in for tests: placeholders {input} {out_dir} {out_file}
        return [t.format(input=src, out_dir=out_dir, out_file=out_file) for t in shlex.split(args.anonymize_cmd)]
    extra = shlex.split(args.extra)
    cmd = [sys.executable, str(HERE / "anonymize.py"), "--input", str(src), "--output", str(out_dir),
           "--mode", args.mode, "--device", args.device]
    if "--workers" not in extra:  # share the CPUs between the jobs running at once, unless told otherwise
        cmd += ["--workers", str(args.job_workers)]
    return cmd + extra


def run_anonymizer(cmd, lost: threading.Event):
    """Runs the command, echoing its output. Returns (exit code, last lines)."""
    tail = collections.deque(maxlen=12)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def watch():  # a lost lease (someone else has the job now) or a shutdown: stop wasting the GPU
        while proc.poll() is None:
            lost.wait(2)
            if lost.is_set() or STOP.is_set():
                proc.terminate()
                return

    threading.Thread(target=watch, daemon=True).start()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            print("   |", line, flush=True)
        return proc.wait(), list(tail)
    except BaseException:
        proc.kill()
        raise


def process(api, args, job):
    rid, lease = job["id"], job["leaseId"]
    log(f"claimed {rid} (attempt {job['attempt']}, {job['bytes']:,} bytes)")
    work = Path(tempfile.mkdtemp(prefix=f"anon-{rid[:8]}-", dir=args.work_dir))
    lost = threading.Event()
    beat = Heartbeat(api, rid, lease, max(5, job["leaseSeconds"] / 3), lost.set)
    beat.start()
    try:
        src_dir, out_dir = work / "in", work / "out"
        src_dir.mkdir()
        out_dir.mkdir()
        src = src_dir / f"{rid}.{job['ext']}"
        out_file = out_dir / src.name

        n, digest = api.download(rid, lease, src)
        if n != job["bytes"] or digest != job["sha256"]:
            raise RuntimeError(f"download does not match the server's checksum ({n} bytes)")
        log(f"downloaded {n:,} bytes, running {'stand-in' if args.anonymize_cmd else 'anonymize.py'}")

        t0 = time.time()
        code, tail = run_anonymizer(build_command(args, src, out_dir, out_file), lost)
        if lost.is_set() or STOP.is_set():
            raise RuntimeError("the claim was lost while anonymizing" if lost.is_set() else "the worker was stopped")
        if code != 0:
            raise RuntimeError(f"anonymizer exited with {code}: " + " / ".join(tail[-4:]))
        if not out_file.is_file() or out_file.stat().st_size == 0:
            raise RuntimeError("the anonymizer reported success but wrote no output file")
        log(f"anonymized in {time.time() - t0:.0f}s, uploading {out_file.stat().st_size:,} bytes")

        api.upload(rid, lease, out_file, sha256_of(out_file), args.mode)
        log(f"done {rid}")
    except BaseException as e:
        # SystemExit/KeyboardInterrupt too: give the job back at once instead of making the server wait for the lease
        msg = f"{type(e).__name__}: {e}"
        log(f"FAILED {rid}: {msg}")
        try:
            r = api.fail(rid, lease, msg)
            log(f"reported to the server (will retry: {r.get('retry')})")
        except Exception as e2:
            log(f"could not report the failure ({e2}); the server will take the claim back when its lease runs out")
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        beat.stop_event.set()
        shutil.rmtree(work, ignore_errors=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=os.environ.get("FIELD_APP_URL"), help="app URL, e.g. https://host.tailnet.ts.net (or FIELD_APP_URL)")
    p.add_argument("--mode", choices=["solid", "gaussian", "pixelate"], default="solid", help="how to hide detections (default solid: irreversible)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--extra", default="", help="more options for anonymize.py, as one quoted string, e.g. '--padding 0.3'")
    p.add_argument("--poll", type=float, default=30, help="seconds to wait when nothing is queued (default 30)")
    p.add_argument("--jobs", type=int, default=1,
                   help="recordings processed at the same time (default 1). The CPUs are shared between them; "
                        "more jobs keep the GPU busy while another job is decoding, encoding or transferring")
    p.add_argument("--once", action="store_true", help="work until the queue is empty, then exit")
    p.add_argument("--work-dir", default=None, help="where to keep temporary files (default: the system temp dir)")
    p.add_argument("--allow-http", action="store_true", help="allow plain http to a non-local host (only for a private Docker network)")
    p.add_argument("--timeout", type=float, default=1800, help="network timeout in seconds (default 1800)")
    p.add_argument("--anonymize-cmd", default=None, help="testing only: replace anonymize.py with this command; placeholders {input} {out_dir} {out_file}")
    a = p.parse_args()
    if not a.server:
        p.error("--server (or FIELD_APP_URL) is required")
    if a.jobs < 1:
        p.error("--jobs must be at least 1")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    a.job_workers = max(1, cpus // a.jobs)
    return a


def work_loop(api, args):
    """Claim, process, repeat. Returns 0, or 1 if --once could not reach the server."""
    idle = 0
    while not STOP.is_set():
        try:
            job = api.claim()
        except Exception as e:
            log(f"claim failed: {e}")
            if args.once:
                return 1
            STOP.wait(args.poll)
            continue
        if job is None:
            if args.once:
                return 0
            if idle == 0:
                log("nothing waiting; polling")
            idle += 1
            STOP.wait(args.poll)
            continue
        idle = 0
        process(api, args, job)
    return 0


def main():
    args = parse_args()
    token = os.environ.get("ANONYMIZER_TOKEN", "")
    if len(token) < 32:
        sys.exit("error: set ANONYMIZER_TOKEN to the server's token (at least 32 characters)")
    if args.work_dir:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    api = Api(args.server, token, args.timeout, args.allow_http)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    log(f"worker started: server {args.server}, mode {args.mode}, {args.jobs} job(s) at a time, {args.job_workers} CPU thread(s) each")
    codes = []

    def run():
        try:
            codes.append(work_loop(api, args))
        except BaseException as e:  # a crash in one slot must not leave the others unaware
            log(f"worker slot stopped: {type(e).__name__}: {e}")
            codes.append(1)

    try:
        threads = [threading.Thread(target=run, name=f"slot-{i}") for i in range(args.jobs)]
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(0.5)  # short joins keep the main thread able to receive Ctrl+C / SIGTERM
    except KeyboardInterrupt:
        STOP.set()
        log("stopping: running jobs are being handed back")
        for t in threads:
            t.join()
        log("stopped")
        return 130
    if args.once:
        log("queue is empty")
    return max(codes or [0])


if __name__ == "__main__":
    sys.exit(main())

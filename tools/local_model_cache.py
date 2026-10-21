"""Prefetch verified model shards into a bounded local cache.

Builds a symlink view with a source fallback. Use --bind-target only when the
loader must keep using an existing directory; otherwise load from --view.
The manifest maps relative file names to {"bytes": int, "sha256": str}.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ARGS = argparse.ArgumentParser(description=__doc__)
ARGS.add_argument("--source", type=Path, required=True)
ARGS.add_argument("--cache", type=Path, required=True)
ARGS.add_argument("--view", type=Path, required=True)
ARGS.add_argument("--manifest", type=Path, required=True)
ARGS.add_argument("--bind-target", type=Path)
ARGS.add_argument("--start-index", type=int, default=0)
ARGS.add_argument("--file-workers", type=int, default=4)
ARGS.add_argument("--range-workers", type=int, default=16)
ARGS.add_argument("--limit-gib", type=float, default=200)
ARGS.add_argument("--reserve-gib", type=float, default=64)
ARGS.add_argument("--transport", choices=["nas", "http"], default="nas")
ARGS.add_argument("--modelscope-model")
ARGS.add_argument("--revision", help="Pinned ModelScope commit for HTTP transport")
ARGS = ARGS.parse_args()
if not (
    1 <= ARGS.file_workers <= 16
    and 1 <= ARGS.range_workers <= 64
    and ARGS.start_index >= 0
    and ARGS.limit_gib > 0
    and ARGS.reserve_gib >= 0
):
    ARGS.error("Invalid worker count, start index, or disk budget")
if ARGS.transport == "http" and not (ARGS.modelscope_model and ARGS.revision):
    ARGS.error("HTTP transport requires --modelscope-model and --revision")
SOURCE = ARGS.source.resolve(strict=True)
CACHE = ARGS.cache.resolve()
VIEW = ARGS.view.resolve()
TARGET = ARGS.bind_target.resolve() if ARGS.bind_target else VIEW
if ARGS.bind_target and (
    TARGET == SOURCE or SOURCE.is_relative_to(TARGET) or TARGET.is_relative_to(SOURCE)
):
    ARGS.error(
        "--bind-target requires a separate, stable source path outside the mount"
    )
if len({SOURCE, CACHE, VIEW}) != 3 or any(
    x.is_relative_to(y) or y.is_relative_to(x)
    for x, y in ((SOURCE, CACHE), (SOURCE, VIEW), (CACHE, VIEW))
):
    ARGS.error("Source, cache, and view must be separate, non-nested directories")
MANIFEST = json.loads(ARGS.manifest.read_text())
for name, entry in MANIFEST.items():
    if Path(name).is_absolute() or ".." in Path(name).parts or Path(name).name != name:
        ARGS.error("Manifest entries must be simple file names")
    if (
        not isinstance(entry.get("bytes"), int)
        or entry["bytes"] < 0
        or not re.fullmatch("[0-9a-f]{64}", entry.get("sha256", ""))
    ):
        ARGS.error("Invalid manifest entry: " + name)
LIMIT = int(ARGS.limit_gib * 1024**3)
MARGIN = int(ARGS.reserve_gib * 1024**3)
CHUNK = 16 * 1024**2
RUNTIME = CACHE / ".state"
RUNTIME.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()
START = time.monotonic()
STATE = {
    "started": datetime.datetime.now().astimezone().isoformat(),
    "pid": os.getpid(),
    "source": str(SOURCE),
    "view": str(VIEW),
    "target": str(TARGET),
    "limit_bytes": LIMIT,
    "reserve_bytes": MARGIN,
    "range_workers_per_file": ARGS.range_workers,
    "file_workers": ARGS.file_workers,
    "transport": ARGS.transport,
    "verified": {},
    "events": [],
}
CACHE.mkdir(parents=True, exist_ok=True)
VIEW.mkdir(parents=True, exist_ok=True)
lockfd = open(RUNTIME / "local-cache.lock", "w")
fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def emit(event, **kw):
    with LOCK:
        row = {
            "time": datetime.datetime.now().astimezone().isoformat(),
            "event": event,
            **kw,
        }
        STATE["events"].append(row)
        STATE["events"] = STATE["events"][-400:]
        STATE["elapsed_seconds"] = time.monotonic() - START
        STATE["free_bytes"] = os.statvfs(CACHE).f_bavail * os.statvfs(CACHE).f_frsize
        STATE["cached_bytes"] = sum(
            p.stat().st_size for p in CACHE.glob("*.safetensors")
        )
        tmp = RUNTIME / "local-cache-status.tmp"
        tmp.write_text(json.dumps(STATE, indent=2))
        tmp.replace(RUNTIME / "local-cache-status.json")
        if event == "prefetch_finished":
            tmp = RUNTIME / "local-cache-verified.tmp"
            tmp.write_text(json.dumps(STATE["verified"], indent=2))
            tmp.replace(RUNTIME / "local-cache-verified.json")
        print(json.dumps(row), flush=True)


def link(name, target):
    p = VIEW / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".link-tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb", buffering=0) as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def verify(name):
    p = CACHE / name
    expected = MANIFEST[name]
    assert p.stat().st_size == expected["bytes"], name
    assert sha(p) == expected["sha256"], "SHA256 mismatch " + name
    with LOCK:
        STATE["verified"][name] = {
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
            "mtime_ns": p.stat().st_mtime_ns,
            "ctime_ns": p.stat().st_ctime_ns,
        }
    link(name, p)
    emit("local_shard_verified", name=name, bytes=expected["bytes"])


def evict(incoming, protected):
    files = sorted(
        (
            p
            for p in CACHE.glob("*.safetensors")
            if p.name in MANIFEST and p.name not in protected
        ),
        key=lambda p: p.stat().st_mtime_ns,
    )
    total = sum(p.stat().st_size for p in CACHE.glob("*.safetensors"))
    while files and (
        total + incoming > LIMIT
        or os.statvfs(CACHE).f_bavail * os.statvfs(CACHE).f_frsize < MARGIN + incoming
    ):
        p = files.pop(0)
        size = p.stat().st_size
        link(p.name, SOURCE / p.name)
        p.unlink()
        total -= size
        STATE["verified"].pop(p.name, None)
        emit("evicted_to_source_fallback", name=p.name)
    if (
        total + incoming > LIMIT
        or os.statvfs(CACHE).f_bavail * os.statvfs(CACHE).f_frsize < MARGIN + incoming
    ):
        raise RuntimeError(
            "Batch exceeds cache budget or reserved free-space margin; reduce --file-workers"
        )


def copy(name):
    size = MANIFEST[name]["bytes"]
    p = CACHE / name
    tmp = p.with_suffix(p.suffix + ".range-partial")
    if p.exists():
        verify(name)
        return
    free = os.statvfs(CACHE).f_bavail * os.statvfs(CACHE).f_frsize
    if free < size + MARGIN:
        raise RuntimeError("Local free space reached reserved margin")
    begin = time.monotonic()
    dst = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(dst, size)
        if ARGS.transport == "http":
            url = (
                "https://modelscope.cn/api/v1/models/"
                + quote(ARGS.modelscope_model, safe="/")
                + "/repo?Revision="
                + quote(ARGS.revision, safe="")
                + "&FilePath="
                + quote(name, safe="")
            )
            with urlopen(
                Request(url, headers={"Range": "bytes=0-1023"}), timeout=30
            ) as response:
                assert response.status == 206
                final = response.geturl()
                response.read(1024)

            def one(offset):
                length = min(64 * 1024**2, size - offset)
                for attempt in range(3):
                    try:
                        req = Request(
                            final,
                            headers={"Range": f"bytes={offset}-{offset+length-1}"},
                        )
                        with urlopen(req, timeout=60) as response:
                            assert response.status == 206 and response.headers.get(
                                "Content-Range", ""
                            ).startswith(f"bytes {offset}-")
                            got = 0
                            while b := response.read(4 * 1024**2):
                                if got + len(b) > length:
                                    raise RuntimeError(
                                        "HTTP range exceeded requested length"
                                    )
                                written = 0
                                while written < len(b):
                                    written += os.pwrite(
                                        dst, b[written:], offset + got + written
                                    )
                                got += len(b)
                            assert got == length, (name, offset, got, length)
                        return
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(0.5 * (attempt + 1))

            with ThreadPoolExecutor(max_workers=ARGS.range_workers) as pool:
                for _ in pool.map(one, range(0, size, 64 * 1024**2)):
                    pass
        else:
            src = os.open(SOURCE / name, os.O_RDONLY)
            try:

                def one(offset):
                    length = min(CHUNK, size - offset)
                    parts = []
                    got = 0
                    while got < length:
                        b = os.pread(src, length - got, offset + got)
                        if not b:
                            raise RuntimeError("Short NAS read " + name)
                        parts.append(b)
                        got += len(b)
                    b = b"".join(parts)
                    written = 0
                    while written < len(b):
                        written += os.pwrite(dst, b[written:], offset + written)

                with ThreadPoolExecutor(max_workers=ARGS.range_workers) as pool:
                    for _ in pool.map(one, range(0, size, CHUNK)):
                        pass
            finally:
                os.close(src)
        os.fsync(dst)
    finally:
        os.close(dst)
    assert tmp.stat().st_size == size
    digest = sha(tmp)
    assert digest == MANIFEST[name]["sha256"], "SHA256 mismatch " + name
    tmp.replace(p)
    with LOCK:
        STATE["verified"][name] = {
            "bytes": size,
            "sha256": digest,
            "mtime_ns": p.stat().st_mtime_ns,
            "ctime_ns": p.stat().st_ctime_ns,
        }
    link(name, p)
    emit(
        "cached",
        name=name,
        bytes=size,
        seconds=time.monotonic() - begin,
        mib_s=size / 1024**2 / (time.monotonic() - begin),
    )


# Every unpopulated entry has an equivalent NAS fallback. Only complete, SHA-verified
# files are atomically published. Existing open descriptors remain valid on eviction.
for name in MANIFEST:
    link(name, SOURCE / name)
existing = sorted(CACHE.glob("*.safetensors"))
receipt_path = RUNTIME / "local-cache-verified.json"
receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
for p in existing:
    previous = receipt.get(p.name)
    if p.name not in MANIFEST:
        raise RuntimeError("Cache contains an unmanaged shard: " + p.name)
    if (
        previous
        and p.stat().st_mtime_ns == previous["mtime_ns"]
        and p.stat().st_ctime_ns == previous.get("ctime_ns")
        and previous["sha256"] == MANIFEST[p.name]["sha256"]
        and p.stat().st_size == MANIFEST[p.name]["bytes"]
    ):
        STATE["verified"][p.name] = previous
        link(p.name, p)
    else:
        verify(p.name)
for item in SOURCE.iterdir():
    if item.is_file() and not item.name.endswith(".safetensors"):
        link(item.name, item)
if ARGS.bind_target:
    mount_target = subprocess.check_output(
        ["findmnt", "-n", "-T", str(TARGET), "-o", "TARGET"], text=True
    ).strip()
    if mount_target == str(TARGET):
        assert (
            TARGET.stat().st_ino == VIEW.stat().st_ino
            and TARGET.stat().st_dev == VIEW.stat().st_dev
        ), "An unrelated model mount already exists"
    else:
        subprocess.run(["mount", "--bind", str(VIEW), str(TARGET)], check=True)
emit("local_view_active", cached_shards=len(existing))
start_index = ARGS.start_index
names = sorted(n for n in MANIFEST if n.endswith(".safetensors"))[start_index:]
try:
    for offset in range(0, len(names), ARGS.file_workers):
        batch = names[offset : offset + ARGS.file_workers]
        missing = [n for n in batch if not (CACHE / n).exists()]
        evict(sum(MANIFEST[n]["bytes"] for n in missing), set(batch))
        with ThreadPoolExecutor(max_workers=ARGS.file_workers) as pool:
            for future in as_completed(
                [
                    pool.submit(copy, n)
                    for n in names[offset : offset + ARGS.file_workers]
                ]
            ):
                future.result()
    emit("prefetch_finished")
except Exception as error:
    for name in MANIFEST:
        (CACHE / (name + ".range-partial")).unlink(missing_ok=True)
    emit("prefetch_error", error_type=type(error).__name__)
    raise RuntimeError("Prefetch failed: " + type(error).__name__) from None

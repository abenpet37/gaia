"""HTTP range-request file-like wrapper so h5py can stream a remote HDF5
without downloading the entire file.

Usage::

    with RemoteHDF5File(url, session) as f:
        with h5py.File(f, "r") as hf:
            data = hf["some/dataset"][()]

Performance notes
-----------------
* The download URL is resolved once (ASF -> Earthdata Login -> cloud URL) so
  range requests go straight to the final host instead of re-walking the
  redirect chain on every block.
* Blocks are large (default 4 MiB) and sequential access triggers background
  read-ahead, so network latency overlaps with h5py decoding/copying.
* Every read returns exactly the bytes asked for, spanning blocks as needed;
  h5py treats short reads as corrupt data.
"""

from __future__ import annotations

import io
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional
from urllib.parse import urlparse

import requests

_MAX_CACHE_BYTES = 256 * 1024 * 1024


class RangeNotSupportedError(IOError):
    """The server answered a ranged GET with the full file (HTTP 200)."""


class RemoteHDF5File(io.RawIOBase):
    """A read-only, seekable, file-like object backed by HTTP range requests.

    Parameters
    ----------
    url:
        Full HTTPS URL of the HDF5 file.
    session:
        A :class:`requests.Session` with any required auth headers already set.
    chunk_cache_mb:
        Block size in MiB (default 4).
    timeout:
        Requests timeout tuple ``(connect, read)`` in seconds.
    progress_callback:
        Optional callable ``(bytes_fetched: int, total_fetched: int)`` invoked
        after each HTTP response is consumed.
    read_ahead:
        Number of following blocks to prefetch on sequential access (0 disables).
    """

    def __init__(
        self,
        url: str,
        session: requests.Session,
        chunk_cache_mb: int = 4,
        timeout: tuple[int, int] = (30, 180),
        progress_callback=None,
        read_ahead: int = 4,
    ) -> None:
        super().__init__()
        self._origin_url = url
        self._url = url
        self._strip_auth = False
        self._session = session
        self._timeout = timeout
        self._progress_callback = progress_callback
        self._pos = 0
        self._size: Optional[int] = None
        self._lock = threading.Lock()
        self._cache: dict[int, bytes] = {}          # block_index -> bytes (insertion ordered)
        self._inflight: dict[int, Future] = {}
        self._block_size = max(256 * 1024, chunk_cache_mb * 1024 * 1024)
        self._max_cache_blocks = max(8, _MAX_CACHE_BYTES // self._block_size)
        self._total_fetched = 0
        self._last_block = -10
        self._read_ahead = max(0, read_ahead)
        self._pool = ThreadPoolExecutor(max_workers=max(1, self._read_ahead)) if self._read_ahead else None
        self._resolve()

    # ------------------------------------------------------------------
    # URL / size discovery
    # ------------------------------------------------------------------

    def _request_headers(self, extra: dict) -> dict:
        headers = dict(extra)
        if self._strip_auth:
            # Pre-signed cloud URLs reject an extra Authorization header.
            headers["Authorization"] = None  # type: ignore[assignment]
        return headers

    def _resolve(self) -> None:
        """Follow redirects once; learn the final URL and the file size."""
        self._url = self._origin_url
        self._strip_auth = False
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self._session.get(
                    self._origin_url,
                    headers={"Range": "bytes=0-0"},
                    timeout=self._timeout,
                    stream=True,
                    allow_redirects=True,
                )
                try:
                    resp.raise_for_status()
                    content_range = resp.headers.get("Content-Range", "")
                    if "/" in content_range:
                        self._size = int(content_range.split("/")[-1])
                    elif resp.headers.get("Content-Length"):
                        self._size = int(resp.headers["Content-Length"])
                    final_url = resp.url
                    if urlparse(final_url).hostname != urlparse(self._origin_url).hostname:
                        self._url = final_url
                        self._strip_auth = True
                finally:
                    resp.close()
                if self._size:
                    return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(1)
        # Fall back to a HEAD request against the original URL.
        try:
            resp = self._session.head(self._origin_url, timeout=self._timeout, allow_redirects=True)
            resp.raise_for_status()
            length = resp.headers.get("Content-Length")
            self._size = int(length) if length else None
            self._url = self._origin_url
            self._strip_auth = False
        except Exception:
            raise last_exc or RuntimeError("Could not determine remote file size")

    # ------------------------------------------------------------------
    # Block fetching
    # ------------------------------------------------------------------

    def _block_bounds(self, block_index: int) -> tuple[int, int]:
        block_start = block_index * self._block_size
        block_end = block_start + self._block_size
        if self._size is not None:
            block_end = min(block_end, self._size)
        return block_start, block_end

    def _download_block(self, block_index: int) -> bytes:
        block_start, block_end = self._block_bounds(block_index)
        expected = block_end - block_start
        headers = {"Range": f"bytes={block_start}-{block_end - 1}"}
        last_exc: Optional[Exception] = None
        for attempt in range(6):
            try:
                resp = self._session.get(
                    self._url,
                    headers=self._request_headers(headers),
                    timeout=self._timeout,
                    stream=True,
                )
                try:
                    if resp.status_code in (400, 401, 403) and self._strip_auth and attempt < 5:
                        # Pre-signed URL likely expired: resolve again and retry.
                        resp.close()
                        self._resolve()
                        continue
                    if resp.status_code not in (200, 206):
                        resp.raise_for_status()
                    whole_file = block_start == 0 and self._size is not None and block_end >= self._size
                    if resp.status_code == 200 and not whole_file:
                        # The server ignored the Range header and is sending the whole
                        # (multi-GB) file. Reading that per block would exhaust memory,
                        # so fail fast without touching the body.
                        raise RangeNotSupportedError(
                            f"Server ignored the Range request for {urlparse(self._url).hostname}; "
                            "cannot stream this file"
                        )
                    body = resp.content
                finally:
                    resp.close()
                if len(body) != expected:
                    raise IOError(f"Truncated range response: got {len(body)} of {expected} bytes")
                with self._lock:
                    self._total_fetched += len(body)
                    total = self._total_fetched
                if self._progress_callback:
                    self._progress_callback(len(body), total)
                return body
            except RangeNotSupportedError:
                raise  # retrying would only start the full download again
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(min(2 ** attempt, 20))
        raise last_exc or RuntimeError("Failed to fetch range after 6 attempts")

    def _store(self, block_index: int, raw: bytes) -> None:
        with self._lock:
            while len(self._cache) >= self._max_cache_blocks:
                self._cache.pop(next(iter(self._cache)))
            self._cache[block_index] = raw

    def _prefetch_task(self, block_index: int) -> bytes:
        try:
            raw = self._download_block(block_index)
            self._store(block_index, raw)
            return raw
        finally:
            with self._lock:
                self._inflight.pop(block_index, None)

    def _schedule_read_ahead(self, block_index: int) -> None:
        if not self._pool or self._size is None:
            return
        last_block = (self._size - 1) // self._block_size
        for nxt in range(block_index + 1, min(block_index + 1 + self._read_ahead, last_block + 1)):
            with self._lock:
                if nxt in self._cache or nxt in self._inflight:
                    continue
                try:
                    self._inflight[nxt] = self._pool.submit(self._prefetch_task, nxt)
                except RuntimeError:  # pool already shut down
                    return

    def _get_block(self, block_index: int) -> bytes:
        with self._lock:
            raw = self._cache.get(block_index)
            pending = self._inflight.get(block_index)
            sequential = block_index in (self._last_block, self._last_block + 1)
            self._last_block = block_index
        if raw is None and pending is not None:
            try:
                raw = pending.result()
            except Exception:  # noqa: BLE001
                raw = None
        if raw is None:
            raw = self._download_block(block_index)
            self._store(block_index, raw)
        if sequential:
            self._schedule_read_ahead(block_index)
        return raw

    def _fetch_range(self, start: int, end: int) -> bytes:
        """Return exactly bytes [start, end), spanning as many blocks as needed."""
        parts: list[bytes] = []
        pos = start
        while pos < end:
            block_index = pos // self._block_size
            raw = self._get_block(block_index)
            offset = pos - block_index * self._block_size
            chunk = raw[offset: offset + (end - pos)]
            if not chunk:
                break
            parts.append(chunk)
            pos += len(chunk)
        return b"".join(parts)

    # ------------------------------------------------------------------
    # io.RawIOBase interface
    # ------------------------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seek(self, pos: int, whence: int = 0) -> int:
        with self._lock:
            if whence == 0:
                self._pos = pos
            elif whence == 1:
                self._pos += pos
            elif whence == 2:
                if self._size is None:
                    raise io.UnsupportedOperation("Cannot seek from end: file size unknown")
                self._pos = self._size + pos
            else:
                raise ValueError(f"Invalid whence value: {whence}")
            return self._pos

    def tell(self) -> int:
        with self._lock:
            return self._pos

    def readinto(self, b: bytearray) -> int:
        n = len(b)
        with self._lock:
            pos = self._pos
        if self._size is not None and pos >= self._size:
            return 0
        end = pos + n
        if self._size is not None:
            end = min(end, self._size)
        data = self._fetch_range(pos, end)
        actual = len(data)
        b[:actual] = data
        with self._lock:
            self._pos += actual
        return actual

    def read(self, n: int = -1) -> bytes:
        with self._lock:
            pos = self._pos
        if n == -1:
            if self._size is None:
                raise io.UnsupportedOperation("Cannot read all: file size unknown")
            n = self._size - pos
        if n <= 0 or (self._size is not None and pos >= self._size):
            return b""
        end = pos + n
        if self._size is not None:
            end = min(end, self._size)
        data = self._fetch_range(pos, end)
        with self._lock:
            self._pos += len(data)
        return data

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        super().close()

    @property
    def total_fetched_bytes(self) -> int:
        with self._lock:
            return self._total_fetched

    @property
    def file_size(self) -> Optional[int]:
        return self._size

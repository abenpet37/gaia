import unittest
from unittest.mock import MagicMock

from remote_hdf5 import RangeNotSupportedError, RemoteHDF5File


def _response(status, body=b"", headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.content = body
    resp.headers = headers or {}
    resp.url = "https://example.test/file.h5"
    return resp


class RemoteHDF5FileTests(unittest.TestCase):
    def _file(self, session):
        return RemoteHDF5File("https://example.test/file.h5", session, chunk_cache_mb=1, read_ahead=0)

    def test_range_request_returns_exact_bytes(self) -> None:
        data = bytes(range(256)) * 4096  # 1 MiB
        session = MagicMock()
        session.get.side_effect = [
            _response(206, b"\x00", {"Content-Range": f"bytes 0-0/{len(data)}"}),
            _response(206, data),
        ]
        f = self._file(session)
        self.assertEqual(f.read(16), data[:16])

    def test_server_ignoring_range_fails_fast_without_reading_body(self) -> None:
        size = 8 * 1024 * 1024
        first = _response(206, b"\x00", {"Content-Range": f"bytes 0-0/{size}"})
        ignored = _response(200, headers={"Content-Length": str(size)})
        body_reads = []
        ignored_type = type(ignored)
        ignored_type.content = property(lambda self: body_reads.append(1) or b"")
        session = MagicMock()
        session.get.side_effect = [first, ignored]
        f = self._file(session)
        f.seek(2 * 1024 * 1024)  # a block that is not the start of the file
        with self.assertRaises(RangeNotSupportedError):
            f.read(10)
        self.assertEqual(body_reads, [])
        self.assertEqual(session.get.call_count, 2)  # no retries


if __name__ == "__main__":
    unittest.main()

"""Tests for the InSAR Hazard Sentinel web server and API endpoints."""

import json
import unittest
from unittest.mock import patch

from server import app


class ServerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_token_status_endpoint(self) -> None:
        res = self.client.get("/api/token-status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("configured", data)
        self.assertIn("source", data)

    def test_save_token_requires_non_empty_string(self) -> None:
        res = self.client.post("/api/save-token", json={"token": ""})
        self.assertEqual(res.status_code, 400)

    @patch("server.clip_faults_to_bbox")
    def test_clip_faults_endpoint(self, mock_clip) -> None:
        mock_clip.return_value = 12
        res = self.client.post("/api/clip-faults", json={
            "west": 36.0,
            "south": 6.0,
            "east": 40.0,
            "north": 10.0,
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["clipped_faults_count"], 12)

    @patch("server.fetch_usgs_events")
    @patch("server.process_event_trigger")
    def test_trigger_search_endpoint(self, mock_process, mock_fetch) -> None:
        mock_fetch.return_value = [
            {
                "id": "mock_event",
                "properties": {"mag": 5.2, "place": "Afar, Ethiopia", "time": 1792065600000},
                "geometry": {"coordinates": [40.5, 12.0, 10.0]},
            }
        ]
        mock_process.return_value = {"event_id": "mock_event"}

        res = self.client.post("/api/trigger-search", json={
            "feed": "4.5_month",
            "min_mag": 4.5,
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["events_count"], 1)

    def test_root_serves_landing_page_and_map_route_redirects(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"GAIA", res.data)
        res.close()
        for route in ("/map", "/map/"):
            res = self.client.get(route)
            self.assertEqual(res.status_code, 302, route)
            self.assertEqual(res.headers["Location"], "/map/map.html")
        # Everything the page loads by relative URL must exist under /map/.
        for path in ("/map/map.html", "/map/tectonic_terrain.html", "/map/map-state.js",
                     "/trans/deformation-overlay.js"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_job_status_unknown(self) -> None:
        res = self.client.get("/api/job-status/nonexistent-id")
        self.assertEqual(res.status_code, 404)

    def test_job_lifecycle(self) -> None:
        from server import _new_job, _push_progress
        job_id = _new_job()
        res = self.client.get(f"/api/job-status/{job_id}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["pct"], 0)

        _push_progress(job_id, "downloading", 45, "Fetching chunks...")
        res2 = self.client.get(f"/api/job-status/{job_id}")
        data2 = res2.get_json()
        self.assertEqual(data2["pct"], 45)
        self.assertEqual(data2["status"], "downloading")
        self.assertIn("Fetching chunks...", data2["log"])


    def test_finished_jobs_are_pruned_and_logs_capped(self) -> None:
        import time
        import server

        old_id = server._new_job()
        server._finish_job(old_id, "done", 100, "finished long ago")
        server._jobs[old_id]["finished_at"] = time.time() - server.JOB_RETENTION_SECONDS - 5
        server._new_job()  # creating a job prunes expired ones
        self.assertNotIn(old_id, server._jobs)
        self.assertNotIn(old_id, server._job_queues)

        job_id = server._new_job()
        for i in range(server.MAX_JOB_LOG_LINES + 25):
            server._push_progress(job_id, "work", 1, f"line {i}")
        self.assertEqual(len(server._jobs[job_id]["log"]), server.MAX_JOB_LOG_LINES)
        self.assertEqual(server._jobs[job_id]["log"][-1], f"line {server.MAX_JOB_LOG_LINES + 24}")

    def test_finish_job_marks_done_and_notifies_once(self) -> None:
        import server

        job_id = server._new_job()
        server._finish_job(job_id, "error", 0, "boom", error="boom")
        job = self.client.get(f"/api/job-status/{job_id}").get_json()
        self.assertTrue(job["done"])
        self.assertEqual(job["error"], "boom")
        self.assertEqual(server._job_queues[job_id].qsize(), 1)

    @patch("server.sort_products_by_proximity", side_effect=lambda pair, bbox: pair)
    @patch("server.latest_acquisition_pair", side_effect=lambda found: found)
    @patch("server.search_products")
    def test_select_products_processes_every_granule_unless_capped(self, mock_search, _pair, _sort) -> None:
        import server

        mock_search.return_value = [{"fileName": f"g{i}.h5"} for i in range(4)]
        base = {"west": 36, "south": 6, "east": 40, "north": 10}

        products, available = server._select_products(base)
        self.assertEqual((len(products), available), (4, 4))

        products, available = server._select_products({**base, "max_granules": "all"})
        self.assertEqual(len(products), 4)

        products, available = server._select_products({**base, "max_granules": 2})
        self.assertEqual((len(products), available), (2, 4))

    @patch("server.search_products", return_value=[])
    def test_select_products_reports_empty_search(self, _search) -> None:
        import server

        with self.assertRaises(LookupError):
            server._select_products({"west": 36, "south": 6, "east": 40, "north": 10})

    @patch("server.threading.Thread")
    @patch("server._earthdata_token", return_value="token")
    @patch("server._select_products")
    def test_both_processing_routes_share_one_implementation(self, mock_select, _token, _thread) -> None:
        mock_select.return_value = ([{"fileName": "a.h5"}], 3)
        for route in ("/api/remote-process", "/api/download-and-process"):
            res = self.client.post(route, json={"west": 1, "south": 1, "east": 2, "north": 2})
            self.assertEqual(res.status_code, 202)
            body = res.get_json()
            self.assertEqual(body["granules"], 1)
            self.assertTrue(body["truncated"])
            self.assertEqual(body["available_granules"], 3)


if __name__ == "__main__":
    unittest.main()

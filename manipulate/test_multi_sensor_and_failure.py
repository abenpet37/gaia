"""Unit tests for Multi-Sensor Fusion, Inverse Velocity failure prediction, and API adapters."""

from datetime import datetime, timedelta
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sat-data"))

from multi_sensor_fusion import calculate_hydro_risk, calculate_tectonic_risk
from inverse_velocity import predict_time_to_failure
from api_connectors import (
    ASFApiConnector,
    GPMImergConnector,
    SMAPSoilMoistureConnector,
    USGSEarthquakeConnector,
    TROPOMISo2Connector,
)


class MultiSensorFusionTests(unittest.TestCase):
    def test_hydro_risk_levels(self) -> None:
        # Green baseline
        green = calculate_hydro_risk(creep_velocity_mm_week=1.2, gpm_rainfall_72h_mm=10.0, smap_saturation_pct=40.0)
        self.assertEqual(green.risk_level, "GREEN")
        self.assertLess(green.total_score, 30)

        # Yellow monitor
        yellow = calculate_hydro_risk(creep_velocity_mm_week=8.5, gpm_rainfall_72h_mm=25.0, smap_saturation_pct=55.0)
        self.assertEqual(yellow.risk_level, "YELLOW")
        self.assertTrue(30 <= yellow.total_score < 55)

        # Orange watch
        orange = calculate_hydro_risk(creep_velocity_mm_week=14.0, gpm_rainfall_72h_mm=85.0, smap_saturation_pct=75.0)
        self.assertEqual(orange.risk_level, "ORANGE")
        self.assertTrue(55 <= orange.total_score < 80)

        # Red critical evacuation
        red = calculate_hydro_risk(creep_velocity_mm_week=25.0, gpm_rainfall_72h_mm=180.0, smap_saturation_pct=92.0)
        self.assertEqual(red.risk_level, "RED")
        self.assertGreaterEqual(red.total_score, 80)
        self.assertIn("CRITICAL", red.summary)

    def test_tectonic_risk_levels(self) -> None:
        # Green baseline
        green = calculate_tectonic_risk(extension_rate_mm_week=3.0, usgs_eq_count_7d=2, usgs_max_mag_7d=2.0, tropomi_so2_du=0.3)
        self.assertEqual(green.risk_level, "GREEN")
        self.assertLess(green.total_score, 30)

        # Red critical volcanic/dike warning
        red = calculate_tectonic_risk(extension_rate_mm_week=85.0, usgs_eq_count_7d=65, usgs_max_mag_7d=5.1, tropomi_so2_du=18.5)
        self.assertEqual(red.risk_level, "RED")
        self.assertGreaterEqual(red.total_score, 80)
        self.assertIn("MAGMATIC", red.summary)


class InverseVelocityTests(unittest.TestCase):
    def test_accelerating_tertiary_creep_forecast(self) -> None:
        # Accelerating displacement curve towards collapse
        # t: 0, 7, 14, 21, 28 days
        # displacements: 5, 12, 28, 75, 250 mm
        base = datetime(2024, 7, 1)
        dates = [base + timedelta(days=7 * i) for i in range(5)]
        displacements = [5.0, 12.0, 28.0, 75.0, 250.0]

        pred = predict_time_to_failure(dates, displacements)
        self.assertTrue(pred.accelerating)
        self.assertGreater(pred.r_squared, 0.65)
        self.assertLess(pred.slope_trend, 0.0)  # 1/v decreasing
        self.assertIsNotNone(pred.hours_to_failure_min)
        self.assertIsNotNone(pred.hours_to_failure_max)
        self.assertIn("hours", pred.forecast_window_text.lower())

    def test_stable_linear_creep_no_failure(self) -> None:
        # Constant linear velocity (no acceleration)
        base = datetime(2024, 7, 1)
        dates = [base + timedelta(days=7 * i) for i in range(5)]
        displacements = [10.0, 20.0, 30.0, 40.0, 50.0]

        pred = predict_time_to_failure(dates, displacements)
        self.assertFalse(pred.accelerating)
        self.assertIn("stable", pred.forecast_window_text.lower())


class ApiConnectorsTests(unittest.TestCase):
    def test_asf_connector_offline(self) -> None:
        conn = ASFApiConnector()
        res = conn.search_granules((36.0, 6.0, 37.0, 7.0), "2024-05-01", "2024-07-25", platform="NISAR")
        self.assertTrue(len(res) > 0)
        self.assertIn("granuleName", res[0])

    def test_gpm_and_smap_connectors(self) -> None:
        gpm = GPMImergConnector()
        rain = gpm.get_precipitation(6.315, 36.848)
        self.assertGreater(rain["accumulation_72h_mm"], 50.0)

        smap = SMAPSoilMoistureConnector()
        soil = smap.get_soil_saturation(6.315, 36.848)
        self.assertGreater(soil["saturation_pct"], 80.0)

    def test_usgs_and_tropomi_connectors(self) -> None:
        usgs = USGSEarthquakeConnector()
        events = usgs.query_events(13.0, 14.5, 40.0, 41.5)
        self.assertTrue(len(events) > 0)

        tropomi = TROPOMISo2Connector()
        so2 = tropomi.get_so2_column(13.51, 40.72)
        self.assertEqual(so2["flag"], "MAGMA_DEGASSING_SPIKE")


if __name__ == "__main__":
    unittest.main()

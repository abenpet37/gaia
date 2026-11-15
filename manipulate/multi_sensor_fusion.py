"""Multi-Sensor Trigger Fusion Engine for InSAR Natural Hazard Warning.

Combines InSAR millimeter ground deformation velocity with external triggers
to filter out false alarms and provide robust decision support:
1. Hydro-Geological Mode (Landslides/Mudslides):
   - InSAR slope creep (mm/week)
   - NASA GPM IMERG 72h cumulative rainfall (mm)
   - NASA SMAP soil moisture saturation (%)
2. Tectonic/Volcanic Mode (Rifting/Eruptions):
   - InSAR crustal opening / magma dike inflation (mm/week)
   - USGS Earthquake swarm intensity (count & max magnitude)
   - Sentinel-5P TROPOMI SO2 degassing column density (Dobson Units)

Calculates weighted Risk Scores (0-100) categorized into:
- GREEN (Baseline, 0-29): Nominal background motion.
- YELLOW (Monitor, 30-54): Elevated InSAR motion without critical secondary triggers.
- ORANGE (Watch, 55-79): Significant ground creep coupled with rising hydrological/seismic triggers.
- RED (Critical Evacuation, 80-100): Accelerating InSAR failure curve + critical trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RiskBreakdown:
    total_score: int
    risk_level: Literal["GREEN", "YELLOW", "ORANGE", "RED"]
    mode: Literal["hydro", "tectonic"]
    insar_points: float
    trigger1_points: float
    trigger2_points: float
    summary: str
    action: str
    details: dict[str, float | str | int]


def calculate_hydro_risk(
    creep_velocity_mm_week: float,
    gpm_rainfall_72h_mm: float,
    smap_saturation_pct: float,
) -> RiskBreakdown:
    """Calculate multi-sensor landslide/mudslide risk score (0-100).

    Weights:
      InSAR Creep: 45 pts (0-5 mm/wk: 0-15; 5-15 mm/wk: 15-30; >15 mm/wk: 30-45)
      GPM 72h Rainfall: 30 pts (<50 mm: 0-10; 50-120 mm: 10-22; >120 mm: 22-30)
      SMAP Saturation: 25 pts (<60%: 0-8; 60-80%: 8-18; >80%: 18-25)
    """
    # InSAR Score (max 45)
    if creep_velocity_mm_week <= 2.0:
        insar_pts = (creep_velocity_mm_week / 2.0) * 10.0
    elif creep_velocity_mm_week <= 10.0:
        insar_pts = 10.0 + ((creep_velocity_mm_week - 2.0) / 8.0) * 20.0
    else:
        insar_pts = min(45.0, 30.0 + ((creep_velocity_mm_week - 10.0) / 15.0) * 15.0)

    # GPM Rainfall Score (max 30)
    if gpm_rainfall_72h_mm <= 30.0:
        rain_pts = (gpm_rainfall_72h_mm / 30.0) * 8.0
    elif gpm_rainfall_72h_mm <= 100.0:
        rain_pts = 8.0 + ((gpm_rainfall_72h_mm - 30.0) / 70.0) * 14.0
    else:
        rain_pts = min(30.0, 22.0 + ((gpm_rainfall_72h_mm - 100.0) / 80.0) * 8.0)

    # SMAP Saturation Score (max 25)
    if smap_saturation_pct <= 50.0:
        smap_pts = (smap_saturation_pct / 50.0) * 6.0
    elif smap_saturation_pct <= 80.0:
        smap_pts = 6.0 + ((smap_saturation_pct - 50.0) / 30.0) * 12.0
    else:
        # Extreme saturation (>85%) drastically compromises shear strength
        smap_pts = min(25.0, 18.0 + ((smap_saturation_pct - 80.0) / 15.0) * 7.0)

    # Critical trigger synergy boost
    total_raw = insar_pts + rain_pts + smap_pts
    if creep_velocity_mm_week > 12.0 and smap_saturation_pct > 85.0 and gpm_rainfall_72h_mm > 100.0:
        # Imminent fluidization condition
        total_raw = max(88.0, total_raw + 10.0)

    score = int(round(min(100.0, max(0.0, total_raw))))

    if score >= 80:
        level: Literal["GREEN", "YELLOW", "ORANGE", "RED"] = "RED"
        summary = "CRITICAL SLOPE COLLAPSE HAZARD: Active InSAR creep combined with saturated highland soil."
        action = "Issue Immediate Evacuation Alert for valley floor and debris flow path."
    elif score >= 55:
        level = "ORANGE"
        summary = "HIGH SLOPE INSTABILITY WATCH: Sustained creep with heavy rainfall infiltration."
        action = "Mobilize civil protection, monitor 1/v inverse velocity trend closely."
    elif score >= 30:
        level = "YELLOW"
        summary = "ELEVATED CREEP DETECTED: Ground motion observed without triggering rainfall."
        action = "Maintain automated L-band InSAR radar tracking."
    else:
        level = "GREEN"
        summary = "NOMINAL BASELINE: Ground creep within natural seasonal thresholds."
        action = "Routine automated satellite surveillance."

    return RiskBreakdown(
        total_score=score,
        risk_level=level,
        mode="hydro",
        insar_points=round(insar_pts, 1),
        trigger1_points=round(rain_pts, 1),
        trigger2_points=round(smap_pts, 1),
        summary=summary,
        action=action,
        details={
            "creep_velocity_mm_week": creep_velocity_mm_week,
            "gpm_rainfall_72h_mm": gpm_rainfall_72h_mm,
            "smap_saturation_pct": smap_saturation_pct,
        },
    )


def calculate_tectonic_risk(
    extension_rate_mm_week: float,
    usgs_eq_count_7d: int,
    usgs_max_mag_7d: float,
    tropomi_so2_du: float,
) -> RiskBreakdown:
    """Calculate multi-sensor rifting/volcanic eruption risk score (0-100).

    Weights:
      InSAR Extension / Dike Inflation: 45 pts
      USGS Earthquake Swarm: 30 pts (count & magnitude)
      Sentinel-5P TROPOMI SO2: 25 pts (magmatic degassing indicator)
    """
    # InSAR Extension (max 45)
    if extension_rate_mm_week <= 5.0:
        insar_pts = (extension_rate_mm_week / 5.0) * 12.0
    elif extension_rate_mm_week <= 25.0:
        insar_pts = 12.0 + ((extension_rate_mm_week - 5.0) / 20.0) * 20.0
    else:
        insar_pts = min(45.0, 32.0 + ((extension_rate_mm_week - 25.0) / 35.0) * 13.0)

    # Seismic Swarm (max 30)
    count_pts = min(15.0, (usgs_eq_count_7d / 40.0) * 15.0)
    mag_pts = 0.0 if usgs_max_mag_7d < 2.5 else min(15.0, ((usgs_max_mag_7d - 2.5) / 3.0) * 15.0)
    eq_pts = count_pts + mag_pts

    # SO2 Degassing (max 25)
    if tropomi_so2_du <= 1.0:
        so2_pts = (tropomi_so2_du / 1.0) * 4.0
    elif tropomi_so2_du <= 5.0:
        so2_pts = 4.0 + ((tropomi_so2_du - 1.0) / 4.0) * 11.0
    else:
        so2_pts = min(25.0, 15.0 + ((tropomi_so2_du - 5.0) / 15.0) * 10.0)

    total_raw = insar_pts + eq_pts + so2_pts
    # Magma intrusion synergy boost
    if extension_rate_mm_week > 30.0 and (tropomi_so2_du > 8.0 or usgs_eq_count_7d > 35):
        total_raw = max(86.0, total_raw + 8.0)

    score = int(round(min(100.0, max(0.0, total_raw))))

    if score >= 80:
        level: Literal["GREEN", "YELLOW", "ORANGE", "RED"] = "RED"
        summary = "CRITICAL MAGMATIC INTRUSION / ERUPTION WARNING: Subsurface dike propagation with SO2 plume spike."
        action = "Issue Regional Civil Aviation Sigmet and Pastoral Settlement Evacuation Order."
    elif score >= 55:
        level = "ORANGE"
        summary = "VOLCANIC/RIFTING UNREST WATCH: Rapid crustal dilation and seismic swarm detected."
        action = "Task daily radar passes, alert local geological observatory."
    elif score >= 30:
        level = "YELLOW"
        summary = "CRUSTAL SWELL DETECTED: Aseismic extension above baseline rifting rate."
        action = "Maintain automated InSAR phase monitoring."
    else:
        level = "GREEN"
        summary = "NOMINAL TECTONIC BASELINE: Normal plate divergence rate."
        action = "Routine regional background monitoring."

    return RiskBreakdown(
        total_score=score,
        risk_level=level,
        mode="tectonic",
        insar_points=round(insar_pts, 1),
        trigger1_points=round(eq_pts, 1),
        trigger2_points=round(so2_pts, 1),
        summary=summary,
        action=action,
        details={
            "extension_rate_mm_week": extension_rate_mm_week,
            "usgs_eq_count_7d": usgs_eq_count_7d,
            "usgs_max_mag_7d": usgs_max_mag_7d,
            "tropomi_so2_du": tropomi_so2_du,
        },
    )

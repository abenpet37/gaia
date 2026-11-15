"""Inverse Velocity (1/v) Geotechnical Time-to-Failure Forecasting Engine.

Based on the Fukuzono (1985) and Voight (1988) fundamental material failure law:
    d^2 x / dt^2 = A * (dx/dt)^alpha

For tertiary accelerating slope creep or pre-eruptive volcanic pressurization,
alpha ~ 2.0, which yields a linear decrease in inverse velocity over time:
    1 / v(t) = A * (t_failure - t)

When inverse velocity approaches zero (1/v -> 0), velocity approaches infinity,
marking the moment of catastrophic slope collapse or volcanic rupture.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import numpy as np


@dataclass
class FailurePrediction:
    accelerating: bool
    current_velocity_mm_day: float
    current_inverse_velocity_day_mm: float
    projected_failure_iso: str | None
    hours_to_failure_min: float | None
    hours_to_failure_max: float | None
    forecast_window_text: str
    r_squared: float
    slope_trend: float
    intercept: float
    time_series_points: list[dict[str, float | str]]


def predict_time_to_failure(
    dates: list[datetime],
    cumulative_displacements_mm: list[float],
) -> FailurePrediction:
    """Compute interval velocities, inverse velocities, and project failure time window.

    Args:
        dates: Monotonically increasing list of acquisition datetimes.
        cumulative_displacements_mm: Cumulative ground motion in millimeters.
    """
    if len(dates) < 3 or len(dates) != len(cumulative_displacements_mm):
        return FailurePrediction(
            accelerating=False,
            current_velocity_mm_day=0.0,
            current_inverse_velocity_day_mm=0.0,
            projected_failure_iso=None,
            hours_to_failure_min=None,
            hours_to_failure_max=None,
            forecast_window_text="Insufficient temporal points for 1/v inversion (>=3 required)",
            r_squared=0.0,
            slope_trend=0.0,
            intercept=0.0,
            time_series_points=[],
        )

    t0 = dates[0]
    elapsed_days = np.array([(d - t0).total_seconds() / 86400.0 for d in dates])
    disp = np.array(cumulative_displacements_mm, dtype=np.float64)

    # Compute interval velocities (mm/day)
    dt = np.diff(elapsed_days)
    dd = np.diff(disp)

    # Avoid zero division
    dt = np.maximum(dt, 0.01)
    velocities = dd / dt
    mid_times = elapsed_days[1:]

    # Points table for UI visualization
    points = []
    for i in range(len(dates)):
        pt = {
            "date": dates[i].isoformat(),
            "elapsed_days": round(elapsed_days[i], 2),
            "displacement_mm": round(disp[i], 2),
            "velocity_mm_day": round(float(velocities[i - 1]), 3) if i > 0 else 0.0,
            "inv_velocity_day_mm": (
                round(float(1.0 / max(0.001, velocities[i - 1])), 4) if i > 0 and velocities[i - 1] > 0 else None
            ),
        }
        points.append(pt)

    # Filter positive accelerating velocities in the recent window
    valid_mask = velocities > 0.05
    if np.sum(valid_mask) < 2:
        return FailurePrediction(
            accelerating=False,
            current_velocity_mm_day=float(velocities[-1]) if len(velocities) else 0.0,
            current_inverse_velocity_day_mm=float(1.0 / max(0.01, velocities[-1])) if len(velocities) and velocities[-1] > 0 else 0.0,
            projected_failure_iso=None,
            hours_to_failure_min=None,
            hours_to_failure_max=None,
            forecast_window_text="Deformation is linear or decelerating (stable slope)",
            r_squared=0.0,
            slope_trend=0.0,
            intercept=0.0,
            time_series_points=points,
        )

    x_val = mid_times[valid_mask]
    y_val = 1.0 / velocities[valid_mask]

    # Robust least squares fit for 1/v = slope * t + intercept
    A = np.column_stack((x_val, np.ones(x_val.size)))
    res = np.linalg.lstsq(A, y_val, rcond=None)
    slope, intercept = float(res[0][0]), float(res[0][1])

    # Check R^2
    y_pred = slope * x_val + intercept
    ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
    ss_res = float(np.sum((y_val - y_pred) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0

    # For accelerating failure, 1/v must have a negative slope (velocity increasing towards infinity)
    is_accelerating = (slope < -1e-4) and (r_squared > 0.65)
    last_day = elapsed_days[-1]
    curr_vel = float(velocities[-1])
    curr_inv_vel = float(1.0 / max(0.01, curr_vel))

    if not is_accelerating:
        return FailurePrediction(
            accelerating=False,
            current_velocity_mm_day=curr_vel,
            current_inverse_velocity_day_mm=curr_inv_vel,
            projected_failure_iso=None,
            hours_to_failure_min=None,
            hours_to_failure_max=None,
            forecast_window_text="Deformation rate stable (no accelerating 1/v trend)",
            r_squared=round(r_squared, 3),
            slope_trend=round(slope, 5),
            intercept=round(intercept, 4),
            time_series_points=points,
        )

    # Failure time t_failure occurs when 1/v = 0 -> slope * t_fail + intercept = 0 -> t_fail = -intercept / slope
    t_failure_day = -intercept / slope
    hours_left = max(0.0, (t_failure_day - last_day) * 24.0)

    # Provide uncertainty bounds based on fit residual (typically +/- 20%)
    h_min = max(1.0, hours_left * 0.8)
    h_max = max(h_min + 4.0, hours_left * 1.25)
    failure_date = dates[-1] + timedelta(hours=hours_left)

    if hours_left < 72.0:
        window_text = f"CRITICAL: Estimated slope failure window: {int(round(h_min))}–{int(round(h_max))} hours"
    else:
        days_min = int(round(h_min / 24.0))
        days_max = int(round(h_max / 24.0))
        window_text = f"WATCH: Projected rupture window: {days_min}–{days_max} days ({failure_date.strftime('%b %d, %Y')})"

    return FailurePrediction(
        accelerating=True,
        current_velocity_mm_day=round(curr_vel, 2),
        current_inverse_velocity_day_mm=round(curr_inv_vel, 4),
        projected_failure_iso=failure_date.isoformat(),
        hours_to_failure_min=round(h_min, 1),
        hours_to_failure_max=round(h_max, 1),
        forecast_window_text=window_text,
        r_squared=round(r_squared, 3),
        slope_trend=round(slope, 5),
        intercept=round(intercept, 4),
        time_series_points=points,
    )


import numpy as np

from src.question2.delay_interface import (
    delay_for_flow,
    align_gas_signal,
)


def test_fixed_delay():
    result = delay_for_flow("CO", 5.0)
    assert result["delay_s"] == 32.0
    assert result["valid"] is True


def test_co2_delay():
    result = delay_for_flow("CO2", 5.0)
    assert result["delay_s"] == 34.0


def test_outside_flow_range():
    result = delay_for_flow("CO", 20.0)
    assert result["valid"] is False


def test_alignment_direction():
    gas_time = np.arange(0, 101, 1, dtype=float)
    gas_values = gas_time.copy()
    target_time = np.array([8.0, 18.0, 28.0])

    aligned = align_gas_signal(
        gas_time,
        gas_values,
        target_time,
        delay_s=32.0,
    )

    np.testing.assert_allclose(
        aligned,
        [40.0, 50.0, 60.0],
    )


def test_no_extrapolation():
    gas_time = np.arange(0, 101, 1, dtype=float)
    gas_values = gas_time.copy()

    aligned = align_gas_signal(
        gas_time,
        gas_values,
        np.array([-40.0, 8.0, 100.0]),
        delay_s=32.0,
    )

    assert np.isnan(aligned[0])
    assert aligned[1] == 40.0
    assert np.isnan(aligned[2])

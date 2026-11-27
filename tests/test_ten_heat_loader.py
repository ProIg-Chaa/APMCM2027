from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


QUESTION2_SOURCE = Path(__file__).resolve().parents[1] / "src" / "question2"
sys.path.insert(0, str(QUESTION2_SOURCE))

import ten_heat_reconstruction_delay as pipeline


@pytest.mark.parametrize("sequence_name", ["序列", "Sequence", "序列Sequence"])
def test_load_heat_accepts_sequence_aliases(monkeypatch, tmp_path, sequence_name):
    frame = pd.DataFrame({
        sequence_name: [1, 2, 3],
        "Cumulative oxygen consumption": [0.0, 1.0, 2.0],
        "CO₂ concentration": [0.1, 0.2, 0.3],
        "CO concentration": [0.4, 0.5, 0.6],
    })
    monkeypatch.setattr(pipeline.pd, "read_excel", lambda *args, **kwargs: frame)

    time, signals = pipeline.load_heat(tmp_path / "heat.xlsb")

    np.testing.assert_allclose(time, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(signals["oxygen"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(signals["co2"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(signals["co"], [0.4, 0.5, 0.6])


def test_load_heat_rejects_nonconsecutive_sequence(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "Sequence": [1, 3, 4],
        "Cumulative oxygen consumption": [0.0, 1.0, 2.0],
        "CO₂ concentration": [0.1, 0.2, 0.3],
        "CO concentration": [0.4, 0.5, 0.6],
    })
    monkeypatch.setattr(pipeline.pd, "read_excel", lambda *args, **kwargs: frame)

    with pytest.raises(ValueError, match="采样序号必须有限且连续"):
        pipeline.load_heat(tmp_path / "heat.xlsb")

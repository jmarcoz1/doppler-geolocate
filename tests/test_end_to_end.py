"""Full synthetic pipeline (audio synthesis -> ridges -> locate). ~2-3 min.

Run with:  RUN_SLOW=1 pytest tests/test_end_to_end.py
"""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


def test_synthetic_levc(tmp_path):
    from dopplerloc import demo
    s = demo.run(tmp_path, verbose=False)
    assert s["error_m"] < 100.0
    assert abs(s["estimate"]["clock_offset_s"] - s["truth"]["clock_offset_s"]) < 0.3

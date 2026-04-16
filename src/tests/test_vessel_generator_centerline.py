import numpy as np

from src.data_gen.lib.vessel_generator import _centerline_arc


def test_centerline_arc_allows_upward_and_downward_bends():
    n = 25
    length = 0.03
    angle_span = np.deg2rad(80.0)

    pts_down, _ = _centerline_arc(n=n, length=length, angle_span=angle_span, bend_sign=1.0)
    pts_up, _ = _centerline_arc(n=n, length=length, angle_span=angle_span, bend_sign=-1.0)

    # Endpoints stay to the right while allowing opposite vertical offsets.
    assert pts_down[-1, 0] > 0.0
    assert pts_up[-1, 0] > 0.0
    assert pts_down[-1, 1] < 0.0
    assert pts_up[-1, 1] > 0.0
